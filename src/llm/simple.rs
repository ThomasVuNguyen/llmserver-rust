use actix::Actor;
use hf_hub::api::sync::Api;
use rkllm_rs::prelude::*;
use serde::Deserialize;
use serde_variant::to_variant_name;
use std::ffi::CString;
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;

use autotokenizer::AutoTokenizer;
use autotokenizer::DefaultPromptMessage;

use crate::AIModel;
use crate::ProcessMessages;
use crate::ShutdownMessages;
use crate::LLM;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SimpleLLMConfig {
    pub modle_path: String,
    pub modle_name: String,
    pub cache_path: Option<String>,
    pub think: bool,
    #[serde(default = "default_legacy")]
    pub legacy: bool,
}

fn default_legacy() -> bool {
    true
}

#[derive(Debug)]
pub struct SimpleRkLLM {
    handle: LLMHandle,
    atoken: AutoTokenizer,
    infer_params: RKLLMInferParam,
    config: SimpleLLMConfig,
}

impl Actor for SimpleRkLLM {
    type Context = actix::Context<Self>;
}

impl actix::Handler<ProcessMessages> for SimpleRkLLM {
    type Result = Result<Pin<Box<dyn futures::Stream<Item = String> + Send + 'static>>, ()>;

    fn handle(&mut self, msg: ProcessMessages, _ctx: &mut Self::Context) -> Self::Result {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let atoken = self.atoken.clone();
        let prompt = msg
            .messages
            .iter()
            .map(|a| {
                let content = match &a.content {
                    Some(crate::Content::String(s)) => s,
                    Some(crate::Content::Array(items)) => &items.join(""),
                    None => "", // 老實說不應該發生
                };
                DefaultPromptMessage::new(to_variant_name(&a.role).unwrap(), &content)
            })
            .collect::<Vec<_>>();

        let mut input = match atoken.apply_chat_template(prompt, true) {
            Ok(parsed) => parsed,
            Err(_) => {
                println!("apply_chat_template failed.");
                "".to_owned()
            }
        };
        // TODO: 用參數判斷要不要think
        if !self.config.think {
            input += "\n\n</think>\n\n";
        }

        let handle = self.handle.clone();
        let infer_params_cloned = self.infer_params.clone();
        actix_web::rt::spawn(async move {
            let cb = CallbackSendSelfChannel { sender: Some(tx) };
            // TODO: Maybe someday should have good error handling
            let _ = handle.run(RKLLMInput::Prompt(input), Some(infer_params_cloned), cb);
        });

        // 將 Receiver 轉換為 Stream
        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

impl actix::Handler<ShutdownMessages> for SimpleRkLLM {
    type Result = Result<(), ()>;

    fn handle(&mut self, _: ShutdownMessages, _: &mut Self::Context) -> Self::Result {
        // TODO: Maybe someday should have good error handling
        let _ = self.handle.destroy();
        Ok(())
    }
}

impl AIModel for SimpleRkLLM {
    type Config = SimpleLLMConfig;
    fn init(config: &SimpleLLMConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Set environment variable for sentencepiece to find the correct library
        std::env::set_var("LD_LIBRARY_PATH", "/usr/local/lib:".to_string() + &std::env::var("LD_LIBRARY_PATH").unwrap_or_default());
        
        let mut param = RKLLMParam {
            ..Default::default()
        };
        
        // Model loading with better error handling
        let api = Api::new().map_err(|e| format!("Failed to initialize HF API: {}", e))?;
        let repo = api.model(config.modle_path.clone());
        let binding = repo.get("model.rkllm").map_err(|e| format!("Failed to get model file: {}", e))?;
        let modle_path = binding.to_string_lossy();
        let c_str = CString::new(modle_path.as_ref()).unwrap();
        param.model_path = c_str.as_ptr();

        // Try to initialize the model with custom error handling
        let handle = match rkllm_init(&mut param) {
            Ok(h) => h,
            Err(e) => {
                // If the error mentions "missing field `legacy`", try to work around it
                if e.to_string().contains("missing field `legacy`") {
                    println!("Warning: Detected 'missing field legacy' error, attempting to continue anyway");
                    // Try to initialize with a different approach or return a fallback
                    return Err("Model initialization failed due to a missing 'legacy' field in the model config. This may be due to a version mismatch between your model and the rkllm-rs library. Try using a different model or updating the rkllm-rs dependency.".into());
                } else {
                    return Err(e);
                }
            }
        };
        
        // Initialize tokenizer with custom error handling
        let atoken = match AutoTokenizer::from_pretrained(config.modle_path.clone(), None) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                if e.to_string().contains("missing field `legacy`") {
                    println!("Warning: Detected missing field 'legacy' error in tokenizer initialization");
                    eprintln!("This error is likely due to a mismatch between your model configuration ");
                    eprintln!("and the autotokenizer library after adding tool calling support.");
                    eprintln!("\nPossible solutions:\n");
                    eprintln!("1. Try a different model that's compatible with your current server version");
                    eprintln!("2. Update the rkllm-rs and autotokenizer crates to versions that support your model");
                    eprintln!("3. Downgrade your server code to a version before adding tool calling support");
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, 
                        "Tokenizer initialization failed: model config is incompatible with current autotokenizer version")));
                } else {
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, 
                        format!("Tokenizer initialization failed: {}", e))));
                }
            }
        };

        let infer_params = RKLLMInferParam {
            mode: RKLLMInferMode::InferGenerate,
            lora_params: None,
            prompt_cache_params: if let Some(cache_path) = &config.cache_path {
                Some(RKLLMPromptCacheParam {
                    save_prompt_cache: true,
                    prompt_cache_path: cache_path.to_owned(),
                })
            } else {
                None
            },
        };

        Ok(SimpleRkLLM {
            handle,
            atoken,
            infer_params,
            config: config.clone(),
        })
    }
}

impl LLM for SimpleRkLLM {}

struct CallbackSendSelfChannel {
    sender: Option<tokio::sync::mpsc::Sender<String>>,
}
impl RkllmCallbackHandler for CallbackSendSelfChannel {
    fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState) {
        match state {
            LLMCallState::Normal => {
                if let Some(result) = result {
                    if let Some(sender) = &self.sender {
                        while sender.try_send(result.text.clone()).is_err() {
                            std::thread::yield_now();
                        }
                    }
                }
            }
            LLMCallState::Waiting => {}
            LLMCallState::Finish => {
                drop(self.sender.take());
            }
            LLMCallState::Error => {}
            LLMCallState::GetLastHiddenLayer => {}
        }
    }
}

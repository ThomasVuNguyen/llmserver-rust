use actix::{Actor, Recipient};
use clap::{Arg, ArgAction, Command};
use serde::Deserialize;
use std::{collections::HashMap, fs::File, io::BufReader, net::Ipv4Addr, path::Path};

use actix_web::{head, middleware::Logger, App, HttpServer, Result};
use llmserver_rs::{
    asr::simple::SimpleASRConfig, huggingface::{check_model_exists, create_config_file, determine_model_type, ModelType},
    llm::simple::SimpleLLMConfig, AIModel, ProcessAudio, ProcessMessages, ShutdownMessages,
};
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

/// Get health of the API.
#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = str, content_type = "text/plain")
    )
)]
#[head("/health")]
async fn health() -> &'static str {
    ""
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    const VERSION: &str = env!("CARGO_PKG_VERSION");
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let matches = Command::new("rkllm")
        .about("Hugging Face model server")
        .version(VERSION)
        .arg_required_else_help(true)
        .arg(Arg::new("model_name"))
        .arg(
            Arg::new("instances")
                .short('i')
                .help("How many model instances do you want to create.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .get_matches();

    // Initialize model
    let mut num_instances = 1;

    if let Some(value) = matches.get_one::<usize>("instances") {
        num_instances = *value;
    }
    let model_id = matches.get_one::<String>("model_name").unwrap();

    // Check if model exists on Hugging Face
    if !check_model_exists(model_id) {
        panic!("Model {} does not exist or is not accessible on Hugging Face", model_id);
    }

    // Determine model type
    let model_type = determine_model_type(model_id)
        .unwrap_or_else(|| panic!("Could not determine model type for {}", model_id));

    // Create config file if it doesn't exist
    let parts: Vec<&str> = model_id.split('/').collect();
    let model_name = if parts.len() == 2 { parts[1] } else { model_id };
    
    let config_file_name = format!("assets/config/{}.json", model_name.to_lowercase().replace('-', "_"));
    if !Path::new(&config_file_name).exists() {
        println!("Creating config file for model: {}", model_id);
        let config_path = create_config_file(model_id, model_type)?;
        println!("Created config file: {}", config_path);
    }

    // Text type LLM
    let mut llm_recipients = HashMap::<String, Vec<Recipient<ProcessMessages>>>::new();
    let mut audio_recipients = HashMap::<String, Vec<Recipient<ProcessAudio>>>::new();
    let mut shutdown_recipients = Vec::new();

    match model_type {
        ModelType::LLM => {
            // Initialize LLM model
            for _ in 0..num_instances {
                let file = File::open(&config_file_name)
                    .expect(&format!("Config {} not found!", config_file_name));
                let mut de = serde_json::Deserializer::from_reader(BufReader::new(file));
                let config = SimpleLLMConfig::deserialize(&mut de)?;
                let model_name = config.modle_name.clone();
                
                match llmserver_rs::llm::simple::SimpleRkLLM::init(&config) {
                    Ok(llm) => {
                        let addr = llm.start();
                        if let Some(vec) = llm_recipients.get_mut(&model_name) {
                            vec.push(addr.clone().recipient::<ProcessMessages>());
                        } else {
                            llm_recipients.insert(model_name, vec![addr.clone().recipient::<ProcessMessages>()]);
                        }
                        shutdown_recipients.push(addr.clone().recipient::<ShutdownMessages>());
                    },
                    Err(e) => {
                        eprintln!("Failed to initialize LLM model {}: {}", model_id, e);
                        panic!("Failed to initialize model");
                    }
                }
            }
        },
        ModelType::ASR => {
            // Initialize ASR model
            for _ in 0..num_instances {
                let file = File::open(&config_file_name)
                    .expect(&format!("Config {} not found!", config_file_name));
                let mut de = serde_json::Deserializer::from_reader(BufReader::new(file));
                let config = SimpleASRConfig::deserialize(&mut de)?;
                let model_name = config.modle_name.clone();
                
                match llmserver_rs::asr::simple::SimpleASR::init(&config) {
                    Ok(asr) => {
                        let addr = asr.start();
                        if let Some(vec) = audio_recipients.get_mut(&model_name) {
                            vec.push(addr.clone().recipient::<ProcessAudio>());
                        } else {
                            audio_recipients.insert(model_name, vec![addr.clone().recipient::<ProcessAudio>()]);
                        }
                        shutdown_recipients.push(addr.clone().recipient::<ShutdownMessages>());
                    },
                    Err(e) => {
                        eprintln!("Failed to initialize ASR model {}: {}", model_id, e);
                        panic!("Failed to initialize model");
                    }
                }
            }
        }
    }

    if audio_recipients.len() == 0 && llm_recipients.len() == 0 {
        panic!("Failed to load any model");
    }

    HttpServer::new(move || {
        let (app, api) = App::new()
            .app_data(actix_web::web::Data::new(llm_recipients.clone()))
            .app_data(actix_web::web::Data::new(audio_recipients.clone()))
            .into_utoipa_app()
            .map(|app| app.wrap(Logger::default()))
            .service(
                scope::scope("/v1")
                    .service(llmserver_rs::chat::chat_completions)
                    .service(llmserver_rs::audio::audio_transcriptions),
            )
            .service(health)
            .split_for_parts();

        app.service(SwaggerUi::new("/swagger-ui/{_:.*}").url("/api-docs/openapi.json", api))
    })
    .bind((Ipv4Addr::UNSPECIFIED, 8080))?
    .run()
    .await?;

    let shutdowns = shutdown_recipients.into_iter().map(|addr| async move {
        let _ = addr.send(ShutdownMessages).await.unwrap();
    });

    tokio::spawn(async {
        futures::future::join_all(shutdowns).await;
    })
    .await?;
    Ok(())
}

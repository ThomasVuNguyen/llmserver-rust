use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Checks if a model exists and is accessible on Hugging Face
pub fn check_model_exists(model_id: &str) -> bool {
    let api = Api::new().expect("Failed to create Hugging Face API client");
    
    // The model_id should be in the format "owner/name"
    // We don't need to split it as the API accepts the full model_id
    api.model(model_id.to_string()).info().is_ok()
}

/// Determines the model type based on files or metadata
pub fn determine_model_type(model_id: &str) -> Option<ModelType> {
    let api = Api::new().expect("Failed to create Hugging Face API client");
    
    // Check for common files that indicate model type
    // The repo_id should be in the format "owner/name"
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    
    // This is a simplified approach - in a real implementation, you'd want to check
    // for specific files or metadata that indicate the model type
    if model_id.to_lowercase().contains("llm") || model_id.to_lowercase().contains("gpt") {
        Some(ModelType::LLM)
    } else if model_id.to_lowercase().contains("voice") || model_id.to_lowercase().contains("asr") {
        Some(ModelType::ASR)
    } else {
        // Default to LLM if we can't determine
        Some(ModelType::LLM)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    LLM,
    ASR,
}

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub modle_path: String,
    pub modle_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
}

/// Creates a config file for a model
pub fn create_config_file(model_id: &str, model_type: ModelType) -> std::io::Result<String> {
    let parts: Vec<&str> = model_id.split('/').collect();
    if parts.len() != 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid model ID format",
        ));
    }
    
    let name = parts[1];
    
    // Create config directory if it doesn't exist
    let config_dir = Path::new("assets/config");
    if !config_dir.exists() {
        fs::create_dir_all(config_dir)?;
    }
    
    // Create config file name
    let file_name = format!("{}.json", name.to_lowercase().replace('-', "_"));
    let config_path = config_dir.join(&file_name);
    
    // Create config content
    let config = ModelConfig {
        modle_path: model_id.to_string(),
        modle_name: name.to_string(),
        think: if model_type == ModelType::LLM { Some(false) } else { None },
    };
    
    // Write config to file
    let config_json = serde_json::to_string_pretty(&config)?;
    let mut file = File::create(&config_path)?;
    file.write_all(config_json.as_bytes())?;
    
    Ok(config_path.to_string_lossy().to_string())
}
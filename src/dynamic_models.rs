//! Dynamic model database that fetches real-world AI model information
//! from Hugging Face, Ollama, and other model repositories.

use anyhow::{Context, Result};
use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, timeout};
use tracing::{debug, info, warn};

use crate::models::ModelDefinition;
use crate::models::{ModelArchitecture, ModelType};
use crate::workloads::QuantizationLevel;

/// Configuration for model fetching
#[derive(Debug, Clone)]
pub struct ModelFetchConfig {
    /// Enable Hugging Face API fetching
    pub enable_huggingface: bool,
    /// Enable Ollama model list fetching
    pub enable_ollama: bool,
    /// Cache duration in seconds
    pub cache_duration_seconds: u64,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Maximum number of models to fetch per source
    pub max_models_per_source: usize,
    /// Minimum downloads threshold for Hugging Face models
    pub min_downloads_threshold: u64,
    /// Enable bulk fetching (fetch all models above threshold)
    pub enable_bulk_fetching: bool,
    /// Maximum number of pages to fetch when bulk fetching
    pub max_pages_to_fetch: usize,
}

impl Default for ModelFetchConfig {
    fn default() -> Self {
        Self {
            enable_huggingface: true,
            enable_ollama: true,
            cache_duration_seconds: 3600, // 1 hour
            request_timeout_seconds: 30,
            max_models_per_source: 100,
            min_downloads_threshold: 1000, // Only models with 1000+ downloads
            enable_bulk_fetching: false,   // Conservative default
            max_pages_to_fetch: 10,        // 10 pages * 50 models = 500 models max
        }
    }
}

/// Dynamic model database that fetches from multiple sources
#[derive(Debug)]
pub struct DynamicModelDatabase {
    config: ModelFetchConfig,
    client: reqwest::Client,
    cached_models: dashmap::DashMap<String, ModelDefinition>,
    last_fetch_time: std::sync::RwLock<Option<chrono::DateTime<chrono::Utc>>>,
}

/// Hugging Face model card structure (subset of useful fields)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct HuggingFaceModel {
    #[serde(rename = "modelId")]
    model_id: String,
    downloads: Option<u64>,
    #[allow(dead_code)]
    likes: Option<u64>,
    #[allow(dead_code)]
    tags: Option<Vec<String>>,
    #[serde(rename = "createdAt")]
    #[allow(dead_code)]
    created_at: Option<String>,
    #[serde(rename = "lastModified")]
    #[allow(dead_code)]
    last_modified: Option<String>,
}

/// Ollama model list structure
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaModel {
    name: String,
    size: Option<u64>,
    description: Option<String>,
    tags: Option<Vec<String>>,
}

/// Ollama API response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaModelsResponse {
    models: Vec<OllamaModel>,
}

impl DynamicModelDatabase {
    /// Create a new dynamic model database
    pub fn new() -> Self {
        Self::with_config(ModelFetchConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ModelFetchConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .user_agent("system-analysis/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            cached_models: dashmap::DashMap::new(),
            last_fetch_time: std::sync::RwLock::new(None),
        }
    }

    /// Enable bulk fetching with custom parameters
    pub fn enable_bulk_fetching(&mut self, min_downloads: u64, max_pages: usize) {
        self.config.enable_bulk_fetching = true;
        self.config.min_downloads_threshold = min_downloads;
        self.config.max_pages_to_fetch = max_pages;
    }

    /// Disable bulk fetching (back to single page)
    pub fn disable_bulk_fetching(&mut self) {
        self.config.enable_bulk_fetching = false;
    }

    /// Set download threshold for bulk fetching
    pub fn set_download_threshold(&mut self, threshold: u64) {
        self.config.min_downloads_threshold = threshold;
    }

    /// Set maximum pages to fetch
    pub fn set_max_pages(&mut self, max_pages: usize) {
        self.config.max_pages_to_fetch = max_pages;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &ModelFetchConfig {
        &self.config
    }

    /// Get all available models, fetching from sources if cache is stale
    pub async fn get_models(&self) -> Result<Vec<ModelDefinition>> {
        if self.should_refresh_cache() {
            self.refresh_model_database().await?;
        }

        Ok(self
            .cached_models
            .iter()
            .map(|entry| entry.value().clone())
            .collect())
    }

    /// Get a specific model by name
    pub async fn get_model(&self, name: &str) -> Result<Option<ModelDefinition>> {
        if self.should_refresh_cache() {
            self.refresh_model_database().await?;
        }

        Ok(self
            .cached_models
            .get(name)
            .map(|entry| entry.value().clone()))
    }

    /// Search models by pattern with fuzzy matching
    pub async fn search_models(&self, query: &str) -> Result<Vec<ModelDefinition>> {
        if self.should_refresh_cache() {
            self.refresh_model_database().await?;
        }

        let query_lower = query.to_lowercase();
        let mut scored_models: Vec<(ModelDefinition, i32)> = self
            .cached_models
            .iter()
            .filter_map(|entry| {
                let model = entry.value();
                let score = self.calculate_fuzzy_score(model, &query_lower);
                if score > 0 {
                    Some((model.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance score (higher is better)
        scored_models.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(scored_models.into_iter().map(|(model, _)| model).collect())
    }

    /// Find similar model names for suggestion (fuzzy matching)
    pub async fn find_similar_models(
        &self,
        partial_name: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        if self.should_refresh_cache() {
            self.refresh_model_database().await?;
        }

        let query_lower = partial_name.to_lowercase();
        let mut scored_names: Vec<(String, i32)> = self
            .cached_models
            .iter()
            .filter_map(|entry| {
                let model = entry.value();
                let score = self.calculate_name_similarity(&model.name, &query_lower);
                if score > 0 {
                    Some((model.name.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance score (higher is better)
        scored_names.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(scored_names
            .into_iter()
            .take(limit)
            .map(|(name, _)| name)
            .collect())
    }

    /// Calculate fuzzy matching score for a model
    fn calculate_fuzzy_score(&self, model: &ModelDefinition, query: &str) -> i32 {
        let mut score = 0;
        let model_name_lower = model.name.to_lowercase();
        let family_lower = model.family.to_lowercase();

        // Exact match in name gets highest score
        if model_name_lower.contains(query) {
            score += 100;
            // Bonus for exact word match
            if model_name_lower
                .split(&['-', '_', '/', ' '][..])
                .any(|word| word == query)
            {
                score += 50;
            }
            // Extra bonus for starting with query
            if model_name_lower.starts_with(query) {
                score += 30;
            }
        }

        // Family match
        if family_lower.contains(query) || family_lower == query {
            score += if family_lower == query { 50 } else { 30 };
        }

        // Only award partial matches for longer queries (3+ chars) to reduce noise
        if query.len() >= 3 {
            // Check for substring matches in model components
            for part in model_name_lower.split(&['-', '_', '/', ' '][..]) {
                if part.contains(query) {
                    score += 20;
                    break; // Only count once
                }
            }
        }

        // Minimum score threshold - only return results with meaningful matches
        if score < 20 { 0 } else { score }
    }

    /// Calculate name similarity score for suggestions
    fn calculate_name_similarity(&self, model_name: &str, query: &str) -> i32 {
        let name_lower = model_name.to_lowercase();

        // Strong preference for names that start with the query
        if name_lower.starts_with(query) {
            return 1000;
        }

        // Check if any part of the model name starts with query
        for part in name_lower.split(&['-', '_', '/', ' '][..]) {
            if part.starts_with(query) {
                return 500;
            }
        }

        // Containment check
        if name_lower.contains(query) {
            return 100;
        }

        // Levenshtein-like simple distance
        let distance = self.simple_edit_distance(&name_lower, query);
        if distance <= 3 {
            return 50 - (distance as i32 * 10);
        }

        0
    }

    /// Simple edit distance calculation
    fn simple_edit_distance(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        if s1_chars.is_empty() {
            return s2_chars.len();
        }
        if s2_chars.is_empty() {
            return s1_chars.len();
        }

        let mut prev_row: Vec<usize> = (0..=s2_chars.len()).collect();

        for (i, &ch1) in s1_chars.iter().enumerate() {
            let mut curr_row = vec![i + 1];

            for (j, &ch2) in s2_chars.iter().enumerate() {
                let cost = if ch1 == ch2 { 0 } else { 1 };
                curr_row.push(
                    (curr_row[j] + 1)
                        .min(prev_row[j + 1] + 1)
                        .min(prev_row[j] + cost),
                );
            }

            prev_row = curr_row;
        }

        prev_row[s2_chars.len()]
    }

    pub async fn refresh_model_database(&self) -> Result<()> {
        info!("Refreshing model database from external sources");
        let mut models = Vec::new();

        // Fetch from Hugging Face
        if self.config.enable_huggingface {
            match self.fetch_huggingface_models().await {
                Ok(mut fetched) => {
                    info!("Fetched {} models from Hugging Face", fetched.len());
                    models.append(&mut fetched);
                }
                Err(e) => {
                    warn!("Failed to fetch from Hugging Face: {}", e);
                }
            }
        }

        // Fetch from Ollama
        if self.config.enable_ollama {
            match self.fetch_ollama_models().await {
                Ok(mut fetched) => {
                    info!("Fetched {} models from Ollama", fetched.len());
                    models.append(&mut fetched);
                }
                Err(e) => {
                    warn!("Failed to fetch from Ollama: {}", e);
                }
            }
        }

        // Add curated high-quality models
        models.append(&mut self.get_curated_models());

        debug!(
            "Total models collected before deduplication: {}",
            models.len()
        );

        // Deduplicate models by name before caching
        let mut seen_models = std::collections::HashSet::new();
        let mut deduplicated_models = Vec::new();

        for model in models {
            if !seen_models.contains(&model.name) {
                seen_models.insert(model.name.clone());
                deduplicated_models.push(model);
            } else {
                debug!("Skipping duplicate model: {}", model.name);
            }
        }

        debug!(
            "After deduplication: {} unique models",
            deduplicated_models.len()
        );

        // Update cache with deduplicated models
        self.cached_models.clear();
        for model in deduplicated_models {
            self.cached_models.insert(model.name.clone(), model);
        }

        // Update last fetch time
        if let Ok(mut last_fetch) = self.last_fetch_time.write() {
            *last_fetch = Some(chrono::Utc::now());
        }

        info!(
            "Model database refresh complete. {} models cached",
            self.cached_models.len()
        );
        Ok(())
    }

    /// Check if cache should be refreshed
    fn should_refresh_cache(&self) -> bool {
        if let Ok(last_fetch) = self.last_fetch_time.read() {
            if let Some(last_time) = *last_fetch {
                let cache_duration =
                    chrono::Duration::seconds(self.config.cache_duration_seconds as i64);
                return chrono::Utc::now() - last_time > cache_duration;
            }
        }
        true // Cache never populated
    }

    /// Fetch popular models from Hugging Face
    async fn fetch_huggingface_models(&self) -> Result<Vec<ModelDefinition>> {
        debug!("Fetching models from Hugging Face API");

        let mut models = if self.config.enable_bulk_fetching {
            // Bulk fetch with pagination
            self.fetch_huggingface_bulk().await?
        } else {
            // Standard single-page fetch
            self.fetch_huggingface_single_page().await?
        };

        // Filter by download threshold
        models.retain(|model| {
            // Estimate downloads from model name/popularity
            self.estimate_downloads_from_model(model) >= self.config.min_downloads_threshold
        });

        info!(
            "Fetched {} models from Hugging Face after filtering",
            models.len()
        );
        Ok(models)
    }

    /// Fetch a single page of models (original behavior)
    async fn fetch_huggingface_single_page(&self) -> Result<Vec<ModelDefinition>> {
        let url = "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1";

        let models_response = timeout(
            Duration::from_secs(self.config.request_timeout_seconds),
            self.client.get(url).send(),
        )
        .await
        .context("Request timeout")?
        .context("Failed to send request")?;

        let hf_models: Vec<HuggingFaceModel> = models_response
            .json()
            .await
            .context("Failed to parse Hugging Face response")?;

        let mut models = Vec::new();
        for hf_model in hf_models
            .into_iter()
            .take(self.config.max_models_per_source)
        {
            if let Some(model_def) = self.convert_huggingface_model(hf_model) {
                models.push(model_def);
            }
        }

        Ok(models)
    }

    /// Fetch multiple pages of models for bulk downloading
    async fn fetch_huggingface_bulk(&self) -> Result<Vec<ModelDefinition>> {
        let mut all_models = Vec::new();
        let models_per_page = 50; // HF API typically returns 50 per page

        for page in 0..self.config.max_pages_to_fetch {
            let offset = page * models_per_page;
            let url = format!(
                "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit={}&offset={}",
                models_per_page, offset
            );

            debug!(
                "Fetching page {} from Hugging Face (offset: {})",
                page + 1,
                offset
            );

            let models_response = match timeout(
                Duration::from_secs(self.config.request_timeout_seconds),
                self.client.get(&url).send(),
            )
            .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(e)) => {
                    warn!("Failed to fetch page {}: {}", page + 1, e);
                    break; // Stop on error
                }
                Err(_) => {
                    warn!("Timeout fetching page {}", page + 1);
                    break; // Stop on timeout
                }
            };

            let hf_models: Vec<HuggingFaceModel> = match models_response.json().await {
                Ok(models) => models,
                Err(e) => {
                    warn!("Failed to parse page {}: {}", page + 1, e);
                    break;
                }
            };

            // If we get fewer models than expected, we've hit the end
            if hf_models.is_empty() {
                debug!("No more models found at page {}, stopping", page + 1);
                break;
            }

            // Filter by download threshold at the API level if possible
            let filtered_models: Vec<_> = hf_models
                .into_iter()
                .filter(|hf_model| {
                    hf_model.downloads.unwrap_or(0) >= self.config.min_downloads_threshold
                })
                .collect();

            // If we're getting models below threshold, we can stop
            if filtered_models.is_empty() {
                debug!(
                    "All models on page {} below download threshold, stopping",
                    page + 1
                );
                break;
            }

            // Convert to our format
            let pre_count = all_models.len();
            let conversion_attempts = filtered_models.len();
            for hf_model in filtered_models {
                if let Some(model_def) = self.convert_huggingface_model(hf_model) {
                    all_models.push(model_def);
                }
            }

            let converted_count = all_models.len() - pre_count;
            debug!(
                "Page {} - Attempted: {}, Converted: {}, Success rate: {:.1}% (total: {})",
                page + 1,
                conversion_attempts,
                converted_count,
                (converted_count as f64 / conversion_attempts as f64) * 100.0,
                all_models.len()
            );

            // Small delay to be respectful to the API
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(all_models)
    }

    /// Estimate downloads for filtering purposes
    fn estimate_downloads_from_model(&self, model: &ModelDefinition) -> u64 {
        // This is a simple heuristic - in bulk mode we already filter by actual downloads
        // For single-page mode or curated models, we estimate based on popularity indicators
        match model.family.as_str() {
            "llama" => 50000,   // Very popular
            "mistral" => 30000, // Popular
            "qwen" => 25000,    // Popular
            "gemma" => 20000,   // Google backed
            "phi" => 15000,     // Microsoft backed
            _ => 5000,          // Default estimate
        }
    }

    /// Fetch models from Ollama registry
    async fn fetch_ollama_models(&self) -> Result<Vec<ModelDefinition>> {
        debug!("Fetching models from Ollama");

        // Note: Ollama doesn't have a public API for model listing currently
        // This is a placeholder for when they add it, or we could scrape their website
        // For now, we'll return curated Ollama-compatible models

        Ok(self.get_popular_ollama_models())
    }

    /// Convert Hugging Face model to our format
    fn convert_huggingface_model(&self, hf_model: HuggingFaceModel) -> Option<ModelDefinition> {
        // Extract model information from the model ID and tags
        let name = hf_model.model_id.clone();

        // Skip models that are clearly not text generation models
        if name.contains("embedding")
            || name.contains("reranker")
            || name.contains("classifier")
            || name.contains("tokenizer")
            || name.contains("dataset")
        {
            debug!("Skipping non-generative model: {}", name);
            return None;
        }

        // Determine family from model name patterns
        let family = self.extract_model_family(&name);

        // Estimate parameters and memory from model name/tags
        let parameters = self.estimate_parameters_from_name(&name);
        let base_memory_gb = (parameters as f64 * 2.0) / 1_000_000_000.0; // Rough estimate: 2 bytes per param

        // Determine supported quantization levels
        let supported_quantization = vec![
            QuantizationLevel::None,
            QuantizationLevel::Int8,
            QuantizationLevel::Int4,
        ];

        // Create architecture info
        let architecture = ModelArchitecture {
            arch_type: if family.contains("bert") {
                "encoder"
            } else {
                "decoder"
            }
            .to_string(),
            layers: self.estimate_layers_from_parameters(parameters),
            hidden_size: self.estimate_hidden_size_from_parameters(parameters),
            attention_heads: Some(self.estimate_attention_heads_from_parameters(parameters)),
            supports_multi_gpu: parameters > 10_000_000_000, // Only large models support multi-GPU
        };

        debug!(
            "Successfully converted model: {} (family: {}, params: {}B)",
            name,
            family,
            parameters / 1_000_000_000
        );

        Some(ModelDefinition {
            name,
            family,
            parameters,
            base_memory_gb,
            min_compute: self.estimate_compute_requirement(parameters),
            supported_quantization,
            model_type: ModelType::Both,
            context_lengths: vec![512, 1024, 2048, 4096],
            architecture,
        })
    }

    /// Extract model family from name
    fn extract_model_family(&self, name: &str) -> String {
        let name_lower = name.to_lowercase();

        // Primary families (order matters - check specific first)
        if name_lower.contains("codellama") {
            "codellama".to_string()
        } else if name_lower.contains("llama") {
            "llama".to_string()
        } else if name_lower.contains("mistral") {
            "mistral".to_string()
        } else if name_lower.contains("qwen") {
            "qwen".to_string()
        } else if name_lower.contains("phi") {
            "phi".to_string()
        } else if name_lower.contains("gemma") {
            "gemma".to_string()
        } else if name_lower.contains("claude") {
            "claude".to_string()
        } else if name_lower.contains("gpt") || name_lower.contains("openai") {
            "gpt".to_string()
        } else if name_lower.contains("bert") {
            "bert".to_string()
        } else if name_lower.contains("t5") || name_lower.contains("flan") {
            "t5".to_string()
        } else if name_lower.contains("deepseek") {
            "deepseek".to_string()
        } else if name_lower.contains("yi") {
            "yi".to_string()
        } else if name_lower.contains("falcon") {
            "falcon".to_string()
        } else if name_lower.contains("vicuna") {
            "vicuna".to_string()
        } else if name_lower.contains("alpaca") {
            "alpaca".to_string()
        } else if name_lower.contains("bloom") {
            "bloom".to_string()
        } else if name_lower.contains("opt") {
            "opt".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Estimate parameters from model name
    fn estimate_parameters_from_name(&self, name: &str) -> u64 {
        let name_lower = name.to_lowercase();

        // Look for common parameter patterns (check larger first)
        if name_lower.contains("405b") {
            405_000_000_000
        } else if name_lower.contains("175b") {
            175_000_000_000
        } else if name_lower.contains("72b") || name_lower.contains("70b") {
            70_000_000_000
        } else if name_lower.contains("34b") || name_lower.contains("32b") {
            34_000_000_000
        } else if name_lower.contains("22b") || name_lower.contains("20b") {
            22_000_000_000
        } else if name_lower.contains("14b") || name_lower.contains("13b") {
            13_000_000_000
        } else if name_lower.contains("9b") || name_lower.contains("8b") {
            8_000_000_000
        } else if name_lower.contains("7b") {
            7_000_000_000
        } else if name_lower.contains("4b") || name_lower.contains("3b") {
            3_000_000_000
        } else if name_lower.contains("1.7b")
            || name_lower.contains("1.5b")
            || name_lower.contains("1.1b")
        {
            1_500_000_000
        } else if name_lower.contains("1b") {
            1_000_000_000
        } else if name_lower.contains("0.6b") || name_lower.contains("0.5b") {
            500_000_000
        } else if name_lower.contains("350m") {
            350_000_000
        } else if name_lower.contains("135m") || name_lower.contains("125m") {
            125_000_000
        } else if name_lower.contains("small") {
            1_000_000_000
        } else if name_lower.contains("base") {
            7_000_000_000
        } else if name_lower.contains("large") {
            13_000_000_000
        } else {
            7_000_000_000
        } // Default assumption for unspecified models
    }

    /// Estimate other model parameters
    fn estimate_layers_from_parameters(&self, params: u64) -> u32 {
        match params {
            p if p >= 100_000_000_000 => 96, // Very large models
            p if p >= 10_000_000_000 => 48,  // Large models
            p if p >= 1_000_000_000 => 24,   // Medium models
            _ => 12,                         // Small models
        }
    }

    fn estimate_hidden_size_from_parameters(&self, params: u64) -> u32 {
        match params {
            p if p >= 100_000_000_000 => 8192,
            p if p >= 10_000_000_000 => 4096,
            p if p >= 1_000_000_000 => 2048,
            _ => 1024,
        }
    }

    fn estimate_attention_heads_from_parameters(&self, params: u64) -> u32 {
        match params {
            p if p >= 100_000_000_000 => 64,
            p if p >= 10_000_000_000 => 32,
            p if p >= 1_000_000_000 => 16,
            _ => 8,
        }
    }

    fn estimate_compute_requirement(&self, params: u64) -> f64 {
        match params {
            p if p >= 100_000_000_000 => 9.5, // Requires top-tier hardware
            p if p >= 10_000_000_000 => 8.0,  // High-end hardware
            p if p >= 1_000_000_000 => 6.0,   // Mid-range hardware
            _ => 4.0,                         // Basic hardware
        }
    }

    /// Get popular Ollama-compatible models
    fn get_popular_ollama_models(&self) -> Vec<ModelDefinition> {
        vec![
            self.create_model_definition("llama3.1:8b", "llama", 8_000_000_000, 16.0, 7.0),
            self.create_model_definition("llama3.1:70b", "llama", 70_000_000_000, 140.0, 9.0),
            self.create_model_definition("mistral:7b", "mistral", 7_000_000_000, 14.0, 6.5),
            self.create_model_definition("qwen2.5:7b", "qwen", 7_000_000_000, 14.0, 6.5),
            self.create_model_definition("phi3:3.8b", "phi", 3_800_000_000, 8.0, 5.5),
            self.create_model_definition("gemma2:9b", "gemma", 9_000_000_000, 18.0, 7.0),
        ]
    }

    /// Get curated high-quality models with accurate specifications
    pub fn get_curated_models(&self) -> Vec<ModelDefinition> {
        vec![
            // Meta Llama models
            self.create_model_definition("Meta-Llama-3.1-8B", "llama", 8_000_000_000, 16.0, 7.0),
            self.create_model_definition("Meta-Llama-3.1-70B", "llama", 70_000_000_000, 140.0, 9.0),
            self.create_model_definition(
                "Meta-Llama-3.1-405B",
                "llama",
                405_000_000_000,
                810.0,
                10.0,
            ),
            // Mistral models
            self.create_model_definition("Mistral-7B-v0.3", "mistral", 7_000_000_000, 14.0, 6.5),
            self.create_model_definition("Mixtral-8x7B", "mistral", 46_000_000_000, 92.0, 8.5),
            // Qwen models
            self.create_model_definition("Qwen2.5-7B", "qwen", 7_000_000_000, 14.0, 6.8),
            self.create_model_definition("Qwen2.5-72B", "qwen", 72_000_000_000, 144.0, 9.0),
            // Code models
            self.create_model_definition("CodeLlama-7B", "codellama", 7_000_000_000, 14.0, 6.5),
            self.create_model_definition("CodeLlama-34B", "codellama", 34_000_000_000, 68.0, 8.5),
        ]
    }

    /// Helper to create model definitions
    fn create_model_definition(
        &self,
        name: &str,
        family: &str,
        parameters: u64,
        base_memory_gb: f64,
        min_compute: f64,
    ) -> ModelDefinition {
        ModelDefinition {
            name: name.to_string(),
            family: family.to_string(),
            parameters,
            base_memory_gb,
            min_compute,
            supported_quantization: vec![
                QuantizationLevel::None,
                QuantizationLevel::Int8,
                QuantizationLevel::Int4,
            ],
            model_type: ModelType::Both,
            context_lengths: vec![512, 1024, 2048, 4096, 8192],
            architecture: ModelArchitecture {
                arch_type: "decoder".to_string(),
                layers: self.estimate_layers_from_parameters(parameters),
                hidden_size: self.estimate_hidden_size_from_parameters(parameters),
                attention_heads: Some(self.estimate_attention_heads_from_parameters(parameters)),
                supports_multi_gpu: parameters > 10_000_000_000, // Only large models support multi-GPU
            },
        }
    }

    /// Get statistics about the model database
    pub fn get_statistics(&self) -> ModelDatabaseStats {
        let total_models = self.cached_models.len();
        let mut family_counts = HashMap::new();
        let mut size_distribution = HashMap::new();

        for model in self.cached_models.iter() {
            let model = model.value();

            // Count by family
            *family_counts.entry(model.family.clone()).or_insert(0) += 1;

            // Count by size category
            let size_category = match model.parameters {
                p if p >= 100_000_000_000 => "100B+",
                p if p >= 10_000_000_000 => "10B-100B",
                p if p >= 1_000_000_000 => "1B-10B",
                _ => "<1B",
            };
            *size_distribution
                .entry(size_category.to_string())
                .or_insert(0) += 1;
        }

        let last_updated = self.last_fetch_time.read().ok().and_then(|time| *time);

        ModelDatabaseStats {
            total_models,
            family_counts,
            size_distribution,
            last_updated,
        }
    }
}

/// Statistics about the model database
#[derive(Debug, Serialize)]
pub struct ModelDatabaseStats {
    pub total_models: usize,
    pub family_counts: HashMap<String, usize>,
    pub size_distribution: HashMap<String, usize>,
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for DynamicModelDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_database_creation() {
        let db = DynamicModelDatabase::new();
        assert!(db.cached_models.is_empty());
    }

    #[tokio::test]
    async fn test_curated_models() {
        let db = DynamicModelDatabase::new();
        let models = db.get_curated_models();
        assert!(!models.is_empty());

        // Verify Llama models are present
        let llama_models: Vec<_> = models.iter().filter(|m| m.family == "llama").collect();
        assert!(!llama_models.is_empty());
    }

    #[test]
    fn test_parameter_estimation() {
        let db = DynamicModelDatabase::new();

        assert_eq!(
            db.estimate_parameters_from_name("llama3.1-8b"),
            8_000_000_000
        );
        assert_eq!(
            db.estimate_parameters_from_name("mistral-7b-v0.3"),
            7_000_000_000
        );
        assert_eq!(
            db.estimate_parameters_from_name("qwen2.5-70b"),
            70_000_000_000
        );
    }

    #[test]
    fn test_family_extraction() {
        let db = DynamicModelDatabase::new();

        assert_eq!(db.extract_model_family("Meta-Llama-3.1-8B"), "llama");
        assert_eq!(db.extract_model_family("mistral-7b-instruct"), "mistral");
        assert_eq!(db.extract_model_family("Qwen2.5-Chat"), "qwen");
    }
}

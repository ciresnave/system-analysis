//! AI/ML model definitions and compatibility analysis
//! 
//! This module provides a comprehensive database of popular AI/ML models
//! and analyzes which models can run on given hardware configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::workloads::QuantizationLevel;
use crate::types::SystemProfile;

/// Popular AI/ML model definitions with their requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// Model name
    pub name: String,
    /// Model family (e.g., "llama", "gpt", "bert")
    pub family: String,
    /// Parameter count
    pub parameters: u64,
    /// Base memory requirement in GB (unquantized)
    pub base_memory_gb: f64,
    /// Minimum compute requirement (0-10 scale)
    pub min_compute: f64,
    /// Supported quantization levels
    pub supported_quantization: Vec<QuantizationLevel>,
    /// Model type (inference, training, both)
    pub model_type: ModelType,
    /// Recommended context lengths
    pub context_lengths: Vec<u32>,
    /// Model architecture details
    pub architecture: ModelArchitecture,
}

/// Model type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Inference only
    Inference,
    /// Training only
    Training,
    /// Both inference and training
    Both,
}

/// Model architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Architecture type (transformer, cnn, rnn, etc.)
    pub arch_type: String,
    /// Number of layers
    pub layers: u32,
    /// Hidden size
    pub hidden_size: u32,
    /// Attention heads (for transformers)
    pub attention_heads: Option<u32>,
    /// Whether model supports multi-GPU
    pub supports_multi_gpu: bool,
}

/// Model runner information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRunner {
    /// Runner name
    pub name: String,
    /// Description
    pub description: String,
    /// Supported model families
    pub supported_families: Vec<String>,
    /// Supported platforms
    pub platforms: Vec<Platform>,
    /// Runner capabilities
    pub capabilities: RunnerCapabilities,
    /// Performance characteristics
    pub performance: RunnerPerformance,
}

/// Platform support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    /// CUDA (NVIDIA GPUs)
    Cuda,
    /// ROCm (AMD GPUs)
    Rocm,
    /// Metal (Apple Silicon)
    Metal,
    /// OpenCL
    OpenCL,
    /// CPU only
    Cpu,
    /// Vulkan
    Vulkan,
    /// DirectML (Windows)
    DirectML,
}

/// Runner capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerCapabilities {
    /// Supports CPU-GPU model splitting
    pub cpu_gpu_split: bool,
    /// Supports dynamic quantization
    pub dynamic_quantization: bool,
    /// Supports batch processing
    pub batch_processing: bool,
    /// Supports streaming inference
    pub streaming: bool,
    /// Maximum context length supported
    pub max_context_length: u32,
    /// Supported quantization levels
    pub quantization_support: Vec<QuantizationLevel>,
}

/// Runner performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerPerformance {
    /// CPU efficiency score (0-10)
    pub cpu_efficiency: f64,
    /// GPU efficiency score (0-10)
    pub gpu_efficiency: f64,
    /// Memory efficiency score (0-10)
    pub memory_efficiency: f64,
    /// Setup overhead (low, medium, high)
    pub setup_overhead: OverheadLevel,
}

/// Overhead level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverheadLevel {
    Low,
    Medium,
    High,
}

/// Model compatibility analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompatibility {
    /// Whether the model can run on the system
    pub can_run: bool,
    /// Recommended quantization level
    pub recommended_quantization: Option<QuantizationLevel>,
    /// Recommended runner
    pub recommended_runner: Option<String>,
    /// Expected memory usage in GB
    pub expected_memory_gb: f64,
    /// Performance estimate (0-10)
    pub performance_estimate: f64,
    /// Required preprocessing steps
    pub preprocessing_steps: Vec<PreprocessingStep>,
    /// Runner-specific settings
    pub runner_settings: HashMap<String, String>,
}

/// Preprocessing step for model optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStep {
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Estimated time to complete
    pub estimated_time_minutes: u32,
    /// Whether this step is required or optional
    pub required: bool,
}

/// Model database containing popular AI/ML models
pub struct ModelDatabase {
    models: HashMap<String, ModelDefinition>,
    runners: HashMap<String, ModelRunner>,
}

impl ModelDatabase {
    /// Create a new model database with popular models
    pub fn new() -> Self {
        let mut database = Self {
            models: HashMap::new(),
            runners: HashMap::new(),
        };
        
        database.populate_models();
        database.populate_runners();
        database
    }
    
    /// Get all available models
    pub fn get_models(&self) -> &HashMap<String, ModelDefinition> {
        &self.models
    }
    
    /// Get all available runners
    pub fn get_runners(&self) -> &HashMap<String, ModelRunner> {
        &self.runners
    }
    
    /// Add a model to the database
    pub fn add_model(&mut self, id: String, model: ModelDefinition) {
        self.models.insert(id, model);
    }
    
    /// Add a runner to the database
    pub fn add_runner(&mut self, id: String, runner: ModelRunner) {
        self.runners.insert(id, runner);
    }
    
    /// Get models that can run on the given system
    pub fn get_compatible_models(&self, system: &SystemProfile) -> Vec<(String, ModelCompatibility)> {
        self.models
            .iter()
            .filter_map(|(name, model)| {
                self.analyze_model_compatibility(model, system)
                    .map(|compat| (name.clone(), compat))
                    .filter(|(_, compat)| compat.can_run)
            })
            .collect()
    }
    
    /// Analyze compatibility for a specific model
    pub fn analyze_model_compatibility(
        &self,
        model: &ModelDefinition,
        system: &SystemProfile,
    ) -> Option<ModelCompatibility> {
        // Check if system meets minimum requirements
        let system_memory_gb = system.system_info.memory_info.total_ram as f64 / 1024.0; // Convert MB to GB
        let gpu_memory_gb = system.system_info.gpu_info.iter()
            .filter_map(|gpu| gpu.vram_size.map(|vram| vram as f64 / 1024.0)) // Convert MB to GB
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        // Determine best quantization level
        let recommended_quantization = self.find_optimal_quantization(
            model, system_memory_gb, gpu_memory_gb
        );
        
        let quantized_memory = self.calculate_quantized_memory(
            model.base_memory_gb, 
            recommended_quantization.as_ref()
        );
        
        // Check if model can run with recommended quantization
        let can_run = quantized_memory <= system_memory_gb.max(gpu_memory_gb);
        
        if !can_run {
            return Some(ModelCompatibility {
                can_run: false,
                recommended_quantization,
                recommended_runner: None,
                expected_memory_gb: quantized_memory,
                performance_estimate: 0.0,
                preprocessing_steps: vec![],
                runner_settings: HashMap::new(),
            });
        }
        
        // Find best runner for this system and model
        let recommended_runner = self.find_optimal_runner(model, system);
        
        // Generate preprocessing steps
        let preprocessing_steps = self.generate_preprocessing_steps(
            model, 
            &recommended_quantization,
            &recommended_runner
        );
        
        // Generate runner settings
        let runner_settings = self.generate_runner_settings(
            model, 
            system, 
            recommended_runner.as_deref()
        );
        
        // Estimate performance
        let performance_estimate = self.estimate_performance(
            model, 
            system, 
            &recommended_quantization,
            recommended_runner.as_deref()
        );
        
        Some(ModelCompatibility {
            can_run: true,
            recommended_quantization,
            recommended_runner,
            expected_memory_gb: quantized_memory,
            performance_estimate,
            preprocessing_steps,
            runner_settings,
        })
    }
    
    /// Find optimal quantization level for the system
    fn find_optimal_quantization(
        &self,
        model: &ModelDefinition,
        system_memory_gb: f64,
        gpu_memory_gb: f64,
    ) -> Option<QuantizationLevel> {
        let max_memory = system_memory_gb.max(gpu_memory_gb);
        
        // Try quantization levels in order of preference
        let quantization_order = [
            QuantizationLevel::None,
            QuantizationLevel::Int8,
            QuantizationLevel::Int4,
        ];
        
        for &quant in &quantization_order {
            if model.supported_quantization.contains(&quant) {
                let required_memory = self.calculate_quantized_memory(model.base_memory_gb, Some(&quant));
                if required_memory <= max_memory * 0.9 { // Leave 10% headroom
                    return Some(quant);
                }
            }
        }
        
        // If nothing fits, return the most aggressive quantization available
        model.supported_quantization
            .iter()
            .min_by_key(|q| match q {
                QuantizationLevel::None => 4,
                QuantizationLevel::Int8 => 2,
                QuantizationLevel::Int4 => 1,
                QuantizationLevel::Custom(_) => 3,
            })
            .cloned()
    }
    
    /// Calculate memory usage with quantization
    fn calculate_quantized_memory(&self, base_memory: f64, quantization: Option<&QuantizationLevel>) -> f64 {
        match quantization {
            Some(QuantizationLevel::Int8) => base_memory * 0.5,
            Some(QuantizationLevel::Int4) => base_memory * 0.25,
            Some(QuantizationLevel::Custom(bits)) => base_memory * (*bits / 32.0),
            _ => base_memory,
        }
    }
    
    /// Find optimal runner for model and system
    fn find_optimal_runner(&self, model: &ModelDefinition, system: &SystemProfile) -> Option<String> {
        let mut best_runner = None;
        let mut best_score = 0.0;
        
        for (name, runner) in &self.runners {
            if !runner.supported_families.contains(&model.family) {
                continue;
            }
            
            let score = self.calculate_runner_score(runner, system);
            if score > best_score {
                best_score = score;
                best_runner = Some(name.clone());
            }
        }
        
        best_runner
    }
    
    /// Calculate runner compatibility score
    fn calculate_runner_score(&self, runner: &ModelRunner, system: &SystemProfile) -> f64 {
        let mut score = 0.0;
        
        // Platform compatibility
        let has_gpu = !system.system_info.gpu_info.is_empty();
        let platform_score = if has_gpu {
            if runner.platforms.contains(&Platform::Cuda) { 10.0 }
            else if runner.platforms.contains(&Platform::Rocm) { 8.0 }
            else if runner.platforms.contains(&Platform::OpenCL) { 6.0 }
            else if runner.platforms.contains(&Platform::Cpu) { 4.0 }
            else { 0.0 }
        } else if runner.platforms.contains(&Platform::Cpu) { 10.0 }
        else { 0.0 };
        
        score += platform_score * 0.4;
        
        // Performance characteristics
        if has_gpu {
            score += runner.performance.gpu_efficiency * 0.3;
        } else {
            score += runner.performance.cpu_efficiency * 0.3;
        }
        
        score += runner.performance.memory_efficiency * 0.2;
        
        // Capabilities bonus
        if runner.capabilities.cpu_gpu_split && has_gpu { score += 1.0; }
        if runner.capabilities.dynamic_quantization { score += 0.5; }
        
        score
    }
    
    /// Generate preprocessing steps
    fn generate_preprocessing_steps(
        &self,
        model: &ModelDefinition,
        quantization: &Option<QuantizationLevel>,
        runner: &Option<String>,
    ) -> Vec<PreprocessingStep> {
        let mut steps = Vec::new();
        
        // Quantization step
        if let Some(quant) = quantization {
            if *quant != QuantizationLevel::None {
                let quant_desc = match quant {
                    QuantizationLevel::Int8 => "8-bit integer".to_string(),
                    QuantizationLevel::Int4 => "4-bit integer".to_string(),
                    QuantizationLevel::Custom(bits) => format!("{bits}-bit custom"),
                    _ => "unknown".to_string(),
                };
                
                steps.push(PreprocessingStep {
                    name: format!("Quantize to {quant:?}"),
                    description: format!("Convert model to {quant_desc} quantization"),
                    estimated_time_minutes: (model.parameters / 1_000_000_000) as u32 * 5,
                    required: true,
                });
            }
        }
        
        // Runner-specific steps
        if let Some(runner_name) = runner {
            if let Some(_runner) = self.runners.get(runner_name) {
                if runner_name.contains("llama.cpp") {
                    steps.push(PreprocessingStep {
                        name: "Convert to GGML format".to_string(),
                        description: "Convert model to llama.cpp compatible GGML format".to_string(),
                        estimated_time_minutes: (model.parameters / 1_000_000_000) as u32 * 10,
                        required: true,
                    });
                } else if runner_name.contains("onnx") {
                    steps.push(PreprocessingStep {
                        name: "Convert to ONNX format".to_string(),
                        description: "Convert model to ONNX runtime compatible format".to_string(),
                        estimated_time_minutes: (model.parameters / 1_000_000_000) as u32 * 15,
                        required: true,
                    });
                }
            }
        }
        
        steps
    }
    
    /// Generate runner-specific settings
    fn generate_runner_settings(
        &self,
        model: &ModelDefinition,
        system: &SystemProfile,
        runner_name: Option<&str>,
    ) -> HashMap<String, String> {
        let mut settings = HashMap::new();
        
        let has_gpu = !system.system_info.gpu_info.is_empty();
        let gpu_memory_gb = system.system_info.gpu_info.iter()
            .filter_map(|gpu| gpu.vram_size.map(|vram| vram as f64 / 1024.0)) // Convert MB to GB
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        if let Some(runner) = runner_name {
            match runner {
                name if name.contains("llama.cpp") => {
                    // llama.cpp specific settings
                    if has_gpu {
                        let gpu_layers = self.calculate_gpu_layers(model, gpu_memory_gb);
                        settings.insert("gpu_layers".to_string(), gpu_layers.to_string());
                        settings.insert("main_gpu".to_string(), "0".to_string());
                    }
                    
                    let threads = (system.system_info.cpu_info.logical_cores / 2).clamp(1, 16);
                    settings.insert("threads".to_string(), threads.to_string());
                    settings.insert("batch_size".to_string(), "512".to_string());
                },
                name if name.contains("onnx") => {
                    // ONNX Runtime settings
                    if has_gpu {
                        settings.insert("provider".to_string(), "CUDAExecutionProvider".to_string());
                    } else {
                        settings.insert("provider".to_string(), "CPUExecutionProvider".to_string());
                    }
                    settings.insert("inter_op_num_threads".to_string(), system.system_info.cpu_info.logical_cores.to_string());
                },
                name if name.contains("transformers") => {
                    // Hugging Face Transformers settings
                    if has_gpu {
                        settings.insert("device".to_string(), "cuda".to_string());
                        settings.insert("torch_dtype".to_string(), "torch.float16".to_string());
                    } else {
                        settings.insert("device".to_string(), "cpu".to_string());
                    }
                },
                _ => {}
            }
        }
        
        settings
    }
    
    /// Calculate optimal number of GPU layers for llama.cpp
    fn calculate_gpu_layers(&self, model: &ModelDefinition, gpu_memory_gb: f64) -> u32 {
        // Rough estimate: each layer uses approximately total_memory / layer_count
        let memory_per_layer = model.base_memory_gb / model.architecture.layers as f64;
        let available_layers = (gpu_memory_gb * 0.8 / memory_per_layer) as u32; // 80% of GPU memory
        available_layers.min(model.architecture.layers)
    }
    
    /// Estimate performance score
    fn estimate_performance(
        &self,
        _model: &ModelDefinition,
        system: &SystemProfile,
        quantization: &Option<QuantizationLevel>,
        runner_name: Option<&str>,
    ) -> f64 {
        let mut score = 5.0; // Base score
        
        // System capability impact
        let has_gpu = !system.system_info.gpu_info.is_empty();
        if has_gpu {
            score += system.gpu_score;
        } else {
            score += system.cpu_score * 0.5;
        }
        
        // Quantization impact
        if let Some(quant) = quantization {
            match quant {
                QuantizationLevel::None => score += 0.0,
                QuantizationLevel::Int8 => score += 1.0, // Faster inference
                QuantizationLevel::Int4 => score += 2.0, // Much faster inference
                QuantizationLevel::Custom(_) => score += 0.5,
            }
        }
        
        // Runner efficiency
        if let Some(runner) = runner_name.and_then(|name| self.runners.get(name)) {
            if has_gpu {
                score += runner.performance.gpu_efficiency * 0.5;
            } else {
                score += runner.performance.cpu_efficiency * 0.5;
            }
        }
        
        score.clamp(0.0, 10.0)
    }
    
    /// Populate the database with popular models
    fn populate_models(&mut self) {
        // Add popular language models
        self.add_llama_models();
        self.add_gpt_models();
        self.add_bert_models();
        self.add_vision_models();
    }
    
    /// Add Llama model family
    fn add_llama_models(&mut self) {
        let models = vec![
            ("llama-7b", 7_000_000_000, 14.0, 32),
            ("llama-13b", 13_000_000_000, 26.0, 40),
            ("llama-30b", 30_000_000_000, 60.0, 60),
            ("llama-65b", 65_000_000_000, 130.0, 80),
            ("llama2-7b", 7_000_000_000, 14.0, 32),
            ("llama2-13b", 13_000_000_000, 26.0, 40),
            ("llama2-70b", 70_000_000_000, 140.0, 80),
            ("codellama-7b", 7_000_000_000, 14.0, 32),
            ("codellama-13b", 13_000_000_000, 26.0, 40),
            ("codellama-34b", 34_000_000_000, 68.0, 64),
        ];
        
        for (name, params, memory, layers) in models {
            self.models.insert(name.to_string(), ModelDefinition {
                name: name.to_string(),
                family: "llama".to_string(),
                parameters: params,
                base_memory_gb: memory,
                min_compute: 4.0,
                supported_quantization: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                    QuantizationLevel::Int4,
                ],
                model_type: ModelType::Both,
                context_lengths: vec![512, 1024, 2048, 4096],
                architecture: ModelArchitecture {
                    arch_type: "transformer".to_string(),
                    layers,
                    hidden_size: 4096,
                    attention_heads: Some(32),
                    supports_multi_gpu: true,
                },
            });
        }
    }
    
    /// Add GPT model family
    fn add_gpt_models(&mut self) {
        let models = vec![
            ("gpt2-small", 117_000_000, 0.5, 12),
            ("gpt2-medium", 345_000_000, 1.4, 24),
            ("gpt2-large", 762_000_000, 3.0, 36),
            ("gpt2-xl", 1_500_000_000, 6.0, 48),
            ("gpt-j-6b", 6_000_000_000, 12.0, 28),
            ("gpt-neox-20b", 20_000_000_000, 40.0, 44),
        ];
        
        for (name, params, memory, layers) in models {
            self.models.insert(name.to_string(), ModelDefinition {
                name: name.to_string(),
                family: "gpt".to_string(),
                parameters: params,
                base_memory_gb: memory,
                min_compute: 3.0,
                supported_quantization: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                    QuantizationLevel::Int4,
                ],
                model_type: ModelType::Both,
                context_lengths: vec![512, 1024, 2048],
                architecture: ModelArchitecture {
                    arch_type: "transformer".to_string(),
                    layers,
                    hidden_size: if params < 1_000_000_000 { 768 } else { 4096 },
                    attention_heads: Some(12),
                    supports_multi_gpu: true,
                },
            });
        }
    }
    
    /// Add BERT model family
    fn add_bert_models(&mut self) {
        let models = vec![
            ("bert-base", 110_000_000, 0.4, 12),
            ("bert-large", 340_000_000, 1.3, 24),
            ("roberta-base", 125_000_000, 0.5, 12),
            ("roberta-large", 355_000_000, 1.4, 24),
        ];
        
        for (name, params, memory, layers) in models {
            self.models.insert(name.to_string(), ModelDefinition {
                name: name.to_string(),
                family: "bert".to_string(),
                parameters: params,
                base_memory_gb: memory,
                min_compute: 2.0,
                supported_quantization: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                ],
                model_type: ModelType::Both,
                context_lengths: vec![128, 256, 512],
                architecture: ModelArchitecture {
                    arch_type: "transformer".to_string(),
                    layers,
                    hidden_size: if layers == 12 { 768 } else { 1024 },
                    attention_heads: Some(if layers == 12 { 12 } else { 16 }),
                    supports_multi_gpu: false,
                },
            });
        }
    }
    
    /// Add vision models
    fn add_vision_models(&mut self) {
        let models = vec![
            ("resnet50", 25_000_000, 0.1, 50),
            ("efficientnet-b0", 5_000_000, 0.02, 16),
            ("efficientnet-b7", 66_000_000, 0.3, 45),
            ("vit-base", 86_000_000, 0.3, 12),
            ("vit-large", 307_000_000, 1.2, 24),
        ];
        
        for (name, params, memory, layers) in models {
            self.models.insert(name.to_string(), ModelDefinition {
                name: name.to_string(),
                family: "vision".to_string(),
                parameters: params,
                base_memory_gb: memory,
                min_compute: 2.0,
                supported_quantization: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                ],
                model_type: ModelType::Both,
                context_lengths: vec![224, 384, 512],
                architecture: ModelArchitecture {
                    arch_type: if name.starts_with("vit") { "transformer" } else { "cnn" }.to_string(),
                    layers,
                    hidden_size: if name.starts_with("vit") { 768 } else { 2048 },
                    attention_heads: if name.starts_with("vit") { Some(12) } else { None },
                    supports_multi_gpu: true,
                },
            });
        }
    }
    
    /// Populate the database with popular model runners
    fn populate_runners(&mut self) {
        // llama.cpp
        self.runners.insert("llama.cpp".to_string(), ModelRunner {
            name: "llama.cpp".to_string(),
            description: "Efficient C++ implementation with CPU/GPU splitting".to_string(),
            supported_families: vec!["llama".to_string(), "gpt".to_string()],
            platforms: vec![Platform::Cuda, Platform::Metal, Platform::OpenCL, Platform::Cpu],
            capabilities: RunnerCapabilities {
                cpu_gpu_split: true,
                dynamic_quantization: true,
                batch_processing: true,
                streaming: true,
                max_context_length: 8192,
                quantization_support: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                    QuantizationLevel::Int4,
                ],
            },
            performance: RunnerPerformance {
                cpu_efficiency: 9.0,
                gpu_efficiency: 8.5,
                memory_efficiency: 9.5,
                setup_overhead: OverheadLevel::Low,
            },
        });
        
        // ONNX Runtime
        self.runners.insert("onnx-runtime".to_string(), ModelRunner {
            name: "ONNX Runtime".to_string(),
            description: "Cross-platform ML inference runtime".to_string(),
            supported_families: vec!["bert".to_string(), "gpt".to_string(), "vision".to_string()],
            platforms: vec![Platform::Cuda, Platform::Cpu, Platform::DirectML],
            capabilities: RunnerCapabilities {
                cpu_gpu_split: false,
                dynamic_quantization: true,
                batch_processing: true,
                streaming: false,
                max_context_length: 2048,
                quantization_support: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                ],
            },
            performance: RunnerPerformance {
                cpu_efficiency: 7.5,
                gpu_efficiency: 8.0,
                memory_efficiency: 7.0,
                setup_overhead: OverheadLevel::Medium,
            },
        });
        
        // Hugging Face Transformers
        self.runners.insert("transformers".to_string(), ModelRunner {
            name: "Hugging Face Transformers".to_string(),
            description: "Python library for transformer models".to_string(),
            supported_families: vec![
                "bert".to_string(), 
                "gpt".to_string(), 
                "llama".to_string(),
                "vision".to_string()
            ],
            platforms: vec![Platform::Cuda, Platform::Cpu],
            capabilities: RunnerCapabilities {
                cpu_gpu_split: false,
                dynamic_quantization: false,
                batch_processing: true,
                streaming: true,
                max_context_length: 4096,
                quantization_support: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                ],
            },
            performance: RunnerPerformance {
                cpu_efficiency: 6.0,
                gpu_efficiency: 7.5,
                memory_efficiency: 6.0,
                setup_overhead: OverheadLevel::High,
            },
        });
        
        // TensorFlow Lite
        self.runners.insert("tflite".to_string(), ModelRunner {
            name: "TensorFlow Lite".to_string(),
            description: "Lightweight inference for mobile and edge".to_string(),
            supported_families: vec!["vision".to_string(), "bert".to_string()],
            platforms: vec![Platform::Cuda, Platform::Cpu],
            capabilities: RunnerCapabilities {
                cpu_gpu_split: false,
                dynamic_quantization: true,
                batch_processing: false,
                streaming: false,
                max_context_length: 512,
                quantization_support: vec![
                    QuantizationLevel::None,
                    QuantizationLevel::Int8,
                    QuantizationLevel::Int4,
                ],
            },
            performance: RunnerPerformance {
                cpu_efficiency: 8.5,
                gpu_efficiency: 6.0,
                memory_efficiency: 9.0,
                setup_overhead: OverheadLevel::Low,
            },
        });
    }
}

impl Default for ModelDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Get a pre-populated model database with popular AI/ML models
pub fn get_model_database() -> ModelDatabase {
    let mut db = ModelDatabase::new();
    
    // Add a sample Llama 7B model
    db.add_model("llama-7b".to_string(), ModelDefinition {
        name: "Llama 7B".to_string(),
        family: "llama".to_string(),
        parameters: 7_000_000_000,
        base_memory_gb: 14.0,
        min_compute: 5.0,
        supported_quantization: vec![
            crate::workloads::QuantizationLevel::None,
            crate::workloads::QuantizationLevel::Int8,
            crate::workloads::QuantizationLevel::Int4,
        ],
        model_type: ModelType::Both,
        context_lengths: vec![2048, 4096],
        architecture: ModelArchitecture {
            arch_type: "Transformer".to_string(),
            layers: 32,
            hidden_size: 4096,
            attention_heads: Some(32),
            supports_multi_gpu: true,
        },
    });
    
    db
}

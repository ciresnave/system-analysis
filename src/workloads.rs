//! Workload definitions and modeling.

use crate::resources::{ResourceRequirement, ResourceType, CapabilityLevel};
use crate::types::SystemProfile;
use crate::error::{SystemAnalysisError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for workload implementations
pub trait Workload: Send + Sync + std::fmt::Debug {
    /// Get the workload type
    fn workload_type(&self) -> WorkloadType;
    
    /// Get resource requirements for this workload
    fn resource_requirements(&self) -> Vec<ResourceRequirement>;
    
    /// Get estimated resource utilization
    fn estimated_utilization(&self) -> HashMap<ResourceType, f64>;
    
    /// Get performance characteristics
    fn performance_characteristics(&self) -> PerformanceCharacteristics;
    
    /// Get workload metadata
    fn metadata(&self) -> WorkloadMetadata;
    
    /// Validate workload configuration
    fn validate(&self) -> crate::error::Result<()>;
    
    /// Clone the workload (for trait objects)
    fn clone_workload(&self) -> Box<dyn Workload>;
}

/// Types of workloads
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadType {
    /// AI/ML inference workload
    AIInference,
    /// AI/ML training workload
    AITraining,
    /// NPU-optimized inference workload
    NPUInference,
    /// TPU-optimized training workload
    TPUTraining,
    /// FPGA-accelerated inference workload
    FPGAInference,
    /// Edge AI workload (ARM-optimized)
    EdgeAI,
    /// Computer vision workload
    ComputerVision,
    /// Natural language processing workload
    NaturalLanguageProcessing,
    /// Large language model inference
    LLMInference,
    /// Large language model training
    LLMTraining,
    /// Robotics workload (ARM + AI accelerators)
    Robotics,
    /// IoT edge processing
    IoTEdge,
    /// Data processing workload
    DataProcessing,
    /// Web application workload
    WebApplication,
    /// Database workload
    Database,
    /// Compute-intensive workload
    ComputeIntensive,
    /// Memory-intensive workload
    MemoryIntensive,
    /// I/O-intensive workload
    IOIntensive,
    /// Custom workload type
    Custom(String),
}

impl std::fmt::Display for WorkloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkloadType::AIInference => write!(f, "AI Inference"),
            WorkloadType::AITraining => write!(f, "AI Training"),
            WorkloadType::NPUInference => write!(f, "NPU Inference"),
            WorkloadType::TPUTraining => write!(f, "TPU Training"),
            WorkloadType::FPGAInference => write!(f, "FPGA Inference"),
            WorkloadType::EdgeAI => write!(f, "Edge AI"),
            WorkloadType::ComputerVision => write!(f, "Computer Vision"),
            WorkloadType::NaturalLanguageProcessing => write!(f, "Natural Language Processing"),
            WorkloadType::LLMInference => write!(f, "Large Language Model Inference"),
            WorkloadType::LLMTraining => write!(f, "Large Language Model Training"),
            WorkloadType::Robotics => write!(f, "Robotics"),
            WorkloadType::IoTEdge => write!(f, "IoT Edge"),
            WorkloadType::DataProcessing => write!(f, "Data Processing"),
            WorkloadType::WebApplication => write!(f, "Web Application"),
            WorkloadType::Database => write!(f, "Database"),
            WorkloadType::ComputeIntensive => write!(f, "Compute Intensive"),
            WorkloadType::MemoryIntensive => write!(f, "Memory Intensive"),
            WorkloadType::IOIntensive => write!(f, "I/O Intensive"),
            WorkloadType::Custom(name) => write!(f, "Custom: {name}"),
        }
    }
}

impl WorkloadType {
    /// Get estimated resource utilization for this workload type
    pub fn estimated_utilization(&self) -> f64 {
        match self {
            WorkloadType::AIInference => 0.6,
            WorkloadType::AITraining => 0.9,
            WorkloadType::NPUInference => 0.7,
            WorkloadType::TPUTraining => 0.95,
            WorkloadType::FPGAInference => 0.8,
            WorkloadType::EdgeAI => 0.5,
            WorkloadType::ComputerVision => 0.7,
            WorkloadType::NaturalLanguageProcessing => 0.6,
            WorkloadType::LLMInference => 0.8,
            WorkloadType::LLMTraining => 0.95,
            WorkloadType::Robotics => 0.6,
            WorkloadType::IoTEdge => 0.4,
            WorkloadType::DataProcessing => 0.5,
            WorkloadType::WebApplication => 0.3,
            WorkloadType::Database => 0.4,
            WorkloadType::ComputeIntensive => 0.8,
            WorkloadType::MemoryIntensive => 0.6,
            WorkloadType::IOIntensive => 0.7,
            WorkloadType::Custom(_) => 0.5,
        }
    }

    /// Validate workload configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        // Basic validation - all workload types are valid by default
        Ok(())
    }
}

/// Performance characteristics of a workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Expected latency profile
    pub latency_profile: LatencyProfile,
    /// Throughput requirements
    pub throughput_profile: ThroughputProfile,
    /// Scalability characteristics
    pub scalability: ScalabilityProfile,
    /// Resource usage patterns
    pub resource_patterns: ResourcePatterns,
}

/// Latency characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyProfile {
    /// Expected latency in milliseconds
    pub expected_latency_ms: f64,
    /// Acceptable latency in milliseconds
    pub acceptable_latency_ms: f64,
    /// Latency sensitivity
    pub sensitivity: LatencySensitivity,
}

/// Latency sensitivity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LatencySensitivity {
    /// Very low latency sensitivity
    VeryLow,
    /// Low latency sensitivity
    Low,
    /// Medium latency sensitivity
    Medium,
    /// High latency sensitivity
    High,
    /// Very high latency sensitivity (real-time)
    VeryHigh,
}

/// Throughput characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputProfile {
    /// Expected throughput (operations per second)
    pub expected_ops_per_sec: f64,
    /// Minimum acceptable throughput
    pub minimum_ops_per_sec: f64,
    /// Peak throughput requirement
    pub peak_ops_per_sec: f64,
    /// Throughput consistency requirement
    pub consistency_requirement: ThroughputConsistency,
}

/// Throughput consistency requirements
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThroughputConsistency {
    /// Variable throughput acceptable
    Variable,
    /// Consistent throughput preferred
    Consistent,
    /// Guaranteed throughput required
    Guaranteed,
}

/// Scalability characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityProfile {
    /// Horizontal scaling capability
    pub horizontal_scaling: ScalingCapability,
    /// Vertical scaling capability
    pub vertical_scaling: ScalingCapability,
    /// Scaling efficiency
    pub scaling_efficiency: f64,
}

/// Scaling capability levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScalingCapability {
    /// No scaling support
    None,
    /// Limited scaling support
    Limited,
    /// Good scaling support
    Good,
    /// Excellent scaling support
    Excellent,
}

/// Resource usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePatterns {
    /// CPU usage pattern
    pub cpu_pattern: UsagePattern,
    /// Memory usage pattern
    pub memory_pattern: UsagePattern,
    /// I/O usage pattern
    pub io_pattern: UsagePattern,
    /// Network usage pattern
    pub network_pattern: UsagePattern,
}

/// Usage pattern types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UsagePattern {
    /// Constant usage
    Constant,
    /// Bursty usage
    Bursty,
    /// Periodic usage
    Periodic,
    /// Ramping usage
    Ramping,
    /// Unpredictable usage
    Unpredictable,
}

/// Workload metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMetadata {
    /// Workload name
    pub name: String,
    /// Description
    pub description: String,
    /// Version
    pub version: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Vendor/source information
    pub vendor: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Documentation URL
    pub documentation_url: Option<String>,
}

/// AI inference workload implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInferenceWorkload {
    /// Model parameters
    pub model_params: ModelParameters,
    /// Inference configuration
    pub inference_config: InferenceConfig,
    /// Metadata
    pub metadata: WorkloadMetadata,
}

impl AIInferenceWorkload {
    /// Create a new AI inference workload
    pub fn new(model_params: ModelParameters) -> Self {
        Self {
            model_params,
            inference_config: InferenceConfig::default(),
            metadata: WorkloadMetadata {
                name: "AI Inference".to_string(),
                description: "AI/ML model inference workload".to_string(),
                version: "1.0.0".to_string(),
                tags: vec!["ai".to_string(), "inference".to_string(), "ml".to_string()],
                vendor: None,
                license: None,
                documentation_url: None,
            },
        }
    }

    /// Set inference configuration
    pub fn with_config(mut self, config: InferenceConfig) -> Self {
        self.inference_config = config;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: WorkloadMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Workload for AIInferenceWorkload {
    fn workload_type(&self) -> WorkloadType {
        WorkloadType::AIInference
    }

    fn resource_requirements(&self) -> Vec<ResourceRequirement> {
        let mut requirements = Vec::new();

        // Memory requirement based on model size
        let memory_gb = self.model_params.memory_required;
        requirements.push(
            ResourceRequirement::new(ResourceType::Memory)
                .minimum_gb(memory_gb)
                .recommended_gb(memory_gb * 1.5)
                .critical()
        );

        // GPU requirement if preferred
        if self.model_params.prefer_gpu {
            let gpu_level = match self.model_params.parameters {
                params if params >= 70_000_000_000 => CapabilityLevel::Exceptional,
                params if params >= 13_000_000_000 => CapabilityLevel::VeryHigh,
                params if params >= 7_000_000_000 => CapabilityLevel::High,
                params if params >= 1_000_000_000 => CapabilityLevel::Medium,
                _ => CapabilityLevel::Low,
            };

            requirements.push(
                ResourceRequirement::new(ResourceType::GPU)
                    .minimum_level(gpu_level)
                    .preferred_vendor(Some("NVIDIA"))
            );
        }

        // CPU requirement
        let cpu_level = match self.model_params.compute_required {
            compute if compute >= 8.0 => CapabilityLevel::VeryHigh,
            compute if compute >= 6.0 => CapabilityLevel::High,
            compute if compute >= 4.0 => CapabilityLevel::Medium,
            compute if compute >= 2.0 => CapabilityLevel::Low,
            _ => CapabilityLevel::VeryLow,
        };

        requirements.push(
            ResourceRequirement::new(ResourceType::CPU)
                .minimum_level(cpu_level)
        );

        // Storage requirement
        let storage_gb = (self.model_params.parameters as f64 * 4.0 / 1_000_000_000.0) + 10.0; // Rough estimate
        requirements.push(
            ResourceRequirement::new(ResourceType::Storage)
                .minimum_gb(storage_gb)
                .recommended_gb(storage_gb * 2.0)
        );

        requirements
    }

    fn estimated_utilization(&self) -> HashMap<ResourceType, f64> {
        let mut utilization = HashMap::new();

        // CPU utilization depends on whether GPU is available
        let cpu_util = if self.model_params.prefer_gpu { 20.0 } else { 80.0 };
        utilization.insert(ResourceType::CPU, cpu_util);

        // GPU utilization if preferred
        if self.model_params.prefer_gpu {
            utilization.insert(ResourceType::GPU, 75.0);
        }

        // Memory utilization
        utilization.insert(ResourceType::Memory, 60.0);

        // Storage utilization (mostly read)
        utilization.insert(ResourceType::Storage, 15.0);

        // Network utilization (if serving inference)
        utilization.insert(ResourceType::Network, 25.0);

        utilization
    }

    fn performance_characteristics(&self) -> PerformanceCharacteristics {
        let latency_ms = if self.model_params.prefer_gpu { 50.0 } else { 200.0 };
        let throughput = if self.model_params.prefer_gpu { 20.0 } else { 5.0 };

        PerformanceCharacteristics {
            latency_profile: LatencyProfile {
                expected_latency_ms: latency_ms,
                acceptable_latency_ms: latency_ms * 2.0,
                sensitivity: LatencySensitivity::High,
            },
            throughput_profile: ThroughputProfile {
                expected_ops_per_sec: throughput,
                minimum_ops_per_sec: throughput * 0.5,
                peak_ops_per_sec: throughput * 1.5,
                consistency_requirement: ThroughputConsistency::Consistent,
            },
            scalability: ScalabilityProfile {
                horizontal_scaling: ScalingCapability::Good,
                vertical_scaling: ScalingCapability::Excellent,
                scaling_efficiency: 0.8,
            },
            resource_patterns: ResourcePatterns {
                cpu_pattern: if self.model_params.prefer_gpu { UsagePattern::Bursty } else { UsagePattern::Constant },
                memory_pattern: UsagePattern::Constant,
                io_pattern: UsagePattern::Bursty,
                network_pattern: UsagePattern::Bursty,
            },
        }
    }

    fn metadata(&self) -> WorkloadMetadata {
        self.metadata.clone()
    }

    fn validate(&self) -> crate::error::Result<()> {
        if self.model_params.parameters == 0 {
            return Err(crate::error::SystemAnalysisError::invalid_workload(
                "Model parameters cannot be zero"
            ));
        }

        if self.model_params.memory_required <= 0.0 {
            return Err(crate::error::SystemAnalysisError::invalid_workload(
                "Memory requirement must be positive"
            ));
        }

        if self.model_params.compute_required <= 0.0 {
            return Err(crate::error::SystemAnalysisError::invalid_workload(
                "Compute requirement must be positive"
            ));
        }

        Ok(())
    }

    fn clone_workload(&self) -> Box<dyn Workload> {
        Box::new(self.clone())
    }
}

/// Model parameters for AI workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Number of parameters in the model
    pub parameters: u64,
    /// Memory required in GB
    pub memory_required: f64,
    /// Compute intensity (0-10 scale)
    pub compute_required: f64,
    /// Prefer GPU acceleration
    pub prefer_gpu: bool,
    /// Model architecture type
    pub architecture: Option<String>,
    /// Quantization level
    pub quantization: QuantizationLevel,
    /// Context length
    pub context_length: Option<u32>,
    /// Batch size
    pub batch_size: u32,
}

impl ModelParameters {
    /// Create new model parameters
    pub fn new() -> Self {
        Self {
            parameters: 0,
            memory_required: 0.0,
            compute_required: 0.0,
            prefer_gpu: false,
            architecture: None,
            quantization: QuantizationLevel::None,
            context_length: None,
            batch_size: 1,
        }
    }

    /// Set number of parameters
    pub fn parameters(mut self, params: u64) -> Self {
        self.parameters = params;
        // Auto-estimate memory requirement based on parameters
        self.memory_required = (params as f64 * 4.0 / 1_000_000_000.0) * 1.2; // 4 bytes per param + overhead
        // Auto-estimate compute requirement based on parameters
        self.compute_required = (params as f64 / 1_000_000_000.0).clamp(1.0, 10.0); // Scale based on model size
        self
    }

    /// Set memory requirement in GB
    pub fn memory_required(mut self, gb: f64) -> Self {
        self.memory_required = gb;
        self
    }

    /// Set compute requirement (0-10 scale)
    pub fn compute_required(mut self, compute: f64) -> Self {
        self.compute_required = compute.clamp(0.0, 10.0);
        self
    }

    /// Set GPU preference
    pub fn prefer_gpu(mut self, prefer: bool) -> Self {
        self.prefer_gpu = prefer;
        self
    }

    /// Set model architecture
    pub fn architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Set quantization level
    pub fn quantization(mut self, quant: QuantizationLevel) -> Self {
        self.quantization = quant;
        // Adjust memory requirement based on quantization
        match quant {
            QuantizationLevel::None => {},
            QuantizationLevel::Int8 => self.memory_required *= 0.5,
            QuantizationLevel::Int4 => self.memory_required *= 0.25,
            QuantizationLevel::Custom(_) => self.memory_required *= 0.5,
        }
        self
    }

    /// Set context length
    pub fn context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: u32) -> Self {
        self.batch_size = size;
        self
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantization levels for models
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantizationLevel {
    /// No quantization (full precision)
    None,
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// Custom quantization with ratio
    Custom(f64),
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum concurrent requests
    pub max_concurrent_requests: u32,
    /// Request timeout in seconds
    pub request_timeout_sec: u32,
    /// Enable batching
    pub enable_batching: bool,
    /// Dynamic batching settings
    pub dynamic_batching: Option<DynamicBatchingConfig>,
    /// Caching configuration
    pub caching: CachingConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            request_timeout_sec: 30,
            enable_batching: true,
            dynamic_batching: Some(DynamicBatchingConfig::default()),
            caching: CachingConfig::default(),
        }
    }
}

/// Dynamic batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchingConfig {
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Maximum wait time in milliseconds
    pub max_wait_time_ms: u32,
    /// Preferred batch sizes
    pub preferred_batch_sizes: Vec<u32>,
}

impl Default for DynamicBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait_time_ms: 100,
            preferred_batch_sizes: vec![1, 2, 4, 8],
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable KV cache
    pub enable_kv_cache: bool,
    /// Cache size in MB
    pub cache_size_mb: u32,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enable_kv_cache: true,
            cache_size_mb: 1024,
            eviction_policy: CacheEvictionPolicy::LRU,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Random eviction
    Random,
}

/// AI training workload (placeholder for future implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITrainingWorkload {
    /// Model parameters
    pub model_params: ModelParameters,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Metadata
    pub metadata: WorkloadMetadata,
}

/// Training configuration (basic structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size for training
    pub batch_size: u32,
    /// Number of epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Distributed training
    pub distributed: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            epochs: 10,
            learning_rate: 0.001,
            distributed: false,
        }
    }
}

/// Trait for defining custom workload types
/// 
/// Users can implement this trait to create custom workload definitions
/// that integrate seamlessly with the system analysis framework.
pub trait CustomWorkload: Send + Sync {
    /// Get the workload name/identifier
    fn name(&self) -> &str;
    
    /// Get the workload description
    fn description(&self) -> &str;
    
    /// Calculate resource requirements for this workload
    fn resource_requirements(&self) -> Vec<ResourceRequirement>;
    
    /// Estimate performance on given system (0-10 scale)
    fn estimate_performance(&self, system: &SystemProfile) -> Result<f64>;
    
    /// Get workload-specific metadata
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// Validate workload configuration
    fn validate(&self) -> Result<()> {
        Ok(())
    }
    
    /// Get recommended system upgrades for optimal performance
    fn recommended_upgrades(&self, _system: &SystemProfile) -> Vec<String> {
        Vec::new()
    }
}

/// Registry for custom workload types
pub struct WorkloadRegistry {
    workloads: HashMap<String, Box<dyn CustomWorkload>>,
}

impl WorkloadRegistry {
    /// Create a new workload registry
    pub fn new() -> Self {
        Self {
            workloads: HashMap::new(),
        }
    }
    
    /// Register a custom workload type
    pub fn register<W: CustomWorkload + 'static>(&mut self, workload: W) -> Result<()> {
        let name = workload.name().to_string();
        
        // Validate the workload before registration
        workload.validate()?;
        
        self.workloads.insert(name, Box::new(workload));
        Ok(())
    }
    
    /// Get a registered workload by name
    pub fn get(&self, name: &str) -> Option<&dyn CustomWorkload> {
        self.workloads.get(name).map(|w| w.as_ref())
    }
    
    /// List all registered workload names
    pub fn list_workloads(&self) -> Vec<&str> {
        self.workloads.keys().map(|s| s.as_str()).collect()
    }
    
    /// Remove a workload from the registry
    pub fn unregister(&mut self, name: &str) -> bool {
        self.workloads.remove(name).is_some()
    }
    
    /// Get the number of registered workloads
    pub fn count(&self) -> usize {
        self.workloads.len()
    }
}

impl Default for WorkloadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating custom workload definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomWorkloadBuilder {
    name: String,
    description: String,
    requirements: Vec<ResourceRequirement>,
    metadata: HashMap<String, String>,
    performance_formula: Option<String>,
}

impl CustomWorkloadBuilder {
    /// Create a new custom workload builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            requirements: Vec::new(),
            metadata: HashMap::new(),
            performance_formula: None,
        }
    }
    
    /// Set the workload description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
    
    /// Add a resource requirement
    pub fn add_requirement(mut self, requirement: ResourceRequirement) -> Self {
        self.requirements.push(requirement);
        self
    }
    
    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Set performance estimation formula (for advanced users)
    pub fn performance_formula(mut self, formula: impl Into<String>) -> Self {
        self.performance_formula = Some(formula.into());
        self
    }
    
    /// Build the custom workload
    pub fn build(self) -> BuiltCustomWorkload {
        BuiltCustomWorkload {
            name: self.name,
            description: self.description,
            requirements: self.requirements,
            metadata: self.metadata,
            performance_formula: self.performance_formula,
        }
    }
}

/// Built custom workload that implements the CustomWorkload trait
pub struct BuiltCustomWorkload {
    name: String,
    description: String,
    requirements: Vec<ResourceRequirement>,
    metadata: HashMap<String, String>,
    performance_formula: Option<String>,
}

impl CustomWorkload for BuiltCustomWorkload {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn resource_requirements(&self) -> Vec<ResourceRequirement> {
        self.requirements.clone()
    }
    
    fn estimate_performance(&self, system: &SystemProfile) -> Result<f64> {
        // If a custom formula is provided, we could evaluate it here
        // For now, use a simple heuristic based on system scores
        if let Some(_formula) = &self.performance_formula {
            // TODO: Implement formula evaluation
            // For now, fall back to default calculation
        }
        
        // Simple performance estimation based on system scores
        let cpu_score = system.cpu_score() / 10.0;
        let gpu_score = system.gpu_score() / 10.0;
        let memory_score = system.memory_score() / 10.0;
        let storage_score = system.storage_score() / 10.0;
        
        // Weight the scores based on requirements
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;
        
        for req in &self.requirements {
            let (score, weight) = match req.resource_type {
                crate::resources::ResourceType::CPU => (cpu_score, 1.0),
                crate::resources::ResourceType::GPU => (gpu_score, 1.0),
                crate::resources::ResourceType::Memory => (memory_score, 1.0),
                crate::resources::ResourceType::Storage => (storage_score, 1.0),
                crate::resources::ResourceType::Network => (0.5, 0.5), // Default network score
                crate::resources::ResourceType::Custom(_) => (0.5, 0.5), // Default custom score
            };
            
            weighted_score += score * weight;
            total_weight += weight;
        }
        
        if total_weight == 0.0 {
            Ok(5.0) // Default score if no requirements
        } else {
            Ok((weighted_score / total_weight * 10.0).clamp(0.0, 10.0))
        }
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }
    
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(SystemAnalysisError::invalid_workload(
                "Workload name cannot be empty".to_string()
            ));
        }
        
        if self.requirements.is_empty() {
            return Err(SystemAnalysisError::invalid_workload(
                "Workload must have at least one resource requirement".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn recommended_upgrades(&self, system: &SystemProfile) -> Vec<String> {
        let mut upgrades = Vec::new();
        
        for req in &self.requirements {
            match req.resource_type {
                crate::resources::ResourceType::CPU => {
                    if system.cpu_score < 7.0 {
                        upgrades.push("Consider upgrading to a higher-performance CPU".to_string());
                    }
                },
                crate::resources::ResourceType::GPU => {
                    if system.gpu_score < 7.0 {
                        upgrades.push("Consider upgrading to a more powerful GPU".to_string());
                    }
                },
                crate::resources::ResourceType::Memory => {
                    let system_memory_gb = system.system_info.memory_info.total_ram as f64 / 1024.0; // Convert MB to GB
                    if let crate::resources::ResourceAmount::Gigabytes(min_gb) = req.minimum {
                        if min_gb > system_memory_gb {
                            upgrades.push(format!(
                                "Increase RAM to at least {min_gb:.1}GB"
                            ));
                        }
                    }
                },
                crate::resources::ResourceType::Storage => {
                    if system.storage_score < 7.0 {
                        upgrades.push("Consider upgrading to faster storage (NVMe SSD)".to_string());
                    }
                },
                crate::resources::ResourceType::Network => {
                    // Network upgrade suggestions would need network capability detection
                },
                crate::resources::ResourceType::Custom(_) => {
                    // Custom resource upgrade suggestions would be workload-specific
                },
            }
        }
        
        upgrades
    }
}

/// Comprehensive AI model definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModel {
    /// Model name
    pub name: String,
    
    /// Model family/architecture
    pub architecture: String,
    
    /// Number of parameters
    pub parameters: u64,
    
    /// Model size in bytes (unquantized)
    pub size_in_bytes: u64,
    
    /// Memory required in GB
    pub memory_required: f64,
    
    /// Current quantization level
    pub quantization: QuantizationLevel,
    
    /// Framework (TensorFlow, PyTorch, ONNX, etc.)
    pub framework: String,
    
    /// Task type (classification, detection, generation, etc.)
    pub task: AITaskType,
    
    /// Maximum input size (tokens, pixels, etc.)
    pub max_input_size: Option<u32>,
    
    /// Maximum context length for sequence models
    pub max_context_length: Option<u32>,
    
    /// Model version
    pub version: String,
    
    /// Additional model metadata
    pub metadata: HashMap<String, String>,
}

impl AIModel {
    /// Create a new AI model
    pub fn new(name: impl Into<String>, architecture: impl Into<String>, parameters: u64) -> Self {
        let params = parameters;
        // Estimate size based on parameters (4 bytes per parameter by default)
        let size_bytes = params * 4;
        // Estimate memory required with overhead
        let memory_gb = (size_bytes as f64 / 1_073_741_824.0) * 1.2; // 20% overhead
        
        Self {
            name: name.into(),
            architecture: architecture.into(),
            parameters: params,
            size_in_bytes: size_bytes,
            memory_required: memory_gb,
            quantization: QuantizationLevel::None,
            framework: "Unknown".to_string(),
            task: AITaskType::Other,
            max_input_size: None,
            max_context_length: None,
            version: "1.0.0".to_string(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set the model framework
    pub fn with_framework(mut self, framework: impl Into<String>) -> Self {
        self.framework = framework.into();
        self
    }
    
    /// Set the model task type
    pub fn with_task(mut self, task: AITaskType) -> Self {
        self.task = task;
        self
    }
    
    /// Set the quantization level
    pub fn with_quantization(mut self, level: QuantizationLevel) -> Self {
        self.quantization = level;
        
        // Adjust memory requirements based on quantization
        match level {
            QuantizationLevel::None => {},
            QuantizationLevel::Int8 => self.memory_required *= 0.5,  // Half precision
            QuantizationLevel::Int4 => self.memory_required *= 0.25, // Quarter precision
            QuantizationLevel::Custom(ratio) => self.memory_required *= ratio,
        }
        
        self
    }
    
    /// Set maximum context length
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.max_context_length = Some(length);
        self
    }
    
    /// Add model metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Types of AI tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AITaskType {
    /// Text generation
    TextGeneration,
    
    /// Image classification
    ImageClassification,
    
    /// Object detection
    ObjectDetection,
    
    /// Image segmentation
    ImageSegmentation,
    
    /// Natural language processing
    NLP,
    
    /// Speech recognition
    SpeechRecognition,
    
    /// Speech synthesis
    SpeechSynthesis,
    
    /// Translation
    Translation,
    
    /// Recommendation
    Recommendation,
    
    /// Other task types
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_parameters_creation() {
        let params = ModelParameters::new()
            .parameters(1_000_000_000) // 1B parameters
            .quantization(QuantizationLevel::Int8)
            .context_length(2048)
            .batch_size(4);

        assert_eq!(params.parameters, 1_000_000_000);
        assert_eq!(params.quantization, QuantizationLevel::Int8);
        assert_eq!(params.context_length, Some(2048));
        assert_eq!(params.batch_size, 4);
    }

    #[test]
    fn test_ai_inference_workload_creation() {
        let model_params = ModelParameters::new()
            .parameters(7_000_000_000) // 7B model
            .context_length(4096);

        let workload = AIInferenceWorkload::new(model_params);
        let requirements = workload.resource_requirements();
        
        assert!(!requirements.is_empty());
        assert_eq!(workload.workload_type(), WorkloadType::AIInference);
    }

    #[test]
    fn test_workload_type_display() {
        assert_eq!(format!("{}", WorkloadType::AIInference), "AI Inference");
        assert_eq!(format!("{}", WorkloadType::AIInference), "AI Inference");
        assert_eq!(format!("{}", WorkloadType::DataProcessing), "Data Processing");
        assert_eq!(format!("{}", WorkloadType::WebApplication), "Web Application");
        assert_eq!(format!("{}", WorkloadType::Custom("Custom Task".to_string())), "Custom: Custom Task");
    }

    #[test]
    fn test_quantization_level() {
        // Test that quantization levels exist and can be used
        let _none = QuantizationLevel::None;
        let _int8 = QuantizationLevel::Int8;
        let _int4 = QuantizationLevel::Int4;
        let _custom = QuantizationLevel::Custom(0.5);
        
        // Basic functionality test - check that all variants can be created
        assert_eq!(format!("{_none:?}"), "None");
        assert_eq!(format!("{_int8:?}"), "Int8");
        assert_eq!(format!("{_int4:?}"), "Int4");
        assert_eq!(format!("{_custom:?}"), "Custom(0.5)");
    }

    #[test]
    fn test_workload_validation() {
        let model_params = ModelParameters::new()
            .parameters(1_000_000_000);

        let workload = AIInferenceWorkload::new(model_params);
        
        // Validation should pass for reasonable parameters
        assert!(workload.validate().is_ok());
    }

    #[test]
    fn test_model_parameters_builder() {
        let params = ModelParameters::new()
            .parameters(1_000_000_000)
            .memory_required(8.0)
            .compute_required(7.5)
            .prefer_gpu(true)
            .architecture("transformer")
            .quantization(QuantizationLevel::Int8)
            .context_length(2048)
            .batch_size(4);

        assert_eq!(params.parameters, 1_000_000_000);
        assert_eq!(params.compute_required, 7.5);
        assert!(params.prefer_gpu);
        assert_eq!(params.architecture, Some("transformer".to_string()));
        assert_eq!(params.context_length, Some(2048));
        assert_eq!(params.batch_size, 4);
    }
}
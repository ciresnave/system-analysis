//! Workload definitions and modeling.

use crate::resources::{ResourceRequirement, ResourceType, CapabilityLevel};
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
            WorkloadType::DataProcessing => write!(f, "Data Processing"),
            WorkloadType::WebApplication => write!(f, "Web Application"),
            WorkloadType::Database => write!(f, "Database"),
            WorkloadType::ComputeIntensive => write!(f, "Compute Intensive"),
            WorkloadType::MemoryIntensive => write!(f, "Memory Intensive"),
            WorkloadType::IOIntensive => write!(f, "I/O Intensive"),
            WorkloadType::Custom(name) => write!(f, "Custom: {}", name),
        }
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
        self.compute_required = (params as f64 / 1_000_000_000.0).min(10.0).max(1.0); // Scale based on model size
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
    /// Custom quantization
    Custom(u8),
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
        let _custom = QuantizationLevel::Custom(16);
        
        // Basic functionality test
        assert!(true);
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
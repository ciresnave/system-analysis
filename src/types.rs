//! Core types for system analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a complete system profile with all capability scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemProfile {
    /// CPU performance score (0-10)
    pub cpu_score: f64,
    /// GPU performance score (0-10)
    pub gpu_score: f64,
    /// NPU (Neural Processing Unit) performance score (0-10)
    pub npu_score: f64,
    /// TPU (Tensor Processing Unit) performance score (0-10)
    pub tpu_score: f64,
    /// FPGA acceleration performance score (0-10)
    pub fpga_score: f64,
    /// Combined AI accelerator score (0-10)
    pub ai_accelerator_score: f64,
    /// ARM system optimization score (0-10) - for edge computing
    pub arm_optimization_score: f64,
    /// Memory performance score (0-10)
    pub memory_score: f64,
    /// Storage performance score (0-10)
    pub storage_score: f64,
    /// Network performance score (0-10)
    pub network_score: f64,
    /// Overall system score (0-10)
    pub overall_score: f64,
    /// AI workload suitability score (0-10)
    pub ai_workload_score: f64,
    /// Edge computing suitability score (0-10)
    pub edge_computing_score: f64,
    /// Detailed system information
    pub system_info: SystemInfo,
    /// Timestamp when profile was created
    pub created_at: DateTime<Utc>,
}

impl SystemProfile {
    /// Create a new system profile with comprehensive AI accelerator scoring
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cpu_score: f64,
        gpu_score: f64,
        npu_score: f64,
        tpu_score: f64,
        fpga_score: f64,
        arm_optimization_score: f64,
        memory_score: f64,
        storage_score: f64,
        network_score: f64,
        system_info: SystemInfo,
    ) -> Self {
        SystemProfileBuilder::new()
            .cpu_score(cpu_score)
            .gpu_score(gpu_score)
            .npu_score(npu_score)
            .tpu_score(tpu_score)
            .fpga_score(fpga_score)
            .arm_optimization_score(arm_optimization_score)
            .memory_score(memory_score)
            .storage_score(storage_score)
            .network_score(network_score)
            .system_info(system_info)
            .build()
    }

    /// Create a builder for SystemProfile
    pub fn builder() -> SystemProfileBuilder {
        SystemProfileBuilder::new()
    }

    /// Get the CPU score
    pub fn cpu_score(&self) -> f64 {
        self.cpu_score
    }

    /// Get the GPU score
    pub fn gpu_score(&self) -> f64 {
        self.gpu_score
    }

    /// Get the memory score
    pub fn memory_score(&self) -> f64 {
        self.memory_score
    }

    /// Get the storage score
    pub fn storage_score(&self) -> f64 {
        self.storage_score
    }

    /// Get the network score
    pub fn network_score(&self) -> f64 {
        self.network_score
    }

    /// Get the overall score
    pub fn overall_score(&self) -> f64 {
        self.overall_score
    }
    
    /// Get AI workload suitability score (0-10)
    pub fn ai_workload_score(&self) -> f64 {
        self.ai_workload_score
    }
    
    /// Get edge computing suitability score (0-10)
    pub fn edge_computing_score(&self) -> f64 {
        self.edge_computing_score
    }
    
    /// Get combined AI accelerator score (0-10)
    pub fn ai_accelerator_score(&self) -> f64 {
        self.ai_accelerator_score
    }
    
    /// Check if system has AI accelerators
    pub fn has_ai_accelerators(&self) -> bool {
        !self.system_info.npu_info.is_empty() || 
        !self.system_info.tpu_info.is_empty() || 
        !self.system_info.fpga_info.is_empty()
    }
    
    /// Get total TOPS (Tera Operations Per Second) performance
    pub fn total_tops_performance(&self) -> f64 {
        let npu_tops: f64 = self.system_info.npu_info.iter()
            .filter_map(|npu| npu.tops_performance)
            .sum();
        let tpu_tops: f64 = self.system_info.tpu_info.iter()
            .filter_map(|tpu| tpu.tops_performance)
            .sum();
        npu_tops + tpu_tops
    }
    
    /// Check if system is ARM-based
    pub fn is_arm_system(&self) -> bool {
        self.system_info.arm_info.is_some()
    }
    
    /// Get ARM system type if available
    pub fn arm_system_type(&self) -> Option<&str> {
        self.system_info.arm_info.as_ref().map(|arm| arm.system_type.as_str())
    }
    
    /// Check if system is suitable for specific AI workload type
    pub fn is_suitable_for_ai_workload(&self, workload_type: &str) -> bool {
        match workload_type.to_lowercase().as_str() {
            "inference" => self.ai_workload_score >= 6.0,
            "training" => self.ai_workload_score >= 8.0 && self.gpu_score >= 7.0,
            "edge" => self.edge_computing_score >= 7.0,
            "lightweight" => self.ai_workload_score >= 4.0,
            _ => self.ai_workload_score >= 6.0,
        }
    }
    
    /// Generate a hash for this system profile for grouping similar systems
    pub fn system_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash key system characteristics
        self.cpu_score.to_bits().hash(&mut hasher);
        self.gpu_score.to_bits().hash(&mut hasher);
        self.memory_score.to_bits().hash(&mut hasher);
        self.system_info.cpu_info.physical_cores.hash(&mut hasher);
        self.system_info.cpu_info.logical_cores.hash(&mut hasher);
        self.system_info.memory_info.total_ram.hash(&mut hasher);
        self.system_info.gpu_info.len().hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }

    /// Get AI capabilities assessment
    pub fn ai_capabilities(&self) -> AICapabilities {
        AICapabilities {
            inference_capability: self.neural_inference_capability(),
            training_capability: self.neural_training_capability(),
            edge_capability: self.edge_computing_capability(),
            max_model_size: self.max_supported_model_size(),
            llm_capability: self.llm_capability(),
            computer_vision_capability: self.computer_vision_capability(),
            supported_frameworks: self.supported_frameworks(),
        }
    }

    /// Evaluate system suitability for neural network inference
    pub fn neural_inference_capability(&self) -> CapabilityLevel {
        // Calculate based on NPU/TPU scores and features
        if self.ai_accelerator_score >= 8.0 {
            CapabilityLevel::Exceptional
        } else if self.ai_accelerator_score >= 5.0 {
            CapabilityLevel::VeryHigh
        } else if self.gpu_score >= 7.0 {
            CapabilityLevel::High
        } else if self.gpu_score >= 5.0 {
            CapabilityLevel::Medium
        } else {
            CapabilityLevel::Basic
        }
    }
    
    /// Evaluate system suitability for neural network training
    pub fn neural_training_capability(&self) -> CapabilityLevel {
        // Training needs more power than inference
        if self.ai_accelerator_score >= 9.0 && self.memory_score >= 8.0 {
            CapabilityLevel::Exceptional
        } else if self.gpu_score >= 8.0 && self.memory_score >= 7.0 {
            CapabilityLevel::VeryHigh
        } else if self.gpu_score >= 6.0 {
            CapabilityLevel::High
        } else if self.gpu_score >= 4.0 {
            CapabilityLevel::Medium
        } else {
            CapabilityLevel::Basic
        }
    }
    
    /// Evaluate system suitability for edge computing
    pub fn edge_computing_capability(&self) -> CapabilityLevel {
        // Edge computing focuses on efficiency
        if self.edge_computing_score >= 8.0 {
            CapabilityLevel::Exceptional
        } else if self.edge_computing_score >= 6.0 {
            CapabilityLevel::VeryHigh
        } else if self.edge_computing_score >= 4.0 {
            CapabilityLevel::Medium
        } else {
            CapabilityLevel::Basic
        }
    }
    
    /// Get maximum supported model size in gigabytes
    pub fn max_supported_model_size(&self) -> f64 {
        // Calculate based on available memory and accelerators
        let mut base_memory_gb = self.system_info.memory_info.total_ram as f64 / 1024.0 / 2.0; // Half of system RAM
        
        // Check GPU VRAM if available
        let gpu_memory_gb = self.system_info.gpu_info.iter()
            .filter_map(|gpu| gpu.vram_size)
            .map(|vram| vram as f64 / 1024.0) // Convert MB to GB
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        if gpu_memory_gb > 0.0 {
            // For GPU inference, use VRAM size with a safety margin
            base_memory_gb = base_memory_gb.max(gpu_memory_gb * 0.8);
        }
        
        // Adjust for quantization capabilities
        if !self.system_info.npu_info.is_empty() || !self.system_info.tpu_info.is_empty() {
            // NPUs/TPUs often support efficient quantization
            base_memory_gb *= 2.0; // Double effective capacity with quantization
        } else if self.gpu_score >= 7.0 {
            // Modern GPUs can handle quantized models
            base_memory_gb *= 1.5; // 50% more with quantization
        }
        
        base_memory_gb
    }
    
    /// Evaluate system suitability for large language models
    pub fn llm_capability(&self) -> LLMCapability {
        let max_model_size = self.max_supported_model_size();
        
        if max_model_size >= 100.0 && self.ai_accelerator_score >= 8.0 {
            LLMCapability::Enterprise // 100+ GB models
        } else if max_model_size >= 40.0 && self.ai_workload_score >= 7.0 {
            LLMCapability::Advanced // 40+ GB models (e.g., 65B parameters)
        } else if max_model_size >= 16.0 && self.ai_workload_score >= 6.0 {
            LLMCapability::Standard // 16+ GB models (e.g., 13B parameters)
        } else if max_model_size >= 8.0 && self.ai_workload_score >= 5.0 {
            LLMCapability::Basic // 8+ GB models (e.g., 7B parameters)
        } else if max_model_size >= 4.0 {
            LLMCapability::Minimal // Small models only (e.g., 3B parameters)
        } else {
            LLMCapability::Unsuitable
        }
    }
    
    /// Evaluate system suitability for computer vision tasks
    pub fn computer_vision_capability(&self) -> CapabilityLevel {
        if self.ai_accelerator_score >= 7.0 || self.gpu_score >= 8.0 {
            CapabilityLevel::Exceptional // Real-time high-res processing
        } else if self.gpu_score >= 6.0 {
            CapabilityLevel::VeryHigh // Fast high-res processing
        } else if self.gpu_score >= 4.0 {
            CapabilityLevel::High // Decent performance
        } else if self.gpu_score >= 2.0 {
            CapabilityLevel::Medium // Basic performance
        } else {
            CapabilityLevel::Basic // Minimal capability
        }
    }
    
    /// Get list of supported AI frameworks based on available hardware
    pub fn supported_frameworks(&self) -> Vec<String> {
        let mut frameworks = Vec::new();
        
        // Add CPU frameworks
        frameworks.push("ONNX Runtime".to_string());
        frameworks.push("TensorFlow Lite".to_string());
        
        // Add GPU frameworks
        if !self.system_info.gpu_info.is_empty() {
            let has_cuda = self.system_info.gpu_info.iter().any(|gpu| gpu.cuda_support);
            if has_cuda {
                frameworks.push("TensorFlow".to_string());
                frameworks.push("PyTorch".to_string());
                frameworks.push("JAX".to_string());
            }
            
            frameworks.push("OpenVINO".to_string());
            frameworks.push("DirectML".to_string());
        }
        
        // Add NPU frameworks
        for npu in &self.system_info.npu_info {
            frameworks.extend(npu.supported_frameworks.clone());
        }
        
        // Add TPU frameworks
        for tpu in &self.system_info.tpu_info {
            frameworks.extend(tpu.supported_frameworks.clone());
        }
        
        // De-duplicate
        frameworks.sort();
        frameworks.dedup();
        
        frameworks
    }

    // ...existing code...
}

/// Detailed system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system name
    pub os_name: String,
    /// OS version
    pub os_version: String,
    /// CPU information
    pub cpu_info: CpuInfo,
    /// GPU information
    pub gpu_info: Vec<GpuInfo>,
    /// NPU information (Neural Processing Units)
    pub npu_info: Vec<NpuInfo>,
    /// TPU information (Tensor Processing Units)  
    pub tpu_info: Vec<TpuInfo>,
    /// FPGA information
    pub fpga_info: Vec<FpgaInfo>,
    /// ARM-specific hardware information
    pub arm_info: Option<ArmInfo>,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// Storage information
    pub storage_info: Vec<StorageInfo>,
    /// Network information
    pub network_info: NetworkInfo,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU brand/model
    pub brand: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores
    pub logical_cores: usize,
    /// Base frequency in MHz
    pub base_frequency: u64,
    /// Maximum frequency in MHz
    pub max_frequency: Option<u64>,
    /// Cache size in MB
    pub cache_size: Option<u64>,
    /// Architecture (x86_64, arm64, etc.)
    pub architecture: String,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// GPU vendor (NVIDIA, AMD, Intel, etc.)
    pub vendor: String,
    /// VRAM size in MB
    pub vram_size: Option<u64>,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<String>,
    /// OpenCL support
    pub opencl_support: bool,
    /// CUDA support
    pub cuda_support: bool,
}

/// NPU (Neural Processing Unit) information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpuInfo {
    /// NPU vendor
    pub vendor: String,
    /// NPU model name
    pub model_name: String,
    /// Performance in TOPS (Tera Operations Per Second)
    pub tops_performance: Option<f64>,
    /// Supported frameworks
    pub supported_frameworks: Vec<String>,
    /// Supported data types
    pub supported_dtypes: Vec<String>,
}

/// TPU (Tensor Processing Unit) information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuInfo {
    /// TPU vendor
    pub vendor: String,
    /// TPU model name
    pub model_name: String,
    /// TPU architecture
    pub architecture: String,
    /// Performance in TOPS
    pub tops_performance: Option<f64>,
    /// Supported frameworks
    pub supported_frameworks: Vec<String>,
    /// Supported data types
    pub supported_dtypes: Vec<String>,
}

/// FPGA information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaInfo {
    /// FPGA vendor
    pub vendor: String,
    /// FPGA family/series
    pub family: String,
    /// Model name
    pub model_name: String,
    /// Logic elements count
    pub logic_elements: Option<u64>,
    /// Memory blocks
    pub memory_blocks: Option<u64>,
    /// DSP blocks for AI acceleration
    pub dsp_blocks: Option<u64>,
}

/// ARM-specific hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmInfo {
    /// ARM system type (Raspberry Pi, Jetson, Apple Silicon, etc.)
    pub system_type: String,
    /// Board model
    pub board_model: String,
    /// CPU architecture details
    pub cpu_architecture: String,
    /// Available acceleration features
    pub acceleration_features: Vec<String>,
    /// ML/AI capabilities
    pub ml_capabilities: HashMap<String, String>,
    /// GPIO/interface availability
    pub interfaces: Vec<String>,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total RAM in MB
    pub total_ram: u64,
    /// Available RAM in MB
    pub available_ram: u64,
    /// Memory type (DDR4, DDR5, etc.)
    pub memory_type: Option<String>,
    /// Memory speed in MHz
    pub memory_speed: Option<u64>,
}

/// Storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Storage device name
    pub name: String,
    /// Storage type (SSD, HDD, NVMe, etc.)
    pub storage_type: String,
    /// Total capacity in GB
    pub total_capacity: u64,
    /// Available capacity in GB
    pub available_capacity: u64,
    /// Read speed in MB/s
    pub read_speed: Option<u64>,
    /// Write speed in MB/s
    pub write_speed: Option<u64>,
}

/// Network information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Internet connectivity
    pub internet_connected: bool,
    /// Estimated bandwidth in Mbps
    pub estimated_bandwidth: Option<u64>,
}

/// Network interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// Interface type (Ethernet, WiFi, etc.)
    pub interface_type: String,
    /// MAC address
    pub mac_address: String,
    /// IP addresses
    pub ip_addresses: Vec<String>,
    /// Connection speed in Mbps
    pub speed: Option<u64>,
}

/// AI capabilities assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AICapabilities {
    /// Neural network inference capability
    pub inference_capability: CapabilityLevel,
    
    /// Neural network training capability
    pub training_capability: CapabilityLevel,
    
    /// Edge computing capability
    pub edge_capability: CapabilityLevel,
    
    /// Maximum supported model size in GB
    pub max_model_size: f64,
    
    /// Large language model capability
    pub llm_capability: LLMCapability,
    
    /// Computer vision capability
    pub computer_vision_capability: CapabilityLevel,
    
    /// Supported frameworks
    pub supported_frameworks: Vec<String>,
}

/// Capability levels for AI features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityLevel {
    /// Basic capability
    Basic,
    
    /// Medium capability
    Medium,
    
    /// High capability
    High,
    
    /// Very high capability
    VeryHigh,
    
    /// Exceptional capability
    Exceptional,
}

/// Large Language Model capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLMCapability {
    /// Unsuitable for LLMs
    Unsuitable,
    
    /// Minimal capability (smallest models only)
    Minimal,
    
    /// Basic capability (small models)
    Basic,
    
    /// Standard capability (medium models)
    Standard,
    
    /// Advanced capability (large models)
    Advanced,
    
    /// Enterprise capability (largest models)
    Enterprise,
}

/// AI accelerator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AIAcceleratorType {
    /// GPU (Graphics Processing Unit)
    GPU,
    /// NPU (Neural Processing Unit)
    NPU,
    /// TPU (Tensor Processing Unit)
    TPU,
    /// FPGA (Field-Programmable Gate Array)
    FPGA,
    /// CPU (Central Processing Unit)
    CPU,
}

/// Specialized requirements for AI workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIWorkloadRequirements {
    /// Base workload requirements
    pub base_requirements: String, // Reference to WorkloadRequirements by name
    
    /// Required AI accelerator types
    pub required_accelerator_types: Vec<AIAcceleratorType>,
    
    /// Minimum TOPS (Tera Operations Per Second) required
    pub required_tops: Option<f64>,
    
    /// Preferred accelerator type
    pub preferred_accelerator: Option<AIAcceleratorType>,
    
    /// Model memory requirements in GB
    pub required_model_memory: f64,
    
    /// Quantization formats supported by this workload
    pub supported_quantization: Vec<crate::workloads::QuantizationLevel>,
    
    /// Minimum inference speed in samples/second
    pub min_inference_speed: Option<f64>,
    
    /// Required framework support (TensorFlow, PyTorch, etc.)
    pub required_frameworks: Vec<String>,
}

/// Result of model compatibility analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompatibilityResult {
    /// Can the model run on this system
    pub can_run: bool,
    
    /// Is there sufficient memory
    pub memory_sufficient: bool,
    
    /// Accelerator compatibility details
    pub accelerator_compatibility: AcceleratorCompatibility,
    
    /// Optimal quantization suggestion
    pub optimal_quantization: QuantizationSuggestion,
    
    /// Expected inference speed (samples/second)
    pub expected_inference_speed: f64,
    
    /// Bottlenecks for this model
    pub bottlenecks: Vec<ModelBottleneck>,
    
    /// Recommended batch size
    pub recommended_batch_size: u32,
}

/// Accelerator compatibility details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorCompatibility {
    /// Is compatible with at least one accelerator
    pub is_compatible: bool,
    
    /// List of compatible devices
    pub compatible_devices: Vec<AcceleratorDevice>,
    
    /// Recommended device for best performance
    pub recommended_device: Option<AcceleratorDevice>,
    
    /// Expected performance level
    pub expected_performance: PerformanceLevel,
}

/// Types of accelerator devices
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorDevice {
    /// CPU
    CPU,
    /// GPU
    GPU,
    /// NPU (Neural Processing Unit)
    NPU,
    /// TPU (Tensor Processing Unit)
    TPU,
    /// FPGA (Field-Programmable Gate Array)
    FPGA,
}

/// Quantization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSuggestion {
    /// Recommended quantization level
    pub recommended_level: crate::workloads::QuantizationLevel,
    
    /// Reasoning for this recommendation
    pub reasoning: String,
    
    /// Expected impact on performance
    pub performance_impact: PerformanceImpact,
}

/// Impact of quantization on performance
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceImpact {
    /// Positive impact (faster)
    Positive,
    
    /// Negative impact (slower)
    Negative,
    
    /// Mixed impact (faster but less accurate)
    Mixed,
    
    /// No significant impact
    None,
}

/// Performance levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceLevel {
    /// Very low performance
    VeryLow,
    
    /// Low performance
    Low,
    
    /// Medium performance
    Medium,
    
    /// High performance
    High,
    
    /// Very high performance
    VeryHigh,
}

/// Model-specific bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBottleneck {
    /// Bottleneck type
    pub bottleneck_type: ModelBottleneckType,
    
    /// Detailed description
    pub description: String,
    
    /// Severity of the bottleneck
    pub severity: BottleneckSeverity,
    
    /// Recommendation to address the bottleneck
    pub recommendation: String,
}

/// Types of model bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelBottleneckType {
    /// Memory bottleneck
    Memory,
    
    /// Compute bottleneck
    Compute,
    
    /// Data transfer bottleneck
    DataTransfer,
    
    /// Precision bottleneck
    Precision,
    
    /// Framework support bottleneck
    FrameworkSupport,
}

/// Severity of bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Low severity
    Low,
    
    /// Medium severity
    Medium,
    
    /// High severity
    High,
    
    /// Critical severity
    Critical,
}

/// Cost estimate for upgrades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Minimum cost in USD
    pub min_cost_usd: f64,
    
    /// Maximum cost in USD
    pub max_cost_usd: f64,
    
    /// Currency
    pub currency: String,
    
    /// Cost breakdown
    pub breakdown: Vec<CostBreakdownItem>,
}

/// Cost breakdown item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdownItem {
    /// Component name
    pub component: String,
    
    /// Cost in USD
    pub cost_usd: f64,
    
    /// Description
    pub description: String,
}

/// Upgrade priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpgradePriority {
    /// Low priority
    Low,
    
    /// Medium priority
    Medium,
    
    /// High priority
    High,
    
    /// Critical priority
    Critical,
}

/// Workload priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadPriority {
    /// Low priority
    Low,
    
    /// Medium priority
    Medium,
    
    /// High priority
    High,
    
    /// Critical priority
    Critical,
}

/// Requirement severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementSeverity {
    /// Low severity
    Low,
    
    /// Medium severity
    Medium,
    
    /// High severity
    High,
    
    /// Critical severity
    Critical,
}

/// System compatibility result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// Whether the system is compatible
    pub is_compatible: bool,
    
    /// Compatibility score (0.0 to 1.0)
    pub score: f64,
    
    /// Performance estimate
    pub performance_estimate: PerformanceEstimate,
    
    /// Missing requirements
    pub missing_requirements: Vec<MissingRequirement>,
    
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Performance estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEstimate {
    /// Expected performance tier
    pub tier: PerformanceTier,
    
    /// Expected utilization percentage
    pub utilization_percent: f64,
    
    /// Estimated latency in milliseconds
    pub latency_ms: f64,
    
    /// Estimated throughput
    pub throughput: f64,
    
    /// Estimated latency in milliseconds (alternative name)
    pub estimated_latency_ms: f64,
    
    /// Estimated throughput (alternative name)
    pub estimated_throughput: f64,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Performance tier (alternative name)
    pub performance_tier: PerformanceTier,
}

/// Performance tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTier {
    /// Low performance
    Low,
    
    /// Medium performance
    Medium,
    
    /// High performance
    High,
    
    /// Very high performance
    VeryHigh,
    
    /// Excellent performance
    Excellent,
    
    /// Good performance
    Good,
    
    /// Fair performance
    Fair,
    
    /// Poor performance
    Poor,
}

/// Missing requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingRequirement {
    /// Resource type
    pub resource_type: String,
    
    /// Required amount
    pub required: String,
    
    /// Current amount
    pub current: String,
    
    /// Available amount
    pub available: String,
    
    /// Severity
    pub severity: RequirementSeverity,
}

/// System bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Resource type causing bottleneck
    pub resource_type: String,
    
    /// Description
    pub description: String,
    
    /// Impact level
    pub impact: BottleneckImpact,
    
    /// Recommended solution
    pub solution: String,
    
    /// Additional suggestions
    pub suggestions: String,
}

/// Bottleneck impact levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckImpact {
    /// Low impact
    Low,
    
    /// Medium impact
    Medium,
    
    /// High impact
    High,
    
    /// Critical impact
    Critical,
    
    /// Severe impact
    Severe,
    
    /// Moderate impact
    Moderate,
    
    /// Minor impact
    Minor,
}

/// Upgrade recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeRecommendation {
    /// Component to upgrade
    pub component: String,
    
    /// Description
    pub description: String,
    
    /// Priority
    pub priority: UpgradePriority,
    
    /// Estimated cost
    pub estimated_cost: Option<CostEstimate>,
    
    /// Resource type
    pub resource_type: crate::resources::ResourceType,
    
    /// Recommendation text
    pub recommendation: String,
    
    /// Estimated improvement
    pub estimated_improvement: String,
    
    /// Cost estimate (alternative name)
    pub cost_estimate: Option<CostEstimate>,
}

/// Optimal system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfiguration {
    /// Configuration name
    pub name: String,
    
    /// CPU recommendation
    pub cpu_recommendation: String,
    
    /// GPU recommendation
    pub gpu_recommendation: Option<String>,
    
    /// Memory recommendation in GB
    pub memory_gb: f64,
    
    /// Storage recommendation in GB
    pub storage_gb: f64,
    
    /// Estimated cost
    pub estimated_cost: Option<CostEstimate>,
    
    /// Memory recommendation text
    pub memory_recommendation: String,
    
    /// Storage recommendation text
    pub storage_recommendation: String,
    
    /// Network recommendation text
    pub network_recommendation: String,
    
    /// Total cost
    pub total_cost: Option<CostEstimate>,
    
    /// Performance projection
    pub performance_projection: String,
}

/// Resource utilization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    
    /// Memory utilization percentage
    pub memory_percent: f64,
    
    /// Storage utilization percentage
    pub storage_percent: f64,
    
    /// Network utilization percentage
    pub network_percent: f64,
    
    /// GPU utilization percentage
    pub gpu_percent: f64,
    
    /// Peak utilization percentage
    pub peak_utilization: f64,
}

/// Performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target latency in milliseconds
    pub target_latency_ms: f64,
    
    /// Target throughput
    pub target_throughput: f64,
    
    /// Target utilization percentage
    pub target_utilization_percent: f64,
}

/// Acceleration benefit estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationBenefit {
    /// Speed improvement factor
    pub speed_improvement_factor: f64,
    
    /// Power efficiency improvement
    pub power_efficiency_improvement: f64,
    
    /// Cost per performance improvement
    pub cost_per_performance: f64,
    
    /// Description of benefits
    pub description: String,
    
    /// Confidence level (0.0 to 1.0)
    pub confidence_level: f64,
}

/// AI hardware upgrade recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIUpgradeRecommendations {
    /// Memory upgrade recommendation
    pub memory_upgrade: Option<MemoryUpgrade>,
    
    /// GPU upgrade recommendation
    pub gpu_upgrade: Option<GPUUpgrade>,
    
    /// Specialized accelerator recommendation
    pub accelerator_recommendation: Option<AcceleratorRecommendation>,
    
    /// Storage recommendation (for model storage)
    pub storage_recommendation: Option<StorageRecommendation>,
    
    /// Total estimated cost
    pub estimated_cost: Option<CostEstimate>,
    
    /// Estimated performance gain
    pub performance_gain: Option<PerformanceGainEstimate>,
    
    /// Upgrade priority
    pub priority: UpgradePriority,
}

/// Memory upgrade recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUpgrade {
    /// Current RAM in GB
    pub current_ram_gb: f64,
    
    /// Recommended RAM in GB
    pub recommended_ram_gb: f64,
    
    /// Upgrade description
    pub description: String,
    
    /// Estimated cost in USD
    pub estimated_cost_usd: f64,
}

/// GPU upgrade recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUUpgrade {
    /// Current GPU
    pub current_gpu: String,
    
    /// Recommended GPU
    pub recommended_gpu: String,
    
    /// Required VRAM in GB
    pub vram_required_gb: f64,
    
    /// Recommended VRAM in GB
    pub vram_recommended_gb: f64,
    
    /// Upgrade description
    pub description: String,
    
    /// Estimated cost in USD
    pub estimated_cost_usd: f64,
}

/// Specialized accelerator recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorRecommendation {
    /// Accelerator name
    pub accelerator_name: String,
    
    /// Accelerator type
    pub accelerator_type: String,
    
    /// TOPS performance
    pub tops_performance: f64,
    
    /// Recommendation description
    pub description: String,
    
    /// Estimated cost in USD
    pub estimated_cost_usd: f64,
}

/// Storage recommendation for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageRecommendation {
    /// Current storage
    pub current_storage: String,
    
    /// Recommended storage
    pub recommended_storage: String,
    
    /// Recommendation description
    pub description: String,
    
    /// Estimated cost in USD
    pub estimated_cost_usd: f64,
}

/// Performance gain estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceGainEstimate {
    /// Latency improvement percentage
    pub latency_improvement_percent: f64,
    
    /// Throughput improvement percentage
    pub throughput_improvement_percent: f64,
    
    /// Energy efficiency improvement percentage
    pub energy_efficiency_improvement_percent: f64,
    
    /// Description of performance improvements
    pub description: String,
}

/// Basic workload requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadRequirements {
    /// Name of the workload
    pub name: String,
    
    /// Required CPU cores
    pub cpu_cores: usize,
    
    /// Required memory in GB
    pub memory_gb: f64,
    
    /// Required storage in GB
    pub storage_gb: f64,
    
    /// Required network bandwidth in Mbps
    pub network_bandwidth_mbps: Option<f64>,
    
    /// CPU architecture requirements
    pub cpu_architecture: Option<String>,
    
    /// Operating system requirements
    pub operating_system: Option<String>,
    
    /// Description of the workload
    pub description: String,
    
    /// Associated workload type
    pub workload: Option<crate::workloads::WorkloadType>,
    
    /// Resource requirements list
    pub resource_requirements: Vec<crate::resources::ResourceRequirement>,
    
    /// Workload priority
    pub priority: WorkloadPriority,
}

impl WorkloadRequirements {
    /// Create a new workload requirements with basic defaults
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            cpu_cores: 1,
            memory_gb: 1.0,
            storage_gb: 10.0,
            network_bandwidth_mbps: None,
            cpu_architecture: None,
            operating_system: None,
            description: String::new(),
            workload: None,
            resource_requirements: Vec::new(),
            priority: WorkloadPriority::Medium,
        }
    }
    
    /// Add a resource requirement
    pub fn add_resource_requirement(&mut self, requirement: crate::resources::ResourceRequirement) {
        self.resource_requirements.push(requirement);
    }
    
    /// Get a resource requirement by type
    pub fn get_resource_requirement(&self, resource_type: &crate::resources::ResourceType) -> Option<&crate::resources::ResourceRequirement> {
        self.resource_requirements.iter().find(|req| req.resource_type == *resource_type)
    }
}

/// Builder for SystemProfile to handle many arguments
#[derive(Debug, Clone)]
pub struct SystemProfileBuilder {
    cpu_score: f64,
    gpu_score: f64,
    npu_score: f64,
    tpu_score: f64,
    fpga_score: f64,
    arm_optimization_score: f64,
    memory_score: f64,
    storage_score: f64,
    network_score: f64,
    system_info: Option<SystemInfo>,
}

impl SystemProfileBuilder {
    pub fn new() -> Self {
        Self {
            cpu_score: 0.0,
            gpu_score: 0.0,
            npu_score: 0.0,
            tpu_score: 0.0,
            fpga_score: 0.0,
            arm_optimization_score: 0.0,
            memory_score: 0.0,
            storage_score: 0.0,
            network_score: 0.0,
            system_info: None,
        }
    }

    pub fn cpu_score(mut self, score: f64) -> Self {
        self.cpu_score = score;
        self
    }

    pub fn gpu_score(mut self, score: f64) -> Self {
        self.gpu_score = score;
        self
    }

    pub fn npu_score(mut self, score: f64) -> Self {
        self.npu_score = score;
        self
    }

    pub fn tpu_score(mut self, score: f64) -> Self {
        self.tpu_score = score;
        self
    }

    pub fn fpga_score(mut self, score: f64) -> Self {
        self.fpga_score = score;
        self
    }

    pub fn arm_optimization_score(mut self, score: f64) -> Self {
        self.arm_optimization_score = score;
        self
    }

    pub fn memory_score(mut self, score: f64) -> Self {
        self.memory_score = score;
        self
    }

    pub fn storage_score(mut self, score: f64) -> Self {
        self.storage_score = score;
        self
    }

    pub fn network_score(mut self, score: f64) -> Self {
        self.network_score = score;
        self
    }

    pub fn system_info(mut self, info: SystemInfo) -> Self {
        self.system_info = Some(info);
        self
    }

    pub fn build(self) -> SystemProfile {
        let system_info = self.system_info.expect("SystemInfo is required");
        
        // Calculate AI accelerator score as the max of available accelerators
        let ai_accelerator_score = self.npu_score.max(self.tpu_score).max(self.fpga_score);
        
        // Calculate AI workload score emphasizing accelerators
        let ai_workload_score = ai_accelerator_score * 0.4 + self.gpu_score * 0.3 + self.cpu_score * 0.2 + self.memory_score * 0.1;
        
        // Calculate edge computing score emphasizing ARM optimization and efficiency
        let edge_computing_score = self.arm_optimization_score * 0.3 + ai_accelerator_score * 0.3 + self.cpu_score * 0.2 + self.memory_score * 0.2;
        
        // Enhanced overall score including AI capabilities
        let overall_score = (self.cpu_score + self.gpu_score + ai_accelerator_score + self.memory_score + self.storage_score + self.network_score) / 6.0;
        
        SystemProfile {
            cpu_score: self.cpu_score,
            gpu_score: self.gpu_score,
            npu_score: self.npu_score,
            tpu_score: self.tpu_score,
            fpga_score: self.fpga_score,
            ai_accelerator_score,
            arm_optimization_score: self.arm_optimization_score,
            memory_score: self.memory_score,
            storage_score: self.storage_score,
            network_score: self.network_score,
            overall_score,
            ai_workload_score,
            edge_computing_score,
            system_info,
            created_at: Utc::now(),
        }
    }
}

impl Default for SystemProfileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

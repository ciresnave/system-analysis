//! Core types for system analysis.

use crate::resources::{ResourceRequirement, ResourceType};
use crate::workloads::Workload;
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
    /// Memory performance score (0-10)
    pub memory_score: f64,
    /// Storage performance score (0-10)
    pub storage_score: f64,
    /// Network performance score (0-10)
    pub network_score: f64,
    /// Overall system score (0-10)
    pub overall_score: f64,
    /// Detailed system information
    pub system_info: SystemInfo,
    /// Timestamp when profile was created
    pub created_at: DateTime<Utc>,
}

impl SystemProfile {
    /// Create a new system profile
    pub fn new(
        cpu_score: f64,
        gpu_score: f64,
        memory_score: f64,
        storage_score: f64,
        network_score: f64,
        system_info: SystemInfo,
    ) -> Self {
        let overall_score = (cpu_score + gpu_score + memory_score + storage_score + network_score) / 5.0;
        
        Self {
            cpu_score,
            gpu_score,
            memory_score,
            storage_score,
            network_score,
            overall_score,
            system_info,
            created_at: Utc::now(),
        }
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

    /// Get AI capabilities assessment
    pub fn ai_capabilities(&self) -> AICapabilities {
        AICapabilities::from_profile(self)
    }
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
    /// AI inference capability level
    pub inference_level: String,
    /// AI training capability level
    pub training_level: String,
    /// Recommended model sizes
    pub recommended_models: Vec<String>,
    /// Hardware acceleration support
    pub hardware_acceleration: Vec<String>,
}

impl AICapabilities {
    /// Create AI capabilities from system profile
    pub fn from_profile(profile: &SystemProfile) -> Self {
        let inference_level = match profile.overall_score {
            score if score >= 8.0 => "High - Can run large models efficiently",
            score if score >= 6.0 => "Medium - Can run medium models",
            score if score >= 4.0 => "Low - Can run small models",
            _ => "Very Low - Limited AI capabilities",
        };

        let training_level = match profile.gpu_score {
            score if score >= 8.0 => "High - Can train large models",
            score if score >= 6.0 => "Medium - Can train medium models",
            score if score >= 4.0 => "Low - Can train small models",
            _ => "Very Low - Training not recommended",
        };

        let recommended_models = match profile.overall_score {
            score if score >= 8.0 => vec![
                "GPT-4 class models".to_string(),
                "Large language models (70B+)".to_string(),
                "Multimodal models".to_string(),
            ],
            score if score >= 6.0 => vec![
                "Medium language models (7B-13B)".to_string(),
                "Image generation models".to_string(),
                "Code generation models".to_string(),
            ],
            score if score >= 4.0 => vec![
                "Small language models (1B-3B)".to_string(),
                "Lightweight models".to_string(),
            ],
            _ => vec!["Quantized models only".to_string()],
        };

        let mut hardware_acceleration = Vec::new();
        if profile.gpu_score >= 6.0 {
            hardware_acceleration.push("GPU acceleration".to_string());
        }
        if profile.cpu_score >= 7.0 {
            hardware_acceleration.push("CPU optimization".to_string());
        }

        Self {
            inference_level: inference_level.to_string(),
            training_level: training_level.to_string(),
            recommended_models,
            hardware_acceleration,
        }
    }

    /// Get the inference level
    pub fn inference_level(&self) -> &str {
        &self.inference_level
    }

    /// Get the training level
    pub fn training_level(&self) -> &str {
        &self.training_level
    }
}

/// Workload requirements specification
#[derive(Debug)]
pub struct WorkloadRequirements {
    /// Workload name/identifier
    pub name: String,
    /// Resource requirements
    pub resource_requirements: Vec<ResourceRequirement>,
    /// Specific workload details
    pub workload: Option<Box<dyn Workload>>,
    /// Priority level
    pub priority: WorkloadPriority,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

impl Clone for WorkloadRequirements {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            resource_requirements: self.resource_requirements.clone(),
            workload: self.workload.as_ref().map(|w| w.clone_workload()),
            priority: self.priority,
            performance_targets: self.performance_targets.clone(),
            created_at: self.created_at,
        }
    }
}
impl WorkloadRequirements {
    /// Create new workload requirements
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            resource_requirements: Vec::new(),
            workload: None,
            priority: WorkloadPriority::Medium,
            performance_targets: PerformanceTargets::default(),
            created_at: Utc::now(),
        }
    }

    /// Add a resource requirement
    pub fn add_resource_requirement(&mut self, requirement: ResourceRequirement) {
        self.resource_requirements.push(requirement);
    }

    /// Set the workload
    pub fn set_workload(&mut self, workload: Box<dyn Workload>) {
        self.workload = Some(workload);
    }

    /// Set the priority
    pub fn set_priority(&mut self, priority: WorkloadPriority) {
        self.priority = priority;
    }

    /// Set performance targets
    pub fn set_performance_targets(&mut self, targets: PerformanceTargets) {
        self.performance_targets = targets;
    }

    /// Get resource requirement by type
    pub fn get_resource_requirement(&self, resource_type: &ResourceType) -> Option<&ResourceRequirement> {
        self.resource_requirements
            .iter()
            .find(|req| &req.resource_type == resource_type)
    }
}

/// Workload priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkloadPriority {
    /// Low priority workload
    Low,
    /// Medium priority workload
    Medium,
    /// High priority workload
    High,
    /// Critical priority workload
    Critical,
}

/// Performance targets for workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target latency in milliseconds
    pub target_latency_ms: Option<f64>,
    /// Target throughput (operations per second)
    pub target_throughput: Option<f64>,
    /// Maximum resource utilization percentage
    pub max_resource_utilization: Option<f64>,
    /// Energy efficiency requirements
    pub energy_efficiency: Option<f64>,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_latency_ms: None,
            target_throughput: None,
            max_resource_utilization: Some(80.0),
            energy_efficiency: None,
        }
    }
}

/// Result of compatibility analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// Whether the system is compatible
    pub is_compatible: bool,
    /// Overall compatibility score (0-10)
    pub score: f64,
    /// Performance estimate
    pub performance_estimate: PerformanceEstimate,
    /// Missing requirements
    pub missing_requirements: Vec<MissingRequirement>,
    /// Bottlenecks identified
    pub bottlenecks: Vec<Bottleneck>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl CompatibilityResult {
    /// Check if system is compatible
    pub fn is_compatible(&self) -> bool {
        self.is_compatible
    }

    /// Get compatibility score
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Get performance estimate
    pub fn performance_estimate(&self) -> &PerformanceEstimate {
        &self.performance_estimate
    }

    /// Get missing requirements
    pub fn missing_requirements(&self) -> &[MissingRequirement] {
        &self.missing_requirements
    }
}

/// Performance estimate for a workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEstimate {
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    /// Estimated throughput (operations per second)
    pub estimated_throughput: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Performance tier
    pub performance_tier: PerformanceTier,
}

impl std::fmt::Display for PerformanceEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (latency: {:.2}ms, throughput: {:.2} ops/s, confidence: {:.1}%)",
            self.performance_tier,
            self.estimated_latency_ms,
            self.estimated_throughput,
            self.confidence * 100.0
        )
    }
}

/// Performance tier classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceTier {
    /// Excellent performance
    Excellent,
    /// Good performance
    Good,
    /// Fair performance
    Fair,
    /// Poor performance
    Poor,
}

impl std::fmt::Display for PerformanceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerformanceTier::Excellent => write!(f, "Excellent"),
            PerformanceTier::Good => write!(f, "Good"),
            PerformanceTier::Fair => write!(f, "Fair"),
            PerformanceTier::Poor => write!(f, "Poor"),
        }
    }
}

/// Missing requirement information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingRequirement {
    /// Resource type that's missing
    pub resource_type: ResourceType,
    /// Required amount/level
    pub required: String,
    /// Currently available amount/level
    pub available: String,
    /// Severity of the missing requirement
    pub severity: RequirementSeverity,
}

impl MissingRequirement {
    /// Get the resource type
    pub fn resource_type(&self) -> &ResourceType {
        &self.resource_type
    }

    /// Get the required amount
    pub fn required(&self) -> &str {
        &self.required
    }

    /// Get the available amount
    pub fn available(&self) -> &str {
        &self.available
    }
}

/// Severity of missing requirements
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RequirementSeverity {
    /// Critical - workload cannot run
    Critical,
    /// High - significant performance impact
    High,
    /// Medium - moderate performance impact
    Medium,
    /// Low - minor performance impact
    Low,
}

/// System bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Resource type that's bottlenecked
    pub resource_type: ResourceType,
    /// Bottleneck description
    pub description: String,
    /// Impact level
    pub impact: BottleneckImpact,
    /// Suggestions to resolve
    pub suggestions: Vec<String>,
}

/// Impact level of bottlenecks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BottleneckImpact {
    /// Severe impact
    Severe,
    /// Moderate impact
    Moderate,
    /// Minor impact
    Minor,
}

/// Resource utilization prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    /// GPU utilization percentage
    pub gpu_percent: f64,
    /// Memory utilization percentage
    pub memory_percent: f64,
    /// Storage utilization percentage
    pub storage_percent: f64,
    /// Network utilization percentage
    pub network_percent: f64,
    /// Peak utilization values
    pub peak_utilization: HashMap<ResourceType, f64>,
}

impl ResourceUtilization {
    /// Get CPU utilization percentage
    pub fn cpu_percent(&self) -> f64 {
        self.cpu_percent
    }

    /// Get GPU utilization percentage
    pub fn gpu_percent(&self) -> f64 {
        self.gpu_percent
    }

    /// Get memory utilization percentage
    pub fn memory_percent(&self) -> f64 {
        self.memory_percent
    }

    /// Get storage utilization percentage
    pub fn storage_percent(&self) -> f64 {
        self.storage_percent
    }

    /// Get network utilization percentage
    pub fn network_percent(&self) -> f64 {
        self.network_percent
    }
}

/// Upgrade recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeRecommendation {
    /// Resource type to upgrade
    pub resource_type: ResourceType,
    /// Upgrade recommendation
    pub recommendation: String,
    /// Estimated improvement
    pub estimated_improvement: String,
    /// Cost estimate
    pub cost_estimate: Option<CostEstimate>,
    /// Priority level
    pub priority: UpgradePriority,
}

impl UpgradeRecommendation {
    /// Get the resource type
    pub fn resource_type(&self) -> &ResourceType {
        &self.resource_type
    }

    /// Get the recommendation
    pub fn recommendation(&self) -> &str {
        &self.recommendation
    }

    /// Get the estimated improvement
    pub fn estimated_improvement(&self) -> &str {
        &self.estimated_improvement
    }
}

/// Cost estimate for upgrades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Minimum cost
    pub min_cost: f64,
    /// Maximum cost
    pub max_cost: f64,
    /// Currency
    pub currency: String,
    /// Time frame
    pub time_frame: String,
}

/// Upgrade priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UpgradePriority {
    /// Low priority upgrade
    Low,
    /// Medium priority upgrade
    Medium,
    /// High priority upgrade
    High,
    /// Critical priority upgrade
    Critical,
}

/// Optimal configuration recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfiguration {
    /// CPU recommendation
    pub cpu_recommendation: String,
    /// GPU recommendation
    pub gpu_recommendation: String,
    /// Memory recommendation
    pub memory_recommendation: String,
    /// Storage recommendation
    pub storage_recommendation: String,
    /// Network recommendation
    pub network_recommendation: String,
    /// Total estimated cost
    pub total_cost: Option<CostEstimate>,
    /// Performance projection
    pub performance_projection: PerformanceEstimate,
}

impl OptimalConfiguration {
    /// Get CPU recommendation
    pub fn cpu_recommendation(&self) -> &str {
        &self.cpu_recommendation
    }

    /// Get GPU recommendation
    pub fn gpu_recommendation(&self) -> &str {
        &self.gpu_recommendation
    }

    /// Get memory recommendation
    pub fn memory_recommendation(&self) -> &str {
        &self.memory_recommendation
    }

    /// Get storage recommendation
    pub fn storage_recommendation(&self) -> &str {
        &self.storage_recommendation
    }

    /// Get network recommendation
    pub fn network_recommendation(&self) -> &str {
        &self.network_recommendation
    }
}

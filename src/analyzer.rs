//! Main system analyzer implementation.

use crate::capabilities::CapabilityProfile;
use crate::error::{Result, SystemAnalysisError};
use crate::resources::{ResourcePool, ResourceType, ResourceAmount, CapabilityLevel};
use crate::types::{
    SystemProfile, SystemInfo, CpuInfo, GpuInfo, MemoryInfo, StorageInfo, NetworkInfo,
    NetworkInterface, CompatibilityResult, PerformanceEstimate, PerformanceTier,
    ResourceUtilization, UpgradeRecommendation, OptimalConfiguration, WorkloadRequirements,
    MissingRequirement, RequirementSeverity, Bottleneck, BottleneckImpact, UpgradePriority,
    CostEstimate,
};
use crate::workloads::WorkloadType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sysinfo::{System, Disks, Networks};
use tracing::{info, debug};

/// Main system analyzer for capability assessment and workload compatibility
#[derive(Debug, Clone)]
pub struct SystemAnalyzer {
    /// Configuration options
    config: AnalyzerConfig,
    /// Cached system information
    cached_system_info: Option<SystemInfo>,
    /// Cached capability profile
    cached_capability_profile: Option<CapabilityProfile>,
    /// Resource pool for tracking available resources
    resource_pool: ResourcePool,
}

/// Configuration for the system analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable GPU detection
    pub enable_gpu_detection: bool,
    /// Enable detailed CPU analysis
    pub enable_detailed_cpu_analysis: bool,
    /// Enable network speed testing
    pub enable_network_testing: bool,
    /// Cache system information duration in seconds
    pub cache_duration_seconds: u64,
    /// Enable performance benchmarking
    pub enable_benchmarking: bool,
    /// Benchmark timeout in seconds
    pub benchmark_timeout_seconds: u64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_gpu_detection: true,
            enable_detailed_cpu_analysis: true,
            enable_network_testing: false, // Can be slow
            cache_duration_seconds: 300, // 5 minutes
            enable_benchmarking: false, // Can be slow
            benchmark_timeout_seconds: 30,
        }
    }
}

impl SystemAnalyzer {
    /// Create a new system analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(AnalyzerConfig::default())
    }

    /// Create a new system analyzer with custom configuration
    pub fn with_config(config: AnalyzerConfig) -> Self {
        Self {
            config,
            cached_system_info: None,
            cached_capability_profile: None,
            resource_pool: ResourcePool::new(),
        }
    }

    /// Analyze the current system and return a comprehensive profile
    pub async fn analyze_system(&mut self) -> Result<SystemProfile> {
        info!("Starting system analysis");

        let system_info = self.get_system_info().await?;
        let capability_profile = CapabilityProfile::from_system_info(&system_info);

        // Update resource pool with detected capabilities
        self.update_resource_pool(&capability_profile);

        // Cache the results
        self.cached_system_info = Some(system_info.clone());
        self.cached_capability_profile = Some(capability_profile.clone());

        let system_profile = SystemProfile::new(
            capability_profile.scores.cpu_score,
            capability_profile.scores.gpu_score,
            capability_profile.scores.memory_score,
            capability_profile.scores.storage_score,
            capability_profile.scores.network_score,
            system_info,
        );

        info!("System analysis completed with overall score: {:.1}", system_profile.overall_score());
        Ok(system_profile)
    }

    /// Check compatibility between system and workload requirements
    pub fn check_compatibility(
        &self,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<CompatibilityResult> {
        debug!("Checking compatibility for workload: {}", workload_requirements.name);

        // Check resource requirements
        let missing_requirements = self.find_missing_requirements(system_profile, workload_requirements)?;
        let is_compatible = missing_requirements.is_empty();

        // Calculate compatibility score
        let score = self.calculate_compatibility_score(system_profile, workload_requirements)?;

        // Estimate performance
        let performance_estimate = self.estimate_performance(system_profile, workload_requirements)?;

        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks(system_profile, workload_requirements)?;

        // Generate recommendations
        let recommendations = self.generate_compatibility_recommendations(
            system_profile,
            workload_requirements,
            &missing_requirements,
            &bottlenecks,
        );

        Ok(CompatibilityResult {
            is_compatible,
            score,
            performance_estimate,
            missing_requirements,
            bottlenecks,
            recommendations,
        })
    }

    /// Predict resource utilization for a workload
    pub fn predict_utilization(
        &self,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<ResourceUtilization> {
        debug!("Predicting resource utilization for workload: {}", workload_requirements.name);

        // Get estimated utilization from workload if available
        let base_utilization = if let Some(workload) = &workload_requirements.workload {
            workload.estimated_utilization()
        } else {
            // Use default estimates based on workload type and requirements
            self.estimate_default_utilization(workload_requirements)?
        };

        // Adjust based on system capabilities
        let cpu_percent = self.adjust_cpu_utilization(
            base_utilization.get(&ResourceType::CPU).copied().unwrap_or(50.0),
            system_profile,
        );

        let gpu_percent = self.adjust_gpu_utilization(
            base_utilization.get(&ResourceType::GPU).copied().unwrap_or(0.0),
            system_profile,
        );

        let memory_percent = self.adjust_memory_utilization(
            base_utilization.get(&ResourceType::Memory).copied().unwrap_or(40.0),
            system_profile,
            workload_requirements,
        );

        let storage_percent = base_utilization.get(&ResourceType::Storage).copied().unwrap_or(10.0);
        let network_percent = base_utilization.get(&ResourceType::Network).copied().unwrap_or(5.0);

        // Calculate peak utilization values
        let mut peak_utilization = HashMap::new();
        peak_utilization.insert(ResourceType::CPU, cpu_percent * 1.2);
        peak_utilization.insert(ResourceType::GPU, gpu_percent * 1.1);
        peak_utilization.insert(ResourceType::Memory, memory_percent * 1.05);
        peak_utilization.insert(ResourceType::Storage, storage_percent * 2.0);
        peak_utilization.insert(ResourceType::Network, network_percent * 3.0);

        Ok(ResourceUtilization {
            cpu_percent,
            gpu_percent,
            memory_percent,
            storage_percent,
            network_percent,
            peak_utilization,
        })
    }

    /// Recommend upgrades for better workload compatibility
    pub fn recommend_upgrades(
        &self,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<Vec<UpgradeRecommendation>> {
        debug!("Generating upgrade recommendations for workload: {}", workload_requirements.name);

        let mut recommendations = Vec::new();
        let missing_requirements = self.find_missing_requirements(system_profile, workload_requirements)?;

        for missing in &missing_requirements {
            let recommendation = self.generate_upgrade_recommendation(
                &missing.resource_type,
                &missing.required,
                &missing.available,
                system_profile,
            )?;
            recommendations.push(recommendation);
        }

        // Add general improvement recommendations
        recommendations.extend(self.generate_general_upgrade_recommendations(system_profile, workload_requirements)?);

        // Sort by priority
        recommendations.sort_by(|a, b| {
            use UpgradePriority::*;
            let priority_order = |p: &UpgradePriority| match p {
                Critical => 0,
                High => 1,
                Medium => 2,
                Low => 3,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });

        Ok(recommendations)
    }

    /// Find optimal hardware configuration for workload requirements
    pub fn find_optimal_configuration(
        &self,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<OptimalConfiguration> {
        debug!("Finding optimal configuration for workload: {}", workload_requirements.name);

        let cpu_recommendation = self.recommend_optimal_cpu(workload_requirements)?;
        let gpu_recommendation = self.recommend_optimal_gpu(workload_requirements)?;
        let memory_recommendation = self.recommend_optimal_memory(workload_requirements)?;
        let storage_recommendation = self.recommend_optimal_storage(workload_requirements)?;
        let network_recommendation = self.recommend_optimal_network(workload_requirements)?;

        // Calculate estimated cost (placeholder implementation)
        let total_cost = Some(CostEstimate {
            min_cost: 2000.0,
            max_cost: 8000.0,
            currency: "USD".to_string(),
            time_frame: "Current market prices".to_string(),
        });

        // Project performance with optimal configuration
        let performance_projection = PerformanceEstimate {
            estimated_latency_ms: 25.0,
            estimated_throughput: 50.0,
            confidence: 0.85,
            performance_tier: PerformanceTier::Excellent,
        };

        Ok(OptimalConfiguration {
            cpu_recommendation,
            gpu_recommendation,
            memory_recommendation,
            storage_recommendation,
            network_recommendation,
            total_cost,
            performance_projection,
        })
    }

    /// Get cached system information or fetch new information
    async fn get_system_info(&mut self) -> Result<SystemInfo> {
        // Check if cached info is still valid
        if let Some(cached) = &self.cached_system_info {
            // In a real implementation, we'd check the cache timestamp
            // For now, return cached if available
            return Ok(cached.clone());
        }

        // Fetch new system information
        let mut system = System::new_all();
        system.refresh_all();

        let cpu_info = self.get_cpu_info(&system)?;
        let gpu_info = self.get_gpu_info().await?;
        let memory_info = self.get_memory_info(&system)?;
        let storage_info = self.get_storage_info(&system)?;
        let network_info = self.get_network_info(&system).await?;

        let system_info = SystemInfo {
            os_name: System::name().unwrap_or_else(|| "Unknown".to_string()),
            os_version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
            cpu_info,
            gpu_info,
            memory_info,
            storage_info,
            network_info,
        };

        Ok(system_info)
    }

    /// Extract CPU information from system
    fn get_cpu_info(&self, system: &System) -> Result<CpuInfo> {
        let cpus = system.cpus();
        
        if cpus.is_empty() {
            return Err(SystemAnalysisError::system_info("No CPU information available"));
        }

        let cpu = &cpus[0];
        let physical_cores = system.physical_core_count().unwrap_or(1);
        let logical_cores = cpus.len();

        Ok(CpuInfo {
            brand: cpu.brand().to_string(),
            physical_cores,
            logical_cores,
            base_frequency: (cpu.frequency() as u64).max(1000), // MHz
            max_frequency: None, // Would need additional system calls
            cache_size: None, // Would need additional system calls
            architecture: std::env::consts::ARCH.to_string(),
        })
    }

    /// Extract GPU information
    async fn get_gpu_info(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        if !self.config.enable_gpu_detection {
            return Ok(gpus);
        }

        #[cfg(feature = "gpu-detection")]
        {
            // Use NVML for NVIDIA GPU detection
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(device_count) = nvml.device_count() {
                    for i in 0..device_count {
                        if let Ok(device) = nvml.device_by_index(i) {
                            if let (Ok(name), Ok(memory_info)) = (device.name(), device.memory_info()) {
                                gpus.push(GpuInfo {
                                    name,
                                    vendor: "NVIDIA".to_string(),
                                    vram_size: Some(memory_info.total / 1024 / 1024), // Convert to MB
                                    compute_capability: device.cuda_compute_capability()
                                        .map(|cc| format!("{}.{}", cc.major, cc.minor))
                                        .ok(),
                                    opencl_support: true, // Assume true for modern NVIDIA cards
                                    cuda_support: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        // If no GPUs detected via NVML, add a placeholder based on common patterns
        if gpus.is_empty() {
            // This is a simplified detection - in a real implementation,
            // we'd use platform-specific APIs (DirectX, Metal, Vulkan, etc.)
            gpus.push(GpuInfo {
                name: "Integrated Graphics".to_string(),
                vendor: "Unknown".to_string(),
                vram_size: None,
                compute_capability: None,
                opencl_support: false,
                cuda_support: false,
            });
        }

        Ok(gpus)
    }

    /// Extract memory information
    fn get_memory_info(&self, system: &System) -> Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_ram: system.total_memory() / 1024, // Convert to MB
            available_ram: system.available_memory() / 1024, // Convert to MB
            memory_type: None, // Would need additional system calls
            memory_speed: None, // Would need additional system calls
        })
    }

    /// Extract storage information
    fn get_storage_info(&self, _system: &System) -> Result<Vec<StorageInfo>> {
        let mut storage_devices = Vec::new();
        let disks = Disks::new_with_refreshed_list();

        for disk in &disks {
            let total_capacity = disk.total_space() / 1024 / 1024 / 1024; // Convert to GB
            let available_capacity = disk.available_space() / 1024 / 1024 / 1024; // Convert to GB

            // Determine storage type based on name/path patterns
            let storage_type = match disk.name().to_str().unwrap_or("") {
                name if name.contains("nvme") => "NVMe SSD".to_string(),
                name if name.contains("ssd") => "SSD".to_string(),
                _ => {
                    // Use filesystem type as a hint
                    match disk.file_system().to_str().unwrap_or("") {
                        "NTFS" | "APFS" | "ext4" => "SSD".to_string(), // Common on SSDs
                        _ => "HDD".to_string(), // Default assumption
                    }
                }
            };

            storage_devices.push(StorageInfo {
                name: disk.name().to_str().unwrap_or("Unknown").to_string(),
                storage_type,
                total_capacity,
                available_capacity,
                read_speed: None, // Would need benchmarking
                write_speed: None, // Would need benchmarking
            });
        }

        if storage_devices.is_empty() {
            // Add a default entry if no disks detected
            storage_devices.push(StorageInfo {
                name: "Primary Storage".to_string(),
                storage_type: "Unknown".to_string(),
                total_capacity: 100, // Default 100GB
                available_capacity: 50,
                read_speed: None,
                write_speed: None,
            });
        }

        Ok(storage_devices)
    }

    /// Extract network information
    async fn get_network_info(&self, _system: &System) -> Result<NetworkInfo> {
        let mut interfaces = Vec::new();
        let networks = Networks::new_with_refreshed_list();

        for (interface_name, network_data) in &networks {
            interfaces.push(NetworkInterface {
                name: interface_name.clone(),
                interface_type: if interface_name.to_lowercase().contains("ethernet") {
                    "Ethernet".to_string()
                } else if interface_name.to_lowercase().contains("wifi") ||
                         interface_name.to_lowercase().contains("wireless") {
                    "WiFi".to_string()
                } else {
                    "Unknown".to_string()
                },
                mac_address: network_data.mac_address().to_string(),
                ip_addresses: vec![], // Would need additional system calls
                speed: None, // Would need additional system calls
            });
        }

        Ok(NetworkInfo {
            interfaces,
            internet_connected: true, // Assume connected for now
            estimated_bandwidth: None, // Would need network testing
        })
    }

    /// Update resource pool with detected capabilities
    fn update_resource_pool(&mut self, capability_profile: &CapabilityProfile) {
        self.resource_pool.set_resource(
            ResourceType::CPU,
            ResourceAmount::Score(capability_profile.scores.cpu_score),
        );

        self.resource_pool.set_resource(
            ResourceType::GPU,
            ResourceAmount::Score(capability_profile.scores.gpu_score),
        );

        self.resource_pool.set_resource(
            ResourceType::Memory,
            ResourceAmount::Gigabytes(capability_profile.memory_capabilities.total_ram_gb),
        );

        self.resource_pool.set_resource(
            ResourceType::Storage,
            ResourceAmount::Gigabytes(capability_profile.storage_capabilities.total_capacity_gb),
        );

        self.resource_pool.set_resource(
            ResourceType::Network,
            ResourceAmount::Score(capability_profile.scores.network_score),
        );
    }

    /// Find missing requirements for a workload
    fn find_missing_requirements(
        &self,
        _system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<Vec<MissingRequirement>> {
        let mut missing = Vec::new();

        for req in &workload_requirements.resource_requirements {
            if let Some(available) = self.resource_pool.get_resource(&req.resource_type) {
                if !req.is_satisfied_by(available) {
                    missing.push(MissingRequirement {
                        resource_type: req.resource_type,
                        required: req.minimum.to_string(),
                        available: available.to_string(),
                        severity: if req.is_critical {
                            RequirementSeverity::Critical
                        } else {
                            RequirementSeverity::High
                        },
                    });
                }
            } else {
                missing.push(MissingRequirement {
                    resource_type: req.resource_type,
                    required: req.minimum.to_string(),
                    available: "Not Available".to_string(),
                    severity: RequirementSeverity::Critical,
                });
            }
        }

        Ok(missing)
    }

    /// Calculate compatibility score between system and workload
    fn calculate_compatibility_score(
        &self,
        _system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<f64> {
        let satisfaction_score = self.resource_pool.satisfaction_score(&workload_requirements.resource_requirements);
        Ok(satisfaction_score)
    }

    /// Estimate performance for a workload on the system
    fn estimate_performance(
        &self,
        system_profile: &SystemProfile,
        _workload_requirements: &WorkloadRequirements,
    ) -> Result<PerformanceEstimate> {
        // Base estimates - would be more sophisticated in real implementation
        let base_latency = 100.0; // ms
        let base_throughput = 10.0; // ops/sec

        // Adjust based on system scores
        let score_multiplier = system_profile.overall_score() / 10.0;
        let estimated_latency_ms = base_latency / score_multiplier.max(0.1);
        let estimated_throughput = base_throughput * score_multiplier;

        let confidence = if system_profile.overall_score() >= 7.0 {
            0.9
        } else if system_profile.overall_score() >= 5.0 {
            0.7
        } else {
            0.5
        };

        let performance_tier = match system_profile.overall_score() {
            score if score >= 8.0 => PerformanceTier::Excellent,
            score if score >= 6.0 => PerformanceTier::Good,
            score if score >= 4.0 => PerformanceTier::Fair,
            _ => PerformanceTier::Poor,
        };

        Ok(PerformanceEstimate {
            estimated_latency_ms,
            estimated_throughput,
            confidence,
            performance_tier,
        })
    }

    /// Identify system bottlenecks for a workload
    fn identify_bottlenecks(
        &self,
        system_profile: &SystemProfile,
        _workload_requirements: &WorkloadRequirements,
    ) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();

        // Check each resource type for bottlenecks
        let scores = [
            (ResourceType::CPU, system_profile.cpu_score()),
            (ResourceType::GPU, system_profile.gpu_score()),
            (ResourceType::Memory, system_profile.memory_score()),
            (ResourceType::Storage, system_profile.storage_score()),
            (ResourceType::Network, system_profile.network_score()),
        ];

        let avg_score = scores.iter().map(|(_, score)| score).sum::<f64>() / scores.len() as f64;

        for (resource_type, score) in scores {
            if score < avg_score - 2.0 { // Significantly below average
                let impact = if score < 3.0 {
                    BottleneckImpact::Severe
                } else if score < 5.0 {
                    BottleneckImpact::Moderate
                } else {
                    BottleneckImpact::Minor
                };

                let suggestions = self.generate_bottleneck_suggestions(&resource_type);

                bottlenecks.push(Bottleneck {
                    resource_type,
                    description: format!("{} performance is below system average ({:.1} vs {:.1})",
                        resource_type, score, avg_score),
                    impact,
                    suggestions,
                });
            }
        }

        Ok(bottlenecks)
    }

    /// Generate suggestions for resolving bottlenecks
    fn generate_bottleneck_suggestions(&self, resource_type: &ResourceType) -> Vec<String> {
        match resource_type {
            ResourceType::CPU => vec![
                "Upgrade to a CPU with more cores or higher clock speed".to_string(),
                "Consider CPUs with newer architecture (e.g., latest Intel or AMD)".to_string(),
                "Ensure adequate cooling for sustained performance".to_string(),
            ],
            ResourceType::GPU => vec![
                "Add a dedicated GPU for compute workloads".to_string(),
                "Upgrade to a GPU with more VRAM".to_string(),
                "Consider GPUs optimized for AI/ML workloads".to_string(),
            ],
            ResourceType::Memory => vec![
                "Increase RAM capacity".to_string(),
                "Upgrade to faster memory (higher frequency)".to_string(),
                "Consider ECC memory for reliability".to_string(),
            ],
            ResourceType::Storage => vec![
                "Upgrade to NVMe SSD for faster I/O".to_string(),
                "Add more storage capacity".to_string(),
                "Consider RAID configuration for performance".to_string(),
            ],
            ResourceType::Network => vec![
                "Upgrade to gigabit Ethernet".to_string(),
                "Improve WiFi signal strength".to_string(),
                "Consider wired connection for consistency".to_string(),
            ],
            ResourceType::Custom(_) => vec![
                "Review custom resource requirements".to_string(),
            ],
        }
    }

    /// Generate compatibility recommendations
    fn generate_compatibility_recommendations(
        &self,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
        missing_requirements: &[MissingRequirement],
        bottlenecks: &[Bottleneck],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if missing_requirements.is_empty() && bottlenecks.is_empty() {
            recommendations.push("System meets all requirements for optimal performance".to_string());
        } else {
            if !missing_requirements.is_empty() {
                recommendations.push(format!("Address {} missing requirements", missing_requirements.len()));
            }
            
            if !bottlenecks.is_empty() {
                recommendations.push(format!("Resolve {} system bottlenecks", bottlenecks.len()));
            }

            // Add specific recommendations based on workload type
            if let Some(workload) = &workload_requirements.workload {
                match workload.workload_type() {
                    WorkloadType::AIInference => {
                        if system_profile.gpu_score() < 6.0 {
                            recommendations.push("Consider GPU acceleration for AI inference".to_string());
                        }
                    }
                    WorkloadType::MemoryIntensive => {
                        if system_profile.memory_score() < 7.0 {
                            recommendations.push("Increase memory capacity for memory-intensive workloads".to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        recommendations
    }

    // Helper methods for resource utilization adjustment
    fn adjust_cpu_utilization(&self, base_util: f64, system_profile: &SystemProfile) -> f64 {
        // Higher CPU scores can handle workloads more efficiently
        let efficiency_factor = (system_profile.cpu_score() / 10.0).max(0.1);
        (base_util / efficiency_factor).min(100.0)
    }

    fn adjust_gpu_utilization(&self, base_util: f64, system_profile: &SystemProfile) -> f64 {
        if system_profile.gpu_score() < 3.0 {
            0.0 // No meaningful GPU utilization
        } else {
            let efficiency_factor = (system_profile.gpu_score() / 10.0).max(0.1);
            (base_util / efficiency_factor).min(100.0)
        }
    }

    fn adjust_memory_utilization(
        &self,
        _base_util: f64,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> f64 {
        // Calculate actual memory usage based on requirements
        let memory_req = workload_requirements.resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::Memory)
            .and_then(|req| match &req.minimum {
                ResourceAmount::Gigabytes(gb) => Some(*gb),
                _ => None,
            })
            .unwrap_or(4.0); // Default 4GB

        let total_memory = system_profile.system_info.memory_info.total_ram as f64 / 1024.0; // Convert to GB
        ((memory_req / total_memory) * 100.0).min(100.0)
    }

    // Helper methods for default utilization estimation
    fn estimate_default_utilization(&self, workload_requirements: &WorkloadRequirements) -> Result<HashMap<ResourceType, f64>> {
        let mut utilization = HashMap::new();

        // Basic estimates based on workload priority and resource requirements
        let base_cpu = match workload_requirements.priority {
            crate::types::WorkloadPriority::Critical => 80.0,
            crate::types::WorkloadPriority::High => 60.0,
            crate::types::WorkloadPriority::Medium => 40.0,
            crate::types::WorkloadPriority::Low => 20.0,
        };

        utilization.insert(ResourceType::CPU, base_cpu);
        utilization.insert(ResourceType::GPU, 0.0); // Default no GPU
        utilization.insert(ResourceType::Memory, 30.0);
        utilization.insert(ResourceType::Storage, 10.0);
        utilization.insert(ResourceType::Network, 5.0);

        Ok(utilization)
    }

    // Helper methods for upgrade recommendations
    fn generate_upgrade_recommendation(
        &self,
        resource_type: &ResourceType,
        required: &str,
        available: &str,
        _system_profile: &SystemProfile,
    ) -> Result<UpgradeRecommendation> {
        let (recommendation, estimated_improvement, priority) = match resource_type {
            ResourceType::CPU => {
                ("Upgrade to a higher-performance CPU with more cores".to_string(),
                 "30-50% performance improvement".to_string(),
                 UpgradePriority::High)
            }
            ResourceType::GPU => {
                ("Add or upgrade GPU for compute acceleration".to_string(),
                 "2-10x performance improvement for GPU workloads".to_string(),
                 UpgradePriority::Critical)
            }
            ResourceType::Memory => {
                (format!("Increase RAM from {} to {}", available, required),
                 "Eliminate memory bottlenecks".to_string(),
                 UpgradePriority::High)
            }
            ResourceType::Storage => {
                ("Upgrade to faster NVMe SSD storage".to_string(),
                 "Reduce I/O latency by 50-90%".to_string(),
                 UpgradePriority::Medium)
            }
            ResourceType::Network => {
                ("Upgrade network connection speed".to_string(),
                 "Reduce network latency and increase throughput".to_string(),
                 UpgradePriority::Low)
            }
            ResourceType::Custom(_) => {
                ("Review custom resource requirements".to_string(),
                 "Variable improvement".to_string(),
                 UpgradePriority::Medium)
            }
        };

        Ok(UpgradeRecommendation {
            resource_type: *resource_type,
            recommendation,
            estimated_improvement,
            cost_estimate: None, // Would be populated in real implementation
            priority,
        })
    }

    fn generate_general_upgrade_recommendations(
        &self,
        system_profile: &SystemProfile,
        _workload_requirements: &WorkloadRequirements,
    ) -> Result<Vec<UpgradeRecommendation>> {
        let mut recommendations = Vec::new();

        // Check for general improvement opportunities
        if system_profile.overall_score() < 6.0 {
            recommendations.push(UpgradeRecommendation {
                resource_type: ResourceType::CPU,
                recommendation: "Consider a comprehensive system upgrade".to_string(),
                estimated_improvement: "Significant overall performance improvement".to_string(),
                cost_estimate: Some(CostEstimate {
                    min_cost: 1500.0,
                    max_cost: 5000.0,
                    currency: "USD".to_string(),
                    time_frame: "Complete system refresh".to_string(),
                }),
                priority: UpgradePriority::Medium,
            });
        }

        Ok(recommendations)
    }

    // Helper methods for optimal configuration recommendations
    fn recommend_optimal_cpu(&self, workload_requirements: &WorkloadRequirements) -> Result<String> {
        // Analyze workload requirements and recommend optimal CPU
        let cpu_req = workload_requirements.resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::CPU);

        let recommendation = match cpu_req {
            Some(req) => match &req.minimum {
                ResourceAmount::Level(level) => match level {
                    CapabilityLevel::Exceptional => "High-end workstation CPU (e.g., Intel Xeon W or AMD Threadripper PRO)",
                    CapabilityLevel::VeryHigh => "High-performance CPU (e.g., Intel Core i9 or AMD Ryzen 9)",
                    CapabilityLevel::High => "Performance CPU (e.g., Intel Core i7 or AMD Ryzen 7)",
                    CapabilityLevel::Medium => "Mid-range CPU (e.g., Intel Core i5 or AMD Ryzen 5)",
                    _ => "Entry-level CPU (e.g., Intel Core i3 or AMD Ryzen 3)",
                },
                _ => "Modern multi-core CPU with good single-thread performance",
            },
            None => "Balanced CPU suitable for general workloads",
        };

        Ok(recommendation.to_string())
    }

    fn recommend_optimal_gpu(&self, workload_requirements: &WorkloadRequirements) -> Result<String> {
        let gpu_req = workload_requirements.resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::GPU);

        let recommendation = match gpu_req {
            Some(_) => {
                // Check if this is an AI workload
                if let Some(workload) = &workload_requirements.workload {
                    match workload.workload_type() {
                        WorkloadType::AIInference | WorkloadType::AITraining => {
                            "High-memory GPU optimized for AI (e.g., NVIDIA RTX 4090, A6000, or H100)"
                        }
                        _ => "Dedicated GPU with good compute performance"
                    }
                } else {
                    "Modern dedicated GPU with adequate VRAM"
                }
            }
            None => "Integrated graphics sufficient, dedicated GPU optional",
        };

        Ok(recommendation.to_string())
    }

    fn recommend_optimal_memory(&self, workload_requirements: &WorkloadRequirements) -> Result<String> {
        let memory_req = workload_requirements.resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::Memory)
            .and_then(|req| match &req.minimum {
                ResourceAmount::Gigabytes(gb) => Some(*gb),
                _ => None,
            })
            .unwrap_or(16.0);

        let recommendation = match memory_req {
            gb if gb >= 128.0 => format!("{}GB+ high-speed DDR5 RAM with ECC support", gb as u32),
            gb if gb >= 64.0 => format!("{}GB+ high-speed DDR5 RAM", gb as u32),
            gb if gb >= 32.0 => format!("{}GB DDR4/DDR5 RAM", gb as u32),
            gb if gb >= 16.0 => format!("{}GB DDR4 RAM", gb as u32),
            _ => "16GB DDR4 RAM (minimum recommended)".to_string(),
        };

        Ok(recommendation)
    }

    fn recommend_optimal_storage(&self, workload_requirements: &WorkloadRequirements) -> Result<String> {
        let storage_req = workload_requirements.resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::Storage)
            .and_then(|req| match &req.minimum {
                ResourceAmount::Gigabytes(gb) => Some(*gb),
                _ => None,
            })
            .unwrap_or(500.0);

        let recommendation = if storage_req >= 2000.0 {
            format!("{}GB+ NVMe SSD with high sequential read/write speeds", storage_req as u32)
        } else if storage_req >= 1000.0 {
            format!("{}GB NVMe SSD", storage_req as u32)
        } else {
            format!("{}GB+ SATA SSD or NVMe SSD", storage_req as u32)
        };

        Ok(recommendation)
    }

    fn recommend_optimal_network(&self, _workload_requirements: &WorkloadRequirements) -> Result<String> {
        Ok("Gigabit Ethernet connection (wired preferred for consistency)".to_string())
    }
}

impl Default for SystemAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResourceRequirement;

    #[test]
    fn test_analyzer_config_default() {
        let config = AnalyzerConfig::default();
        assert!(config.enable_gpu_detection);
        assert!(config.enable_detailed_cpu_analysis);
        assert!(!config.enable_network_testing);
        assert_eq!(config.cache_duration_seconds, 300);
        assert!(!config.enable_benchmarking);
        assert_eq!(config.benchmark_timeout_seconds, 30);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = SystemAnalyzer::new();
        assert!(analyzer.cached_system_info.is_none());
        assert!(analyzer.cached_capability_profile.is_none());
    }

    #[test]
    fn test_analyzer_with_config() {
        let config = AnalyzerConfig {
            enable_gpu_detection: false,
            enable_detailed_cpu_analysis: false,
            enable_network_testing: true,
            cache_duration_seconds: 600,
            enable_benchmarking: false,
            benchmark_timeout_seconds: 60,
        };
        
        let analyzer = SystemAnalyzer::with_config(config.clone());
        assert_eq!(analyzer.config.enable_gpu_detection, false);
        assert_eq!(analyzer.config.enable_network_testing, true);
        assert_eq!(analyzer.config.cache_duration_seconds, 600);
    }

    #[tokio::test]
    async fn test_system_analysis_basic() {
        let mut analyzer = SystemAnalyzer::new();
        let result = analyzer.analyze_system().await;
        assert!(result.is_ok());
        
        let profile = result.unwrap();
        // Verify score ranges
        assert!(profile.overall_score() >= 0.0 && profile.overall_score() <= 10.0);
        assert!(profile.cpu_score() >= 0.0 && profile.cpu_score() <= 10.0);
        assert!(profile.gpu_score() >= 0.0 && profile.gpu_score() <= 10.0);
        assert!(profile.memory_score() >= 0.0 && profile.memory_score() <= 10.0);
        assert!(profile.storage_score() >= 0.0 && profile.storage_score() <= 10.0);
        assert!(profile.network_score() >= 0.0 && profile.network_score() <= 10.0);
    }

    #[tokio::test]
    async fn test_workload_compatibility_simple() {
        let mut analyzer = SystemAnalyzer::new();
        let system_profile = analyzer.analyze_system().await.unwrap();
        
        let mut workload_requirements = WorkloadRequirements::new("test-workload");
        workload_requirements.add_resource_requirement(
            ResourceRequirement::new(ResourceType::Memory)
                .minimum_gb(4.0)
                .recommended_gb(8.0)
        );
        
        let compatibility = analyzer.check_compatibility(&system_profile, &workload_requirements);
        assert!(compatibility.is_ok());
        
        let result = compatibility.unwrap();
        assert!(result.is_compatible());
        assert!(result.score >= 0.0 && result.score <= 10.0);
    }

    #[test]
    fn test_workload_requirements_builder() {
        let mut requirements = WorkloadRequirements::new("test-workload");
        
        requirements.add_resource_requirement(
            ResourceRequirement::new(ResourceType::CPU)
                .minimum_level(CapabilityLevel::Medium)
                .recommended_level(CapabilityLevel::High)
        );
        
        requirements.add_resource_requirement(
            ResourceRequirement::new(ResourceType::Memory)
                .minimum_gb(8.0)
                .recommended_gb(16.0)
        );
        
        assert_eq!(requirements.name, "test-workload");
        assert_eq!(requirements.resource_requirements.len(), 2);
        
        // Test getting specific requirements
        let cpu_req = requirements.get_resource_requirement(&ResourceType::CPU);
        assert!(cpu_req.is_some());
        assert_eq!(cpu_req.unwrap().resource_type, ResourceType::CPU);
        
        let memory_req = requirements.get_resource_requirement(&ResourceType::Memory);
        assert!(memory_req.is_some());
        assert_eq!(memory_req.unwrap().resource_type, ResourceType::Memory);
        
        // Test getting non-existent requirement
        let gpu_req = requirements.get_resource_requirement(&ResourceType::GPU);
        assert!(gpu_req.is_none());
    }
}
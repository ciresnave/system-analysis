//! Main system analyzer implementation.

use crate::capabilities::CapabilityProfile;
use crate::error::{Result, SystemAnalysisError};
use crate::resources::{CapabilityLevel, ResourceAmount, ResourcePool, ResourceType};
use crate::types::{
    Bottleneck, BottleneckImpact, CompatibilityResult, CostEstimate, CpuInfo, GpuInfo, MemoryInfo,
    MissingRequirement, NetworkInfo, NetworkInterface, OptimalConfiguration, PerformanceEstimate,
    PerformanceTier, RequirementSeverity, ResourceUtilization, StorageInfo, SystemInfo,
    SystemProfile, UpgradePriority, UpgradeRecommendation, WorkloadRequirements,
};
use crate::workloads::WorkloadType;
use hardware_query::HardwareInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sysinfo::{Disks, Networks, System};
use tracing::{debug, info};

/// Main system analyzer for capability assessment and workload compatibility
#[derive(Debug, Clone)]
pub struct SystemAnalyzer {
    /// Configuration options
    #[allow(dead_code)]
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
            cache_duration_seconds: 300,   // 5 minutes
            enable_benchmarking: false,    // Can be slow
            benchmark_timeout_seconds: 30,
        }
    }
}

impl SystemAnalyzer {
    /// Returns a concise summary of key system specs for quick display.
    pub async fn quick_summary(&mut self) -> Result<String> {
        let info = self.get_system_info().await?;
        let cpu = &info.cpu_info;
        let gpus = &info.gpu_info;
        let ram_gb = info.memory_info.total_ram as f64 / 1024.0;
        let os = &info.os_name;
        let mut summary = format!(
            "OS: {} {}\nCPU: {} ({}C/{}T, {} MHz)\nRAM: {:.1} GB\n",
            os,
            info.os_version,
            cpu.brand,
            cpu.physical_cores,
            cpu.logical_cores,
            cpu.base_frequency,
            ram_gb
        );
        if !gpus.is_empty() {
            for (i, gpu) in gpus.iter().enumerate() {
                let vram = gpu
                    .vram_size
                    .map(|v| format!("{} MB", v))
                    .unwrap_or_else(|| "N/A".into());
                summary.push_str(&format!(
                    "GPU {}: {} [{}] VRAM: {}\n",
                    i + 1,
                    gpu.name,
                    gpu.vendor,
                    vram
                ));
            }
        }
        Ok(summary)
    }

    /// Returns detailed NVIDIA GPU info using nvml-wrapper (if enabled via "nvidia" feature).
    #[cfg(feature = "gpu-vendor-nvidia")]
    pub async fn nvidia_gpu_details(&self) -> Result<Vec<GpuInfo>> {
        use nvml_wrapper::Nvml;
        let nvml = Nvml::init()
            .map_err(|e| SystemAnalysisError::system_info(format!("NVML init failed: {e}")))?;
        let count = nvml
            .device_count()
            .map_err(|e| SystemAnalysisError::system_info(format!("NVML count failed: {e}")))?;
        let mut gpus = Vec::new();
        for i in 0..count {
            let device = nvml.device_by_index(i).map_err(|e| {
                SystemAnalysisError::system_info(format!("NVML device failed: {e}"))
            })?;
            let name = device.name().unwrap_or_else(|_| "Unknown".into());
            let mem = device.memory_info().ok();
            gpus.push(GpuInfo {
                name,
                vendor: "NVIDIA".into(),
                vram_size: mem.map(|m| m.total / 1024 / 1024),
                compute_capability: device
                    .cuda_compute_capability()
                    .ok()
                    .map(|cc| format!("{}.{}", cc.major, cc.minor)),
                opencl_support: true,
                cuda_support: true,
            });
        }
        Ok(gpus)
    }
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

        let system_profile = SystemProfile::builder()
            .cpu_score(capability_profile.scores.cpu_score)
            .gpu_score(capability_profile.scores.gpu_score)
            .npu_score(capability_profile.scores.npu_score.unwrap_or(0.0))
            .tpu_score(capability_profile.scores.tpu_score.unwrap_or(0.0))
            .fpga_score(capability_profile.scores.fpga_score.unwrap_or(0.0))
            .arm_optimization_score(
                capability_profile
                    .scores
                    .arm_optimization_score
                    .unwrap_or(0.0),
            )
            .memory_score(capability_profile.scores.memory_score)
            .storage_score(capability_profile.scores.storage_score)
            .network_score(capability_profile.scores.network_score)
            .system_info(system_info)
            .build();

        info!(
            "System analysis completed with overall score: {:.1}",
            system_profile.overall_score()
        );
        Ok(system_profile)
    }

    /// Check compatibility between system and workload requirements
    pub fn check_compatibility(
        &self,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<CompatibilityResult> {
        debug!(
            "Checking compatibility for workload: {}",
            workload_requirements.name
        );

        // Check resource requirements
        let missing_requirements =
            self.find_missing_requirements(system_profile, workload_requirements)?;
        let is_compatible = missing_requirements.is_empty();

        // Calculate compatibility score
        let score = self.calculate_compatibility_score(system_profile, workload_requirements)?;

        // Estimate performance
        let performance_estimate =
            self.estimate_performance(system_profile, workload_requirements)?;

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
        debug!(
            "Predicting resource utilization for workload: {}",
            workload_requirements.name
        );

        // Get estimated utilization from workload if available
        let base_utilization = if let Some(workload) = &workload_requirements.workload {
            let util = workload.estimated_utilization();
            let mut utilization_map = HashMap::new();
            utilization_map.insert(ResourceType::CPU, util * 100.0);
            utilization_map.insert(ResourceType::GPU, util * 80.0);
            utilization_map.insert(ResourceType::Memory, util * 60.0);
            utilization_map
        } else {
            // Use default estimates based on workload type and requirements
            self.estimate_default_utilization(workload_requirements)?
        };

        // Adjust based on system capabilities
        let cpu_percent = self.adjust_cpu_utilization(
            base_utilization
                .get(&ResourceType::CPU)
                .copied()
                .unwrap_or(50.0),
            system_profile,
        );

        let gpu_percent = self.adjust_gpu_utilization(
            base_utilization
                .get(&ResourceType::GPU)
                .copied()
                .unwrap_or(0.0),
            system_profile,
        );

        let memory_percent = self.adjust_memory_utilization(
            base_utilization
                .get(&ResourceType::Memory)
                .copied()
                .unwrap_or(40.0),
            system_profile,
            workload_requirements,
        );

        let storage_percent = base_utilization
            .get(&ResourceType::Storage)
            .copied()
            .unwrap_or(10.0);
        let network_percent = base_utilization
            .get(&ResourceType::Network)
            .copied()
            .unwrap_or(5.0);

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
            peak_utilization: peak_utilization.values().cloned().fold(0.0, f64::max),
        })
    }

    /// Recommend upgrades for better workload compatibility
    pub fn recommend_upgrades(
        &self,
        system_profile: &SystemProfile,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<Vec<UpgradeRecommendation>> {
        debug!(
            "Generating upgrade recommendations for workload: {}",
            workload_requirements.name
        );

        let mut recommendations = Vec::new();
        let missing_requirements =
            self.find_missing_requirements(system_profile, workload_requirements)?;

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
        recommendations.extend(
            self.generate_general_upgrade_recommendations(system_profile, workload_requirements)?,
        );

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
        debug!(
            "Finding optimal configuration for workload: {}",
            workload_requirements.name
        );

        let cpu_recommendation = self.recommend_optimal_cpu(workload_requirements)?;
        let gpu_recommendation = self.recommend_optimal_gpu(workload_requirements)?;
        let memory_recommendation = self.recommend_optimal_memory(workload_requirements)?;
        let storage_recommendation = self.recommend_optimal_storage(workload_requirements)?;
        let network_recommendation = self.recommend_optimal_network(workload_requirements)?;

        // Calculate estimated cost (placeholder implementation)
        let total_cost = Some(CostEstimate {
            min_cost_usd: 2000.0,
            max_cost_usd: 8000.0,
            currency: "USD".to_string(),
            breakdown: Vec::new(),
        });

        // Project performance with optimal configuration
        let performance_projection = PerformanceEstimate {
            tier: PerformanceTier::Excellent,
            utilization_percent: 85.0,
            latency_ms: 25.0,
            throughput: 50.0,
            estimated_latency_ms: 25.0,
            estimated_throughput: 50.0,
            confidence: 0.85,
            performance_tier: PerformanceTier::Excellent,
        };

        Ok(OptimalConfiguration {
            name: "AI-Optimized Configuration".to_string(),
            cpu_recommendation,
            gpu_recommendation: Some(gpu_recommendation),
            memory_gb: 32.0,    // Default recommendation
            storage_gb: 1000.0, // Default recommendation
            estimated_cost: total_cost.clone(),
            memory_recommendation,
            storage_recommendation,
            network_recommendation,
            total_cost,
            performance_projection: format!(
                "Expected performance: {:?}",
                performance_projection.tier
            ),
        })
    }

    /// Get cached system information or fetch new information
    async fn get_system_info(&mut self) -> Result<SystemInfo> {
        // Check if cached info is still valid
        if let Some(cached) = &self.cached_system_info {
            // TODO: Implement cache timestamp validation
            // For now, return cached if available within cache duration
            return Ok(cached.clone());
        }

        info!("Gathering system information using hardware-query");

        // Use hardware-query for comprehensive hardware detection
        let hardware_info = HardwareInfo::query().map_err(|e| {
            SystemAnalysisError::system_info(format!("Hardware query failed: {}", e))
        })?;

        // Convert hardware-query data to our SystemInfo format
        let cpu = hardware_info.cpu();
        let cpu_info = CpuInfo {
            brand: format!("{} {}", cpu.vendor(), cpu.model_name()),
            physical_cores: cpu.physical_cores() as usize,
            logical_cores: cpu.logical_cores() as usize,
            base_frequency: cpu.base_frequency() as u64,
            max_frequency: Some(cpu.max_frequency() as u64),
            cache_size: None, // Not provided by hardware-query
            architecture: cpu.architecture().to_string(),
        };

        // Convert GPU information
        let gpu_info: Vec<GpuInfo> = hardware_info
            .gpus()
            .iter()
            .map(|gpu| {
                GpuInfo {
                    name: gpu.model_name().to_string(),
                    vendor: format!("{:?}", gpu.vendor()), // Convert enum to string
                    vram_size: if gpu.memory_gb() > 0.0 {
                        Some((gpu.memory_gb() * 1024.0) as u64)
                    } else {
                        None
                    },
                    compute_capability: None, // Not provided by hardware-query
                    opencl_support: false,    // Would need additional detection
                    cuda_support: format!("{:?}", gpu.vendor())
                        .to_lowercase()
                        .contains("nvidia"),
                }
            })
            .collect();

        // Convert NPU information if available
        let npu_info = hardware_info
            .npus()
            .iter()
            .map(|npu| {
                crate::types::NpuInfo {
                    vendor: format!("{:?}", npu.vendor()),
                    model_name: npu.model_name().to_string(),
                    tops_performance: None, // Not provided by current hardware-query API
                    supported_frameworks: Vec::new(), // Not provided by current hardware-query API
                    supported_dtypes: Vec::new(), // Not provided by current hardware-query API
                }
            })
            .collect();

        // Convert TPU information if available
        let tpu_info = hardware_info
            .tpus()
            .iter()
            .map(|tpu| {
                crate::types::TpuInfo {
                    vendor: format!("{:?}", tpu.vendor()),
                    model_name: tpu.model_name().to_string(),
                    architecture: "Unknown".to_string(), // Not provided by current hardware-query API
                    tops_performance: None, // Not provided by current hardware-query API
                    supported_frameworks: Vec::new(), // Not provided by current hardware-query API
                    supported_dtypes: Vec::new(), // Not provided by current hardware-query API
                }
            })
            .collect();

        // Convert FPGA information if available
        let fpga_info = hardware_info
            .fpgas()
            .iter()
            .map(|fpga| {
                crate::types::FpgaInfo {
                    vendor: format!("{:?}", fpga.vendor),
                    family: format!("{:?}", fpga.family),
                    model_name: "Unknown".to_string(), // Not provided by current hardware-query API
                    logic_elements: None,              // Not provided by current hardware-query API
                    memory_blocks: None,               // Not provided by current hardware-query API
                    dsp_blocks: None,                  // Not provided by current hardware-query API
                }
            })
            .collect();

        // Convert ARM information if available
        let arm_info = hardware_info
            .arm_hardware()
            .map(|arm| crate::types::ArmInfo {
                system_type: format!("{:?}", arm.system_type),
                board_model: "Unknown".to_string(), // Not provided by current hardware-query API
                cpu_architecture: cpu.architecture().to_string(),
                acceleration_features: Vec::new(), // Would need additional detection
                ml_capabilities: std::collections::HashMap::new(), // Would need additional detection
                interfaces: Vec::new(), // Would need additional detection
            });

        let memory = hardware_info.memory();
        let memory_info = MemoryInfo {
            total_ram: (memory.total_gb() * 1024.0) as u64, // Convert GB to MB
            available_ram: (memory.available_gb() * 1024.0) as u64, // Convert GB to MB
            memory_type: Some("Unknown".to_string()), // Not provided by current hardware-query API
            memory_speed: None,                       // Not provided by current hardware-query API
        };

        // Fallback to sysinfo for storage and network info
        let mut system = System::new_all();
        system.refresh_all();
        let storage_info = self.get_storage_info(&system)?;
        let network_info = self.get_network_info(&system).await?;

        // Get OS information from sysinfo since hardware-query doesn't expose it directly
        let system_info = SystemInfo {
            os_name: System::name().unwrap_or_else(|| "Unknown".to_string()),
            os_version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
            cpu_info,
            gpu_info,
            npu_info,
            tpu_info,
            fpga_info,
            arm_info,
            memory_info,
            storage_info,
            network_info,
        };

        debug!(
            "Hardware detection complete: {} GPUs, {} NPUs, {} TPUs, {} FPGAs",
            system_info.gpu_info.len(),
            system_info.npu_info.len(),
            system_info.tpu_info.len(),
            system_info.fpga_info.len()
        );

        Ok(system_info)
    }

    /// Extract CPU information from system
    #[allow(dead_code)]
    fn get_cpu_info(&self, system: &System) -> Result<CpuInfo> {
        let cpus = system.cpus();

        if cpus.is_empty() {
            return Err(SystemAnalysisError::system_info(
                "No CPU information available",
            ));
        }

        let cpu = &cpus[0];
        let physical_cores = system.physical_core_count().unwrap_or(1);
        let logical_cores = cpus.len();

        Ok(CpuInfo {
            brand: cpu.brand().to_string(),
            physical_cores,
            logical_cores,
            base_frequency: cpu.frequency().max(1000), // MHz
            max_frequency: None,                       // Would need additional system calls
            cache_size: None,                          // Would need additional system calls
            architecture: std::env::consts::ARCH.to_string(),
        })
    }

    /// Extract GPU information
    #[allow(dead_code)]
    async fn get_gpu_info(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        if !self.config.enable_gpu_detection {
            return Ok(gpus);
        }

        #[cfg(feature = "gpu-detection")]
        {
            // GPU detection via NVML would be implemented here
            // Currently disabled until nvml_wrapper is available
            /*
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
            */
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
    #[allow(dead_code)]
    fn get_memory_info(&self, system: &System) -> Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_ram: system.total_memory() / 1024, // Convert to MB
            available_ram: system.available_memory() / 1024, // Convert to MB
            memory_type: None,                       // Would need additional system calls
            memory_speed: None,                      // Would need additional system calls
        })
    }

    /// Extract storage information
    fn get_storage_info(&self, _system: &System) -> Result<Vec<StorageInfo>> {
        let mut storage_devices = Vec::new();
        let disks = Disks::new_with_refreshed_list();

        for disk in &disks {
            let total_capacity = disk.total_space() / 1024 / 1024 / 1024; // Convert to GB
            let available_capacity = disk.available_space() / 1024 / 1024 / 1024; // Convert to GB

            storage_devices.push(StorageInfo {
                name: disk.name().to_string_lossy().to_string(),
                storage_type: format!("{:?}", disk.kind()),
                total_capacity,
                available_capacity,
                read_speed: None,  // Would need additional system calls
                write_speed: None, // Would need additional system calls
            });
        }

        if storage_devices.is_empty() {
            storage_devices.push(StorageInfo {
                name: "Unknown".to_string(),
                storage_type: "Unknown".to_string(),
                total_capacity: 1000,    // 1TB placeholder
                available_capacity: 500, // 500GB placeholder
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

        for (interface_name, _network) in &networks {
            interfaces.push(NetworkInterface {
                name: interface_name.clone(),
                interface_type: "Ethernet".to_string(), // Placeholder
                mac_address: "Unknown".to_string(),     // Would need additional system calls
                ip_addresses: vec![],                   // Would need additional system calls
                speed: Some(1000),                      // 1Gbps placeholder
            });
        }

        if interfaces.is_empty() {
            interfaces.push(NetworkInterface {
                name: "lo".to_string(),
                interface_type: "Loopback".to_string(),
                mac_address: "00:00:00:00:00:00".to_string(),
                ip_addresses: vec!["127.0.0.1".to_string()],
                speed: None,
            });
        }

        let estimated_bandwidth = interfaces
            .iter()
            .filter_map(|interface| interface.speed)
            .sum();

        Ok(NetworkInfo {
            interfaces,
            internet_connected: true, // Assume connected
            estimated_bandwidth: if estimated_bandwidth > 0 {
                Some(estimated_bandwidth)
            } else {
                None
            },
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
                        resource_type: req.resource_type.to_string(),
                        required: req.minimum.to_string(),
                        current: available.to_string(),
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
                    resource_type: req.resource_type.to_string(),
                    required: req.minimum.to_string(),
                    current: "Not Available".to_string(),
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
        let satisfaction_score = self
            .resource_pool
            .satisfaction_score(&workload_requirements.resource_requirements);
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
            tier: performance_tier,
            utilization_percent: 75.0,
            latency_ms: estimated_latency_ms,
            throughput: estimated_throughput,
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
            if score < avg_score - 2.0 {
                // Significantly below average
                let impact = if score < 3.0 {
                    BottleneckImpact::Severe
                } else if score < 5.0 {
                    BottleneckImpact::Moderate
                } else {
                    BottleneckImpact::Minor
                };

                let suggestions = self.generate_bottleneck_suggestions(&resource_type);

                bottlenecks.push(Bottleneck {
                    resource_type: resource_type.to_string(),
                    description: format!("{resource_type} performance is below system average ({score:.1} vs {avg_score:.1})"),
                    impact,
                    solution: suggestions.join(", "),
                    suggestions: suggestions.join(", "),
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
            ResourceType::Custom(_) => vec!["Review custom resource requirements".to_string()],
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
            recommendations
                .push("System meets all requirements for optimal performance".to_string());
        } else {
            if !missing_requirements.is_empty() {
                recommendations.push(format!(
                    "Address {} missing requirements",
                    missing_requirements.len()
                ));
            }

            if !bottlenecks.is_empty() {
                recommendations.push(format!("Resolve {} system bottlenecks", bottlenecks.len()));
            }

            // Add specific recommendations based on workload type
            if let Some(workload) = &workload_requirements.workload {
                match workload {
                    WorkloadType::AIInference => {
                        if system_profile.gpu_score() < 6.0 {
                            recommendations
                                .push("Consider GPU acceleration for AI inference".to_string());
                        }
                    }
                    WorkloadType::MemoryIntensive => {
                        if system_profile.memory_score() < 7.0 {
                            recommendations.push(
                                "Increase memory capacity for memory-intensive workloads"
                                    .to_string(),
                            );
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
        let memory_req = workload_requirements
            .resource_requirements
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
    fn estimate_default_utilization(
        &self,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<HashMap<ResourceType, f64>> {
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
        resource_type: &str,
        required: &str,
        available: &str,
        _system_profile: &SystemProfile,
    ) -> Result<UpgradeRecommendation> {
        let (recommendation, estimated_improvement, priority) = match resource_type {
            "CPU" => (
                "Upgrade to a higher-performance CPU with more cores".to_string(),
                "30-50% performance improvement".to_string(),
                UpgradePriority::High,
            ),
            "GPU" => (
                "Add or upgrade GPU for compute acceleration".to_string(),
                "2-10x performance improvement for GPU workloads".to_string(),
                UpgradePriority::Critical,
            ),
            "Memory" => (
                format!("Increase RAM from {available} to {required}"),
                "Eliminate memory bottlenecks".to_string(),
                UpgradePriority::High,
            ),
            "Storage" => (
                "Upgrade to faster NVMe SSD storage".to_string(),
                "Reduce I/O latency by 50-90%".to_string(),
                UpgradePriority::Medium,
            ),
            "Network" => (
                "Upgrade network connection speed".to_string(),
                "Reduce network latency and increase throughput".to_string(),
                UpgradePriority::Low,
            ),
            _ => (
                "Review custom resource requirements".to_string(),
                "Variable improvement".to_string(),
                UpgradePriority::Medium,
            ),
        };

        let resource_type_enum = match resource_type {
            "CPU" => crate::resources::ResourceType::CPU,
            "GPU" => crate::resources::ResourceType::GPU,
            "Memory" => crate::resources::ResourceType::Memory,
            "Storage" => crate::resources::ResourceType::Storage,
            "Network" => crate::resources::ResourceType::Network,
            _ => crate::resources::ResourceType::Custom(0),
        };

        Ok(UpgradeRecommendation {
            component: resource_type.to_string(),
            description: recommendation.clone(),
            priority,
            estimated_cost: None, // Would be populated in real implementation
            resource_type: resource_type_enum,
            recommendation,
            estimated_improvement,
            cost_estimate: None,
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
                component: "System".to_string(),
                description: "Consider a comprehensive system upgrade".to_string(),
                priority: UpgradePriority::Medium,
                estimated_cost: Some(CostEstimate {
                    min_cost_usd: 1500.0,
                    max_cost_usd: 5000.0,
                    currency: "USD".to_string(),
                    breakdown: Vec::new(),
                }),
                resource_type: ResourceType::CPU,
                recommendation: "Consider a comprehensive system upgrade".to_string(),
                estimated_improvement: "Significant overall performance improvement".to_string(),
                cost_estimate: Some(CostEstimate {
                    min_cost_usd: 1500.0,
                    max_cost_usd: 5000.0,
                    currency: "USD".to_string(),
                    breakdown: Vec::new(),
                }),
            });
        }

        Ok(recommendations)
    }

    // Helper methods for optimal configuration recommendations
    fn recommend_optimal_cpu(
        &self,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<String> {
        // Analyze workload requirements and recommend optimal CPU
        let cpu_req = workload_requirements
            .resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::CPU);

        let recommendation = match cpu_req {
            Some(req) => match &req.minimum {
                ResourceAmount::Level(level) => match level {
                    CapabilityLevel::Exceptional => {
                        "High-end workstation CPU (e.g., Intel Xeon W or AMD Threadripper PRO)"
                    }
                    CapabilityLevel::VeryHigh => {
                        "High-performance CPU (e.g., Intel Core i9 or AMD Ryzen 9)"
                    }
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

    fn recommend_optimal_gpu(
        &self,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<String> {
        let gpu_req = workload_requirements
            .resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::GPU);

        let recommendation = match gpu_req {
            Some(_) => {
                // Check if this is an AI workload
                if let Some(workload) = &workload_requirements.workload {
                    match workload {
                        WorkloadType::AIInference | WorkloadType::AITraining => {
                            "High-memory GPU optimized for AI (e.g., NVIDIA RTX 4090, A6000, or H100)"
                        }
                        _ => "Dedicated GPU with good compute performance",
                    }
                } else {
                    "Modern dedicated GPU with adequate VRAM"
                }
            }
            None => "Integrated graphics sufficient, dedicated GPU optional",
        };

        Ok(recommendation.to_string())
    }

    fn recommend_optimal_memory(
        &self,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<String> {
        let memory_req = workload_requirements
            .resource_requirements
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
            gb if gb >= 32.0 => format!("{}GB+ DDR4/DDR5 RAM", gb as u32),
            gb if gb >= 16.0 => format!("{}GB+ DDR4 RAM", gb as u32),
            gb if gb >= 8.0 => format!("{}GB+ DDR4 RAM", gb as u32),
            _ => "8GB+ DDR4 RAM".to_string(),
        };

        Ok(recommendation)
    }

    fn recommend_optimal_storage(
        &self,
        workload_requirements: &WorkloadRequirements,
    ) -> Result<String> {
        let storage_req = workload_requirements
            .resource_requirements
            .iter()
            .find(|req| req.resource_type == ResourceType::Storage)
            .and_then(|req| match &req.minimum {
                ResourceAmount::Gigabytes(gb) => Some(*gb),
                _ => None,
            })
            .unwrap_or(500.0);

        let recommendation = match storage_req {
            gb if gb >= 2000.0 => format!("{}GB+ high-speed NVMe SSD (PCIe 4.0+)", gb as u32),
            gb if gb >= 1000.0 => format!("{}GB+ NVMe SSD (PCIe 3.0+)", gb as u32),
            gb if gb >= 500.0 => format!("{}GB+ SATA SSD", gb as u32),
            gb if gb >= 250.0 => format!("{}GB+ SSD", gb as u32),
            _ => "250GB+ SSD".to_string(),
        };

        Ok(recommendation)
    }

    fn recommend_optimal_network(
        &self,
        _workload_requirements: &WorkloadRequirements,
    ) -> Result<String> {
        Ok("Gigabit Ethernet connection (wired preferred for consistency)".to_string())
    }

    /// Check if a specific AI model can run efficiently on the current system
    pub fn check_model_compatibility(
        &self,
        model: &crate::workloads::AIModel,
    ) -> Result<crate::types::ModelCompatibilityResult> {
        let system_profile = match &self.cached_system_info {
            Some(info) => {
                let capability_profile = CapabilityProfile::from_system_info(info);
                SystemProfile::builder()
                    .cpu_score(capability_profile.scores.cpu_score)
                    .gpu_score(capability_profile.scores.gpu_score)
                    .npu_score(capability_profile.scores.npu_score.unwrap_or(0.0))
                    .tpu_score(capability_profile.scores.tpu_score.unwrap_or(0.0))
                    .fpga_score(capability_profile.scores.fpga_score.unwrap_or(0.0))
                    .arm_optimization_score(
                        capability_profile
                            .scores
                            .arm_optimization_score
                            .unwrap_or(0.0),
                    )
                    .memory_score(capability_profile.scores.memory_score)
                    .storage_score(capability_profile.scores.storage_score)
                    .network_score(capability_profile.scores.network_score)
                    .system_info(info.clone())
                    .build()
            }
            None => {
                return Err(SystemAnalysisError::system_info(
                    "No system information available. Run analyze_system() first.",
                ));
            }
        };

        // Determine memory requirements based on model size and quantization
        let memory_required = match model.quantization {
            crate::workloads::QuantizationLevel::None => {
                model.size_in_bytes as f64 / 1_073_741_824.0
            } // Convert to GB
            crate::workloads::QuantizationLevel::Int8 => {
                model.size_in_bytes as f64 / 2_147_483_648.0
            } // Half precision
            crate::workloads::QuantizationLevel::Int4 => {
                model.size_in_bytes as f64 / 4_294_967_296.0
            } // Quarter precision
            crate::workloads::QuantizationLevel::Custom(ratio) => {
                model.size_in_bytes as f64 * ratio / 1_073_741_824.0
            }
        };

        // Check if system has enough memory
        let has_enough_memory =
            system_profile.system_info.memory_info.total_ram as f64 / 1024.0 >= memory_required;

        // Check for hardware accelerator compatibility
        let accelerator_compatibility =
            self.check_accelerator_compatibility(&system_profile, model);

        // Analyze optimal quantization
        let optimal_quantization = self.suggest_optimal_quantization(&system_profile, model);

        // Calculate expected inference speed
        let inference_speed = self.calculate_inference_speed(&system_profile, model);

        Ok(crate::types::ModelCompatibilityResult {
            can_run: has_enough_memory && accelerator_compatibility.is_compatible,
            memory_sufficient: has_enough_memory,
            accelerator_compatibility,
            optimal_quantization,
            expected_inference_speed: inference_speed,
            bottlenecks: self.identify_model_bottlenecks(&system_profile, model),
            recommended_batch_size: self.suggest_batch_size(&system_profile, model),
        })
    }

    /// Check if available accelerators can run the model efficiently
    fn check_accelerator_compatibility(
        &self,
        profile: &SystemProfile,
        model: &crate::workloads::AIModel,
    ) -> crate::types::AcceleratorCompatibility {
        // Default to CPU-only compatibility
        let mut compatibility = crate::types::AcceleratorCompatibility {
            is_compatible: true, // Assume at least CPU compatibility
            compatible_devices: Vec::new(),
            recommended_device: None,
            expected_performance: crate::types::PerformanceLevel::Low,
        };

        // Check GPU compatibility first
        if !profile.system_info.gpu_info.is_empty() {
            let gpu_memory_sufficient = profile.system_info.gpu_info.iter().any(|gpu| {
                if let Some(vram) = gpu.vram_size {
                    // Convert vram from MB to GB and check against model requirements
                    (vram as f64 / 1024.0) >= (model.memory_required * 0.9) // 10% buffer
                } else {
                    false
                }
            });

            if gpu_memory_sufficient {
                compatibility
                    .compatible_devices
                    .push(crate::types::AcceleratorDevice::GPU);
                compatibility.expected_performance = crate::types::PerformanceLevel::Medium;
                compatibility.recommended_device = Some(crate::types::AcceleratorDevice::GPU);
            }
        }

        // Check NPU compatibility
        if !profile.system_info.npu_info.is_empty() {
            let npu_compatible = profile.system_info.npu_info.iter().any(|npu| {
                // Check if NPU supports the model's framework
                npu.supported_frameworks.contains(&model.framework)
            });

            if npu_compatible {
                compatibility
                    .compatible_devices
                    .push(crate::types::AcceleratorDevice::NPU);
                compatibility.expected_performance = crate::types::PerformanceLevel::High;
                compatibility.recommended_device = Some(crate::types::AcceleratorDevice::NPU);
            }
        }

        // Check TPU compatibility
        if !profile.system_info.tpu_info.is_empty() {
            let tpu_compatible = profile.system_info.tpu_info.iter().any(|tpu| {
                // Check if TPU supports the model's framework
                tpu.supported_frameworks.contains(&model.framework)
            });

            if tpu_compatible {
                compatibility
                    .compatible_devices
                    .push(crate::types::AcceleratorDevice::TPU);
                // TPUs typically provide best performance for compatible models
                compatibility.expected_performance = crate::types::PerformanceLevel::VeryHigh;
                compatibility.recommended_device = Some(crate::types::AcceleratorDevice::TPU);
            }
        }

        // Add CPU as fallback option
        compatibility
            .compatible_devices
            .push(crate::types::AcceleratorDevice::CPU);

        compatibility
    }

    /// Suggest optimal quantization based on available hardware
    fn suggest_optimal_quantization(
        &self,
        profile: &SystemProfile,
        model: &crate::workloads::AIModel,
    ) -> crate::types::QuantizationSuggestion {
        // If model is small enough to fit in memory without quantization, no need to quantize
        if model.size_in_bytes as f64 / 1_073_741_824.0
            < profile.system_info.memory_info.total_ram as f64 / 2048.0
        {
            // Half of available RAM in GB
            return crate::types::QuantizationSuggestion {
                recommended_level: crate::workloads::QuantizationLevel::None,
                reasoning: "Model fits in memory without quantization".to_string(),
                performance_impact: crate::types::PerformanceImpact::None,
            };
        }

        // Check NPU/TPU support for quantized models
        let has_dedicated_accelerator =
            !profile.system_info.npu_info.is_empty() || !profile.system_info.tpu_info.is_empty();

        if has_dedicated_accelerator {
            // Many NPUs/TPUs prefer or require quantization for optimal performance
            return crate::types::QuantizationSuggestion {
                recommended_level: crate::workloads::QuantizationLevel::Int8,
                reasoning: "Optimal for neural accelerators with minimal accuracy loss".to_string(),
                performance_impact: crate::types::PerformanceImpact::Positive,
            };
        }

        // If memory is tight but GPU is available
        if profile.system_info.memory_info.total_ram as f64 / 1024.0
            < model.size_in_bytes as f64 / 1_073_741_824.0 * 2.0
        {
            return crate::types::QuantizationSuggestion {
                recommended_level: crate::workloads::QuantizationLevel::Int8,
                reasoning: "Memory constraints require quantization with reasonable accuracy"
                    .to_string(),
                performance_impact: crate::types::PerformanceImpact::Mixed,
            };
        }

        // Default recommendation for most scenarios
        crate::types::QuantizationSuggestion {
            recommended_level: crate::workloads::QuantizationLevel::None,
            reasoning: "No quantization needed for optimal accuracy".to_string(),
            performance_impact: crate::types::PerformanceImpact::None,
        }
    }

    /// Calculate expected inference speed for a model
    fn calculate_inference_speed(
        &self,
        profile: &SystemProfile,
        model: &crate::workloads::AIModel,
    ) -> f64 {
        // Base speed depends on model size and complexity
        let base_speed = match model.parameters {
            params if params >= 100_000_000_000 => 0.5, // Very large models
            params if params >= 10_000_000_000 => 2.0,  // Large models
            params if params >= 1_000_000_000 => 10.0,  // Medium models
            params if params >= 100_000_000 => 50.0,    // Small models
            _ => 100.0,                                 // Very small models
        };

        // Adjust based on available hardware
        let hardware_multiplier = if !profile.system_info.tpu_info.is_empty() {
            10.0 // TPUs are very fast for supported models
        } else if !profile.system_info.npu_info.is_empty() {
            5.0 // NPUs provide good acceleration
        } else if !profile.system_info.gpu_info.is_empty() {
            2.0 // GPUs provide moderate acceleration
        } else {
            1.0 // CPU only
        };

        // Adjust for quantization
        let quantization_multiplier = match model.quantization {
            crate::workloads::QuantizationLevel::None => 1.0,
            crate::workloads::QuantizationLevel::Int8 => 1.5,
            crate::workloads::QuantizationLevel::Int4 => 2.0,
            crate::workloads::QuantizationLevel::Custom(ratio) => 1.0 / ratio,
        };

        base_speed * hardware_multiplier * quantization_multiplier
    }

    /// Identify model-specific bottlenecks
    fn identify_model_bottlenecks(
        &self,
        profile: &SystemProfile,
        model: &crate::workloads::AIModel,
    ) -> Vec<crate::types::ModelBottleneck> {
        let mut bottlenecks = Vec::new();

        // Check memory bottleneck
        let memory_required_gb = model.size_in_bytes as f64 / 1_073_741_824.0;
        let available_memory_gb = profile.system_info.memory_info.total_ram as f64 / 1024.0;

        if memory_required_gb > available_memory_gb * 0.8 {
            bottlenecks.push(crate::types::ModelBottleneck {
                bottleneck_type: crate::types::ModelBottleneckType::Memory,
                description: format!("Model requires {memory_required_gb:.1}GB but only {available_memory_gb:.1}GB available"),
                severity: if memory_required_gb > available_memory_gb {
                    crate::types::BottleneckSeverity::Critical
                } else {
                    crate::types::BottleneckSeverity::High
                },
                recommendation: "Consider upgrading RAM or using model quantization".to_string(),
            });
        }

        // Check compute bottleneck
        if model.parameters > 10_000_000_000
            && profile.gpu_score < 6.0
            && profile.ai_accelerator_score < 6.0
        {
            bottlenecks.push(crate::types::ModelBottleneck {
                bottleneck_type: crate::types::ModelBottleneckType::Compute,
                description: "Large model requires significant compute resources".to_string(),
                severity: crate::types::BottleneckSeverity::High,
                recommendation: "Consider upgrading GPU or adding AI accelerator".to_string(),
            });
        }

        // Check framework support
        if !profile.supported_frameworks().contains(&model.framework) {
            bottlenecks.push(crate::types::ModelBottleneck {
                bottleneck_type: crate::types::ModelBottleneckType::FrameworkSupport,
                description: format!(
                    "Framework {} not supported by available hardware",
                    model.framework
                ),
                severity: crate::types::BottleneckSeverity::Medium,
                recommendation: "Install appropriate framework or convert model format".to_string(),
            });
        }

        bottlenecks
    }

    /// Suggest optimal batch size for a model
    fn suggest_batch_size(
        &self,
        profile: &SystemProfile,
        model: &crate::workloads::AIModel,
    ) -> u32 {
        // Base batch size depends on model size and available memory
        let available_memory_gb = profile.system_info.memory_info.total_ram as f64 / 1024.0;
        let model_memory_gb = model.size_in_bytes as f64 / 1_073_741_824.0;

        // Calculate how much memory is left for batch processing
        let remaining_memory_gb = available_memory_gb - model_memory_gb;

        if remaining_memory_gb <= 2.0 {
            1 // Very tight memory
        } else if remaining_memory_gb <= 8.0 {
            2 // Limited memory
        } else if remaining_memory_gb <= 16.0 {
            4 // Moderate memory
        } else {
            8 // Ample memory
        }
    }

    /// Generate AI-specific hardware recommendations
    pub fn recommend_ai_hardware_upgrades(
        &self,
        workload: &crate::types::AIWorkloadRequirements,
    ) -> Result<crate::types::AIUpgradeRecommendations> {
        let system_profile = match &self.cached_system_info {
            Some(info) => {
                let capability_profile = CapabilityProfile::from_system_info(info);
                SystemProfile::builder()
                    .cpu_score(capability_profile.scores.cpu_score)
                    .gpu_score(capability_profile.scores.gpu_score)
                    .npu_score(capability_profile.scores.npu_score.unwrap_or(0.0))
                    .tpu_score(capability_profile.scores.tpu_score.unwrap_or(0.0))
                    .fpga_score(capability_profile.scores.fpga_score.unwrap_or(0.0))
                    .arm_optimization_score(
                        capability_profile
                            .scores
                            .arm_optimization_score
                            .unwrap_or(0.0),
                    )
                    .memory_score(capability_profile.scores.memory_score)
                    .storage_score(capability_profile.scores.storage_score)
                    .network_score(capability_profile.scores.network_score)
                    .system_info(info.clone())
                    .build()
            }
            None => {
                return Err(SystemAnalysisError::system_info(
                    "No system information available. Run analyze_system() first.",
                ));
            }
        };

        let mut recommendations = crate::types::AIUpgradeRecommendations {
            memory_upgrade: None,
            gpu_upgrade: None,
            accelerator_recommendation: None,
            storage_recommendation: None,
            estimated_cost: None,
            performance_gain: None,
            priority: crate::types::UpgradePriority::Medium,
        };

        // Check if memory upgrade is needed
        if system_profile.system_info.memory_info.total_ram as f64 / 1024.0
            < workload.required_model_memory * 1.5
        {
            let current_ram_gb = system_profile.system_info.memory_info.total_ram as f64 / 1024.0;
            let recommended_ram_gb = (workload.required_model_memory * 2.0).max(16.0); // At least double required memory or 16GB

            recommendations.memory_upgrade = Some(crate::types::MemoryUpgrade {
                current_ram_gb,
                recommended_ram_gb,
                description: format!(
                    "Upgrade RAM from {current_ram_gb:.1} GB to {recommended_ram_gb:.1} GB for optimal AI model performance"
                ),
                estimated_cost_usd: (recommended_ram_gb - current_ram_gb).max(0.0) * 10.0, // ~$10 per GB
            });

            recommendations.priority = crate::types::UpgradePriority::High;
        }

        // Check if GPU upgrade is needed
        if !workload
            .required_accelerator_types
            .contains(&crate::types::AIAcceleratorType::NPU)
            && !workload
                .required_accelerator_types
                .contains(&crate::types::AIAcceleratorType::TPU)
            && workload
                .required_accelerator_types
                .contains(&crate::types::AIAcceleratorType::GPU)
        {
            // Calculate if current GPU is sufficient
            let has_sufficient_gpu = system_profile.system_info.gpu_info.iter().any(|gpu| {
                if let Some(vram) = gpu.vram_size {
                    // VRAM in GB
                    let vram_gb = vram as f64 / 1024.0;
                    // Check if VRAM is sufficient for the model with some overhead
                    vram_gb >= workload.required_model_memory * 1.1 &&
                    // Check if GPU has CUDA support if needed
                    (!workload.required_frameworks.contains(&"CUDA".to_string()) || gpu.cuda_support)
                } else {
                    false
                }
            });

            if !has_sufficient_gpu {
                // Recommend appropriate GPU based on model size
                let (gpu_model, vram_gb, estimated_cost) = if workload.required_model_memory <= 8.0
                {
                    ("NVIDIA RTX 3060 or AMD RX 6700", 12.0, 400.0)
                } else if workload.required_model_memory <= 24.0 {
                    ("NVIDIA RTX 4080 or AMD RX 7900", 16.0, 800.0)
                } else {
                    ("NVIDIA RTX 4090 or A6000", 24.0, 1500.0)
                };

                recommendations.gpu_upgrade = Some(crate::types::GPUUpgrade {
                    current_gpu: system_profile
                        .system_info
                        .gpu_info
                        .first()
                        .map(|g| g.name.clone())
                        .unwrap_or_else(|| "Unknown".to_string()),
                    recommended_gpu: gpu_model.to_string(),
                    vram_required_gb: workload.required_model_memory,
                    vram_recommended_gb: vram_gb,
                    description: format!(
                        "Upgrade to {gpu_model} with {vram_gb}GB VRAM for optimal AI performance"
                    ),
                    estimated_cost_usd: estimated_cost,
                });

                recommendations.priority = crate::types::UpgradePriority::Critical;
            }
        }

        // Check if specialized AI accelerator is needed
        if (workload
            .required_accelerator_types
            .contains(&crate::types::AIAcceleratorType::NPU)
            || workload
                .required_accelerator_types
                .contains(&crate::types::AIAcceleratorType::TPU))
            && system_profile.system_info.npu_info.is_empty()
            && system_profile.system_info.tpu_info.is_empty()
        {
            // Select an appropriate accelerator based on workload requirements
            let (accelerator_name, accelerator_type, tops, estimated_cost) =
                if let Some(required_tops) = workload.required_tops {
                    if required_tops > 200.0 {
                        ("NVIDIA Jetson AGX Orin", "NPU", 275.0, 2000.0)
                    } else if required_tops > 100.0 {
                        ("Google Coral Dev Board", "TPU", 150.0, 120.0)
                    } else {
                        ("Intel Neural Compute Stick 2", "NPU", 100.0, 80.0)
                    }
                } else {
                    ("Google Coral Dev Board", "TPU", 150.0, 120.0)
                };

            recommendations.accelerator_recommendation =
                Some(crate::types::AcceleratorRecommendation {
                    accelerator_name: accelerator_name.to_string(),
                    accelerator_type: accelerator_type.to_string(),
                    tops_performance: tops,
                    description: format!(
                        "Add {accelerator_name} ({tops} TOPS) for specialized AI acceleration"
                    ),
                    estimated_cost_usd: estimated_cost,
                });

            recommendations.priority = crate::types::UpgradePriority::High;
        }

        // Calculate estimated performance gain
        recommendations.performance_gain = Some(crate::types::PerformanceGainEstimate {
            latency_improvement_percent: 60.0,
            throughput_improvement_percent: 80.0,
            energy_efficiency_improvement_percent: 30.0,
            description: "Significant performance improvements for AI workloads".to_string(),
        });

        // Calculate total estimated cost
        let total_cost = recommendations
            .memory_upgrade
            .as_ref()
            .map(|u| u.estimated_cost_usd)
            .unwrap_or(0.0)
            + recommendations
                .gpu_upgrade
                .as_ref()
                .map(|u| u.estimated_cost_usd)
                .unwrap_or(0.0)
            + recommendations
                .accelerator_recommendation
                .as_ref()
                .map(|u| u.estimated_cost_usd)
                .unwrap_or(0.0);

        if total_cost > 0.0 {
            recommendations.estimated_cost = Some(crate::types::CostEstimate {
                min_cost_usd: total_cost * 0.8, // 20% potential discount
                max_cost_usd: total_cost * 1.2, // 20% potential premium
                currency: "USD".to_string(),
                breakdown: Vec::new(),
            });
        }

        Ok(recommendations)
    }

    /// Estimate performance gain from specialized AI hardware
    pub fn estimate_ai_acceleration_benefit(
        &self,
        _workload: &crate::types::AIWorkloadRequirements,
    ) -> Result<crate::types::AccelerationBenefit> {
        let system_profile = match &self.cached_system_info {
            Some(info) => {
                let capability_profile = CapabilityProfile::from_system_info(info);
                SystemProfile::builder()
                    .cpu_score(capability_profile.scores.cpu_score)
                    .gpu_score(capability_profile.scores.gpu_score)
                    .npu_score(capability_profile.scores.npu_score.unwrap_or(0.0))
                    .tpu_score(capability_profile.scores.tpu_score.unwrap_or(0.0))
                    .fpga_score(capability_profile.scores.fpga_score.unwrap_or(0.0))
                    .arm_optimization_score(
                        capability_profile
                            .scores
                            .arm_optimization_score
                            .unwrap_or(0.0),
                    )
                    .memory_score(capability_profile.scores.memory_score)
                    .storage_score(capability_profile.scores.storage_score)
                    .network_score(capability_profile.scores.network_score)
                    .system_info(info.clone())
                    .build()
            }
            None => {
                return Err(SystemAnalysisError::system_info(
                    "No system information available. Run analyze_system() first.",
                ));
            }
        };

        // Base performance on current hardware
        let has_gpu = !system_profile.system_info.gpu_info.is_empty();
        let has_npu = !system_profile.system_info.npu_info.is_empty();
        let has_tpu = !system_profile.system_info.tpu_info.is_empty();

        // Calculate potential improvements with specialized accelerators
        let current_performance = if has_npu || has_tpu {
            1.0 // Already has specialized hardware
        } else if has_gpu {
            0.3 // GPU acceleration vs. specialized hardware
        } else {
            0.1 // CPU only vs. specialized hardware
        };

        // Calculate specific metrics
        let _latency_improvement = (1.0 / current_performance - 1.0) * 100.0;
        let _throughput_improvement = ((1.0 / current_performance) * 1.2 - 1.0) * 100.0; // 20% extra for parallelism
        let power_efficiency_improvement = (1.0 / current_performance * 0.7 - 1.0) * 100.0; // 30% less efficiency due to specialized hardware

        let description = if has_npu || has_tpu {
            "System already has AI acceleration hardware. No significant improvement expected."
                .to_string()
        } else if has_gpu {
            "Specialized AI accelerators would provide significant performance improvements over GPU-only acceleration.".to_string()
        } else {
            "Dedicated AI acceleration hardware would provide massive performance improvements over CPU-only inference.".to_string()
        };

        Ok(crate::types::AccelerationBenefit {
            speed_improvement_factor: 1.0 / current_performance,
            power_efficiency_improvement,
            cost_per_performance: 1.0, // Placeholder
            description,
            confidence_level: 0.8,
        })
    }

    // ...existing code...
}

impl Default for SystemAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::ResourceRequirement;

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
        assert!(!analyzer.config.enable_gpu_detection);
        assert!(analyzer.config.enable_network_testing);
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
                .recommended_gb(8.0),
        );

        let compatibility = analyzer.check_compatibility(&system_profile, &workload_requirements);
        assert!(compatibility.is_ok());

        let result = compatibility.unwrap();
        assert!(result.is_compatible);
        assert!(result.score >= 0.0 && result.score <= 10.0);
    }

    #[test]
    fn test_workload_requirements_builder() {
        let mut requirements = WorkloadRequirements::new("test-workload");

        requirements.add_resource_requirement(
            ResourceRequirement::new(ResourceType::CPU)
                .minimum_level(CapabilityLevel::Medium)
                .recommended_level(CapabilityLevel::High),
        );

        requirements.add_resource_requirement(
            ResourceRequirement::new(ResourceType::Memory)
                .minimum_gb(8.0)
                .recommended_gb(16.0),
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

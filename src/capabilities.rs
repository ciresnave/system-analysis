//! Hardware capability analysis and profiling.

use crate::resources::{ResourceType, CapabilityLevel, ResourceAmount};
use crate::types::{SystemInfo, CpuInfo, GpuInfo, MemoryInfo, StorageInfo, NetworkInfo};
use serde::{Deserialize, Serialize};

/// Comprehensive capability profile for a system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityProfile {
    /// CPU capabilities
    pub cpu_capabilities: CpuCapabilities,
    /// GPU capabilities
    pub gpu_capabilities: GpuCapabilities,
    /// Memory capabilities
    pub memory_capabilities: MemoryCapabilities,
    /// Storage capabilities
    pub storage_capabilities: StorageCapabilities,
    /// Network capabilities
    pub network_capabilities: NetworkCapabilities,
    /// Overall capability scores
    pub scores: CapabilityScores,
    /// Capability metadata
    pub metadata: CapabilityMetadata,
}

impl CapabilityProfile {
    /// Create a capability profile from system information
    pub fn from_system_info(system_info: &SystemInfo) -> Self {
        let cpu_capabilities = CpuCapabilities::from_cpu_info(&system_info.cpu_info);
        let gpu_capabilities = GpuCapabilities::from_gpu_info(&system_info.gpu_info);
        let memory_capabilities = MemoryCapabilities::from_memory_info(&system_info.memory_info);
        let storage_capabilities = StorageCapabilities::from_storage_info(&system_info.storage_info);
        let network_capabilities = NetworkCapabilities::from_network_info(&system_info.network_info);

        let scores = CapabilityScores::calculate(
            &cpu_capabilities,
            &gpu_capabilities,
            &memory_capabilities,
            &storage_capabilities,
            &network_capabilities,
        );

        let metadata = CapabilityMetadata {
            analysis_version: "1.0".to_string(),
            created_at: chrono::Utc::now(),
            system_fingerprint: Self::generate_fingerprint(system_info),
        };

        Self {
            cpu_capabilities,
            gpu_capabilities,
            memory_capabilities,
            storage_capabilities,
            network_capabilities,
            scores,
            metadata,
        }
    }

    /// Generate a unique fingerprint for the system
    fn generate_fingerprint(system_info: &SystemInfo) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        system_info.cpu_info.brand.hash(&mut hasher);
        system_info.cpu_info.physical_cores.hash(&mut hasher);
        system_info.memory_info.total_ram.hash(&mut hasher);
        
        for gpu in &system_info.gpu_info {
            gpu.name.hash(&mut hasher);
            gpu.vendor.hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Get capability level for a specific resource type
    pub fn get_capability_level(&self, resource_type: &ResourceType) -> CapabilityLevel {
        match resource_type {
            ResourceType::CPU => self.scores.cpu_score.into(),
            ResourceType::GPU => self.scores.gpu_score.into(),
            ResourceType::Memory => self.scores.memory_score.into(),
            ResourceType::Storage => self.scores.storage_score.into(),
            ResourceType::Network => self.scores.network_score.into(),
            ResourceType::Custom(_) => CapabilityLevel::Medium, // Default for custom types
        }
    }

    /// Get resource amount for a specific resource type
    pub fn get_resource_amount(&self, resource_type: &ResourceType) -> Option<ResourceAmount> {
        match resource_type {
            ResourceType::CPU => Some(ResourceAmount::Score(self.scores.cpu_score)),
            ResourceType::GPU => Some(ResourceAmount::Score(self.scores.gpu_score)),
            ResourceType::Memory => Some(ResourceAmount::Gigabytes(
                self.memory_capabilities.total_ram_gb
            )),
            ResourceType::Storage => Some(ResourceAmount::Gigabytes(
                self.storage_capabilities.total_capacity_gb
            )),
            ResourceType::Network => Some(ResourceAmount::Score(self.scores.network_score)),
            ResourceType::Custom(_) => None,
        }
    }

    /// Check if system supports specific features
    pub fn supports_feature(&self, feature: &SystemFeature) -> bool {
        match feature {
            SystemFeature::CudaCompute => self.gpu_capabilities.cuda_support,
            SystemFeature::OpenCLCompute => self.gpu_capabilities.opencl_support,
            SystemFeature::AVXInstructions => self.cpu_capabilities.avx_support,
            SystemFeature::NVMeStorage => self.storage_capabilities.nvme_support,
            SystemFeature::HighBandwidthMemory => self.memory_capabilities.high_bandwidth,
            SystemFeature::VirtualizationSupport => self.cpu_capabilities.virtualization_support,
            SystemFeature::HighSpeedNetwork => self.network_capabilities.high_speed_support,
        }
    }
}

/// CPU-specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapabilities {
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores
    pub logical_cores: usize,
    /// Base frequency in MHz
    pub base_frequency_mhz: u64,
    /// Maximum frequency in MHz
    pub max_frequency_mhz: Option<u64>,
    /// Cache size in MB
    pub cache_size_mb: Option<u64>,
    /// Architecture family
    pub architecture: String,
    /// Instruction set extensions
    pub instruction_sets: Vec<String>,
    /// AVX support
    pub avx_support: bool,
    /// Virtualization support
    pub virtualization_support: bool,
    /// Thread performance score (0-10)
    pub thread_performance: f64,
    /// Multi-core efficiency (0-10)
    pub multicore_efficiency: f64,
}

impl CpuCapabilities {
    /// Create CPU capabilities from CPU info
    pub fn from_cpu_info(cpu_info: &CpuInfo) -> Self {
        let avx_support = cpu_info.brand.to_lowercase().contains("intel") || 
                         cpu_info.brand.to_lowercase().contains("amd");
        
        let virtualization_support = cpu_info.logical_cores > cpu_info.physical_cores;
        
        let thread_performance = Self::calculate_thread_performance(cpu_info);
        let multicore_efficiency = Self::calculate_multicore_efficiency(cpu_info);
        
        Self {
            physical_cores: cpu_info.physical_cores,
            logical_cores: cpu_info.logical_cores,
            base_frequency_mhz: cpu_info.base_frequency,
            max_frequency_mhz: cpu_info.max_frequency,
            cache_size_mb: cpu_info.cache_size,
            architecture: cpu_info.architecture.clone(),
            instruction_sets: vec![], // Would be populated from CPUID in real implementation
            avx_support,
            virtualization_support,
            thread_performance,
            multicore_efficiency,
        }
    }

    /// Calculate thread performance score
    fn calculate_thread_performance(cpu_info: &CpuInfo) -> f64 {
        let base_score = (cpu_info.base_frequency as f64 / 1000.0).min(5.0); // Up to 5 points for frequency
        let cache_bonus = cpu_info.cache_size.map(|c| (c as f64 / 16.0).min(2.0)).unwrap_or(0.0); // Up to 2 points for cache
        let arch_bonus = if cpu_info.architecture.contains("x86_64") { 1.0 } else { 0.5 };
        
        (base_score + cache_bonus + arch_bonus).min(10.0)
    }

    /// Calculate multicore efficiency score
    fn calculate_multicore_efficiency(cpu_info: &CpuInfo) -> f64 {
        let core_score = (cpu_info.physical_cores as f64 / 2.0).min(5.0); // Up to 5 points for cores
        let hyperthreading_bonus = if cpu_info.logical_cores > cpu_info.physical_cores { 2.0 } else { 0.0 };
        let architecture_bonus = if cpu_info.architecture.contains("x86_64") { 1.0 } else { 0.5 };
        
        (core_score + hyperthreading_bonus + architecture_bonus).min(10.0)
    }

    /// Get AI capability level based on CPU performance
    pub fn ai_capability_level(&self) -> crate::resources::CapabilityLevel {
        let combined_score = (self.thread_performance + self.multicore_efficiency) / 2.0;
        crate::resources::CapabilityLevel::from_numeric(combined_score)
    }
}

/// GPU-specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// Available GPUs
    pub gpus: Vec<GpuDevice>,
    /// Best GPU for compute
    pub primary_compute_gpu: Option<usize>,
    /// Total VRAM across all GPUs
    pub total_vram_gb: f64,
    /// CUDA support available
    pub cuda_support: bool,
    /// OpenCL support available
    pub opencl_support: bool,
    /// AI acceleration score (0-10)
    pub ai_acceleration_score: f64,
    /// Graphics performance score (0-10)
    pub graphics_score: f64,
    /// Compute performance score (0-10)
    pub compute_score: f64,
}

impl GpuCapabilities {
    /// Create GPU capabilities from GPU info
    pub fn from_gpu_info(gpu_info: &[GpuInfo]) -> Self {
        let gpus: Vec<GpuDevice> = gpu_info.iter().map(GpuDevice::from_gpu_info).collect();
        
        let total_vram_gb = gpus.iter()
            .map(|gpu| gpu.vram_gb.unwrap_or(0.0))
            .sum();
        
        let cuda_support = gpus.iter().any(|gpu| gpu.cuda_support);
        let opencl_support = gpus.iter().any(|gpu| gpu.opencl_support);
        
        let primary_compute_gpu = Self::find_best_compute_gpu(&gpus);
        
        let ai_acceleration_score = Self::calculate_ai_score(&gpus);
        let graphics_score = Self::calculate_graphics_score(&gpus);
        let compute_score = Self::calculate_compute_score(&gpus);
        
        Self {
            gpus,
            primary_compute_gpu,
            total_vram_gb,
            cuda_support,
            opencl_support,
            ai_acceleration_score,
            graphics_score,
            compute_score,
        }
    }

    /// Find the best GPU for compute workloads
    fn find_best_compute_gpu(gpus: &[GpuDevice]) -> Option<usize> {
        gpus.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.compute_capability_score.partial_cmp(&b.compute_capability_score).unwrap())
            .map(|(idx, _)| idx)
    }

    /// Calculate AI acceleration score
    fn calculate_ai_score(gpus: &[GpuDevice]) -> f64 {
        if gpus.is_empty() {
            return 0.0;
        }

        let best_gpu = gpus.iter()
            .max_by(|a, b| a.compute_capability_score.partial_cmp(&b.compute_capability_score).unwrap())
            .unwrap();

        let vram_score = best_gpu.vram_gb.map(|v| (v / 8.0).min(4.0)).unwrap_or(0.0); // Up to 4 points for VRAM
        let vendor_bonus = if best_gpu.vendor.to_lowercase().contains("nvidia") { 3.0 } else { 1.0 };
        let cuda_bonus = if best_gpu.cuda_support { 2.0 } else { 0.0 };
        
        (vram_score + vendor_bonus + cuda_bonus).min(10.0)
    }

    /// Calculate graphics performance score
    fn calculate_graphics_score(gpus: &[GpuDevice]) -> f64 {
        if gpus.is_empty() {
            return 0.0;
        }

        let best_gpu = gpus.iter()
            .max_by(|a, b| a.graphics_score.partial_cmp(&b.graphics_score).unwrap())
            .unwrap();

        best_gpu.graphics_score
    }

    /// Calculate compute performance score
    fn calculate_compute_score(gpus: &[GpuDevice]) -> f64 {
        if gpus.is_empty() {
            return 0.0;
        }

        let best_gpu = gpus.iter()
            .max_by(|a, b| a.compute_capability_score.partial_cmp(&b.compute_capability_score).unwrap())
            .unwrap();

        best_gpu.compute_capability_score
    }
}

/// Individual GPU device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// GPU name
    pub name: String,
    /// GPU vendor
    pub vendor: String,
    /// VRAM in GB
    pub vram_gb: Option<f64>,
    /// Compute capability
    pub compute_capability: Option<String>,
    /// CUDA support
    pub cuda_support: bool,
    /// OpenCL support
    pub opencl_support: bool,
    /// Graphics performance score (0-10)
    pub graphics_score: f64,
    /// Compute capability score (0-10)
    pub compute_capability_score: f64,
    /// Power efficiency score (0-10)
    pub power_efficiency: f64,
}

impl GpuDevice {
    /// Create GPU device from GPU info
    pub fn from_gpu_info(gpu_info: &GpuInfo) -> Self {
        let vram_gb = gpu_info.vram_size.map(|v| v as f64 / 1024.0);
        
        let graphics_score = Self::calculate_graphics_score_for_device(gpu_info);
        let compute_score = Self::calculate_compute_score_for_device(gpu_info);
        let power_efficiency = Self::calculate_power_efficiency(gpu_info);
        
        Self {
            name: gpu_info.name.clone(),
            vendor: gpu_info.vendor.clone(),
            vram_gb,
            compute_capability: gpu_info.compute_capability.clone(),
            cuda_support: gpu_info.cuda_support,
            opencl_support: gpu_info.opencl_support,
            graphics_score,
            compute_capability_score: compute_score,
            power_efficiency,
        }
    }

    /// Calculate graphics score for individual device
    fn calculate_graphics_score_for_device(gpu_info: &GpuInfo) -> f64 {
        let vram_score = gpu_info.vram_size.map(|v| (v as f64 / 1024.0 / 4.0).min(3.0)).unwrap_or(0.0);
        let vendor_score = match gpu_info.vendor.to_lowercase().as_str() {
            vendor if vendor.contains("nvidia") => 4.0,
            vendor if vendor.contains("amd") => 3.5,
            vendor if vendor.contains("intel") => 2.0,
            _ => 1.0,
        };
        let modern_bonus = if gpu_info.name.contains("RTX") || gpu_info.name.contains("RX") { 2.0 } else { 0.0 };
        
        (vram_score + vendor_score + modern_bonus).min(10.0)
    }

    /// Calculate compute score for individual device
    fn calculate_compute_score_for_device(gpu_info: &GpuInfo) -> f64 {
        let mut score = 0.0;
        
        // VRAM contribution
        if let Some(vram) = gpu_info.vram_size {
            score += (vram as f64 / 1024.0 / 8.0).min(4.0); // Up to 4 points for VRAM
        }
        
        // CUDA support bonus
        if gpu_info.cuda_support {
            score += 3.0;
        }
        
        // Vendor scoring
        score += match gpu_info.vendor.to_lowercase().as_str() {
            vendor if vendor.contains("nvidia") => 2.0,
            vendor if vendor.contains("amd") => 1.5,
            vendor if vendor.contains("intel") => 0.5,
            _ => 0.0,
        };
        
        // Modern architecture bonus
        if gpu_info.name.contains("RTX") || gpu_info.name.contains("A100") || gpu_info.name.contains("H100") {
            score += 1.0;
        }
        
        score.min(10.0)
    }

    /// Calculate power efficiency score
    fn calculate_power_efficiency(gpu_info: &GpuInfo) -> f64 {
        // This would typically require power consumption data
        // For now, use heuristics based on generation and vendor
        match gpu_info.vendor.to_lowercase().as_str() {
            vendor if vendor.contains("nvidia") && gpu_info.name.contains("RTX") => 8.0,
            vendor if vendor.contains("amd") && gpu_info.name.contains("RX") => 7.0,
            vendor if vendor.contains("intel") => 6.0,
            _ => 5.0,
        }
    }
}

/// Memory-specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCapabilities {
    /// Total RAM in GB
    pub total_ram_gb: f64,
    /// Available RAM in GB
    pub available_ram_gb: f64,
    /// Memory type
    pub memory_type: Option<String>,
    /// Memory speed in MHz
    pub memory_speed_mhz: Option<u64>,
    /// Memory bandwidth score (0-10)
    pub bandwidth_score: f64,
    /// Memory capacity score (0-10)
    pub capacity_score: f64,
    /// High-bandwidth memory support
    pub high_bandwidth: bool,
    /// ECC memory support
    pub ecc_support: bool,
}

impl MemoryCapabilities {
    /// Create memory capabilities from memory info
    pub fn from_memory_info(memory_info: &MemoryInfo) -> Self {
        let total_ram_gb = memory_info.total_ram as f64 / 1024.0;
        let available_ram_gb = memory_info.available_ram as f64 / 1024.0;
        
        let capacity_score = Self::calculate_capacity_score(total_ram_gb);
        let bandwidth_score = Self::calculate_bandwidth_score(memory_info);
        
        let high_bandwidth = memory_info.memory_speed.map(|s| s >= 3200).unwrap_or(false);
        let ecc_support = memory_info.memory_type.as_ref()
            .map(|t| t.to_lowercase().contains("ecc"))
            .unwrap_or(false);
        
        Self {
            total_ram_gb,
            available_ram_gb,
            memory_type: memory_info.memory_type.clone(),
            memory_speed_mhz: memory_info.memory_speed,
            bandwidth_score,
            capacity_score,
            high_bandwidth,
            ecc_support,
        }
    }

    /// Calculate memory capacity score
    fn calculate_capacity_score(total_gb: f64) -> f64 {
        match total_gb {
            gb if gb >= 128.0 => 10.0,
            gb if gb >= 64.0 => 9.0,
            gb if gb >= 32.0 => 8.0,
            gb if gb >= 16.0 => 7.0,
            gb if gb >= 8.0 => 5.0,
            gb if gb >= 4.0 => 3.0,
            _ => 1.0,
        }
    }

    /// Calculate memory bandwidth score
    fn calculate_bandwidth_score(memory_info: &MemoryInfo) -> f64 {
        let speed_score = memory_info.memory_speed
            .map(|speed| (speed as f64 / 800.0).min(5.0)) // Up to 5 points for speed
            .unwrap_or(2.0);
        
        let type_bonus = memory_info.memory_type.as_ref()
            .map(|t| match t.to_lowercase().as_str() {
                t if t.contains("ddr5") => 3.0,
                t if t.contains("ddr4") => 2.0,
                t if t.contains("ddr3") => 1.0,
                _ => 0.5,
            })
            .unwrap_or(1.0);
        
        (speed_score + type_bonus).min(10.0)
    }
}

/// Storage-specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCapabilities {
    /// Storage devices
    pub devices: Vec<StorageDevice>,
    /// Total capacity in GB
    pub total_capacity_gb: f64,
    /// Available capacity in GB
    pub available_capacity_gb: f64,
    /// Fastest device performance score (0-10)
    pub performance_score: f64,
    /// NVMe support
    pub nvme_support: bool,
    /// SSD ratio (percentage of storage that is SSD)
    pub ssd_ratio: f64,
}

impl StorageCapabilities {
    /// Create storage capabilities from storage info
    pub fn from_storage_info(storage_info: &[StorageInfo]) -> Self {
        let devices: Vec<StorageDevice> = storage_info.iter()
            .map(StorageDevice::from_storage_info)
            .collect();
        
        let total_capacity_gb = devices.iter()
            .map(|d| d.total_capacity_gb)
            .sum();
        
        let available_capacity_gb = devices.iter()
            .map(|d| d.available_capacity_gb)
            .sum();
        
        let performance_score = devices.iter()
            .map(|d| d.performance_score)
            .fold(0.0, f64::max);
        
        let nvme_support = devices.iter()
            .any(|d| d.storage_type.to_lowercase().contains("nvme"));
        
        let ssd_capacity: f64 = devices.iter()
            .filter(|d| d.is_ssd())
            .map(|d| d.total_capacity_gb)
            .sum();
        
        let ssd_ratio = if total_capacity_gb > 0.0 {
            ssd_capacity / total_capacity_gb * 100.0
        } else {
            0.0
        };
        
        Self {
            devices,
            total_capacity_gb,
            available_capacity_gb,
            performance_score,
            nvme_support,
            ssd_ratio,
        }
    }
}

/// Individual storage device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDevice {
    /// Device name
    pub name: String,
    /// Storage type
    pub storage_type: String,
    /// Total capacity in GB
    pub total_capacity_gb: f64,
    /// Available capacity in GB
    pub available_capacity_gb: f64,
    /// Read speed in MB/s
    pub read_speed_mbps: Option<u64>,
    /// Write speed in MB/s
    pub write_speed_mbps: Option<u64>,
    /// Performance score (0-10)
    pub performance_score: f64,
    /// Reliability score (0-10)
    pub reliability_score: f64,
}

impl StorageDevice {
    /// Create storage device from storage info
    pub fn from_storage_info(storage_info: &StorageInfo) -> Self {
        let performance_score = Self::calculate_performance_score(storage_info);
        let reliability_score = Self::calculate_reliability_score(storage_info);
        
        Self {
            name: storage_info.name.clone(),
            storage_type: storage_info.storage_type.clone(),
            total_capacity_gb: storage_info.total_capacity as f64,
            available_capacity_gb: storage_info.available_capacity as f64,
            read_speed_mbps: storage_info.read_speed,
            write_speed_mbps: storage_info.write_speed,
            performance_score,
            reliability_score,
        }
    }

    /// Check if this is an SSD
    pub fn is_ssd(&self) -> bool {
        let storage_type = self.storage_type.to_lowercase();
        storage_type.contains("ssd") || storage_type.contains("nvme")
    }

    /// Calculate performance score
    fn calculate_performance_score(storage_info: &StorageInfo) -> f64 {
        let type_score = match storage_info.storage_type.to_lowercase().as_str() {
            t if t.contains("nvme") => 8.0,
            t if t.contains("ssd") => 6.0,
            t if t.contains("hdd") => 2.0,
            _ => 3.0,
        };
        
        let speed_bonus = storage_info.read_speed
            .map(|s| (s as f64 / 1000.0).min(2.0)) // Up to 2 points for read speed
            .unwrap_or(0.0);
        
        (type_score + speed_bonus).min(10.0)
    }

    /// Calculate reliability score
    fn calculate_reliability_score(storage_info: &StorageInfo) -> f64 {
        match storage_info.storage_type.to_lowercase().as_str() {
            t if t.contains("nvme") => 9.0,
            t if t.contains("ssd") => 8.0,
            t if t.contains("hdd") => 6.0,
            _ => 5.0,
        }
    }
}

/// Network-specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Internet connectivity
    pub internet_connected: bool,
    /// Estimated bandwidth in Mbps
    pub estimated_bandwidth_mbps: Option<u64>,
    /// Network performance score (0-10)
    pub performance_score: f64,
    /// High-speed network support
    pub high_speed_support: bool,
    /// Latency estimate in milliseconds
    pub estimated_latency_ms: Option<f64>,
}

impl NetworkCapabilities {
    /// Create network capabilities from network info
    pub fn from_network_info(network_info: &NetworkInfo) -> Self {
        let interfaces: Vec<NetworkInterface> = network_info.interfaces.iter()
            .map(NetworkInterface::from_network_interface_info)
            .collect();
        
        let performance_score = Self::calculate_network_performance(&interfaces, network_info.estimated_bandwidth);
        let high_speed_support = network_info.estimated_bandwidth
            .map(|b| b >= 1000)
            .unwrap_or(false);
        
        Self {
            interfaces,
            internet_connected: network_info.internet_connected,
            estimated_bandwidth_mbps: network_info.estimated_bandwidth,
            performance_score,
            high_speed_support,
            estimated_latency_ms: None, // Would need actual measurements
        }
    }

    /// Calculate network performance score
    fn calculate_network_performance(interfaces: &[NetworkInterface], bandwidth: Option<u64>) -> f64 {
        let bandwidth_score = bandwidth
            .map(|b| (b as f64 / 100.0).min(5.0)) // Up to 5 points for bandwidth
            .unwrap_or(1.0);
        
        let interface_score = if interfaces.iter().any(|i| i.is_high_speed()) {
            3.0
        } else if interfaces.iter().any(|i| i.is_ethernet()) {
            2.0
        } else {
            1.0
        };
        
        let connection_bonus = if interfaces.iter().any(|i| !i.ip_addresses.is_empty()) {
            2.0
        } else {
            0.0
        };
        
        (bandwidth_score + interface_score + connection_bonus).min(10.0)
    }
}

/// Network interface capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// Interface type
    pub interface_type: String,
    /// MAC address
    pub mac_address: String,
    /// IP addresses
    pub ip_addresses: Vec<String>,
    /// Connection speed in Mbps
    pub speed_mbps: Option<u64>,
    /// Interface quality score (0-10)
    pub quality_score: f64,
}

impl NetworkInterface {
    /// Create network interface from network interface info
    pub fn from_network_interface_info(interface_info: &crate::types::NetworkInterface) -> Self {
        let quality_score = Self::calculate_quality_score(interface_info);
        
        Self {
            name: interface_info.name.clone(),
            interface_type: interface_info.interface_type.clone(),
            mac_address: interface_info.mac_address.clone(),
            ip_addresses: interface_info.ip_addresses.clone(),
            speed_mbps: interface_info.speed,
            quality_score,
        }
    }

    /// Check if this is a high-speed interface
    pub fn is_high_speed(&self) -> bool {
        self.speed_mbps.map(|s| s >= 1000).unwrap_or(false)
    }

    /// Check if this is an Ethernet interface
    pub fn is_ethernet(&self) -> bool {
        self.interface_type.to_lowercase().contains("ethernet")
    }

    /// Calculate interface quality score
    fn calculate_quality_score(interface_info: &crate::types::NetworkInterface) -> f64 {
        let type_score = match interface_info.interface_type.to_lowercase().as_str() {
            t if t.contains("ethernet") => 4.0,
            t if t.contains("wifi") => 3.0,
            t if t.contains("wireless") => 3.0,
            _ => 2.0,
        };
        
        let speed_score = interface_info.speed
            .map(|s| (s as f64 / 500.0).min(3.0)) // Up to 3 points for speed
            .unwrap_or(1.0);
        
        let connection_bonus = if !interface_info.ip_addresses.is_empty() {
            3.0
        } else {
            0.0
        };
        
        (type_score + speed_score + connection_bonus).min(10.0)
    }
}

/// Overall capability scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityScores {
    /// CPU score (0-10)
    pub cpu_score: f64,
    /// GPU score (0-10)
    pub gpu_score: f64,
    /// NPU score (0-10)
    pub npu_score: Option<f64>,
    /// TPU score (0-10)
    pub tpu_score: Option<f64>,
    /// FPGA score (0-10)
    pub fpga_score: Option<f64>,
    /// ARM optimization score (0-10)
    pub arm_optimization_score: Option<f64>,
    /// Memory score (0-10)
    pub memory_score: f64,
    /// Storage score (0-10)
    pub storage_score: f64,
    /// Network score (0-10)
    pub network_score: f64,
    /// Overall system score (0-10)
    pub overall_score: f64,
}

impl CapabilityScores {
    /// Calculate scores from individual capabilities
    pub fn calculate(
        cpu: &CpuCapabilities,
        gpu: &GpuCapabilities,
        memory: &MemoryCapabilities,
        storage: &StorageCapabilities,
        network: &NetworkCapabilities,
    ) -> Self {
        let cpu_score = (cpu.thread_performance + cpu.multicore_efficiency) / 2.0;
        let gpu_score = (gpu.ai_acceleration_score + gpu.compute_score) / 2.0;
        let memory_score = (memory.capacity_score + memory.bandwidth_score) / 2.0;
        let storage_score = storage.performance_score;
        let network_score = network.performance_score;
        
        // AI accelerator scores - set to None for now until hardware-query integration
        let npu_score = None; // TODO: Implement with hardware-query
        let tpu_score = None; // TODO: Implement with hardware-query  
        let fpga_score = None; // TODO: Implement with hardware-query
        let arm_optimization_score = None; // TODO: Implement with hardware-query
        
        let overall_score = (cpu_score + gpu_score + memory_score + storage_score + network_score) / 5.0;
        
        Self {
            cpu_score,
            gpu_score,
            npu_score,
            tpu_score,
            fpga_score,
            arm_optimization_score,
            memory_score,
            storage_score,
            network_score,
            overall_score,
        }
    }
}

/// Capability analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMetadata {
    /// Analysis version
    pub analysis_version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// System fingerprint for caching
    pub system_fingerprint: String,
}

/// System features that can be checked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemFeature {
    /// CUDA compute support
    CudaCompute,
    /// OpenCL compute support
    OpenCLCompute,
    /// AVX instruction set support
    AVXInstructions,
    /// NVMe storage support
    NVMeStorage,
    /// High-bandwidth memory
    HighBandwidthMemory,
    /// Virtualization support
    VirtualizationSupport,
    /// High-speed network (1Gbps+)
    HighSpeedNetwork,
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CpuInfo, MemoryInfo, StorageInfo, NetworkInfo, NetworkInterface};

    #[test]
    fn test_cpu_capability_creation() {
        let cpu_info = CpuInfo {
            brand: "Intel Core i7-8700K".to_string(),
            physical_cores: 6,
            logical_cores: 12,
            base_frequency: 3700,
            max_frequency: Some(4700),
            cache_size: Some(12288),
            architecture: "x86_64".to_string(),
        };

        let cpu_capability = CpuCapabilities::from_cpu_info(&cpu_info);
        
        assert_eq!(cpu_capability.architecture, "x86_64");
        assert_eq!(cpu_capability.physical_cores, 6);
        assert_eq!(cpu_capability.logical_cores, 12);
        assert_eq!(cpu_capability.base_frequency_mhz, 3700);
        assert!(cpu_capability.thread_performance > 0.0);
        assert!(cpu_capability.multicore_efficiency > 0.0);
    }

    #[test]
    fn test_memory_capability_creation() {
        let memory_info = MemoryInfo {
            total_ram: 16_000_000_000, // 16GB in bytes
            available_ram: 12_000_000_000, // 12GB in bytes
            memory_speed: Some(3200),
            memory_type: Some("DDR4".to_string()),
        };

        let memory_capability = MemoryCapabilities::from_memory_info(&memory_info);
        
        assert!(memory_capability.total_ram_gb > 0.0);
        assert!(memory_capability.available_ram_gb > 0.0);
        assert_eq!(memory_capability.memory_type, Some("DDR4".to_string()));
        assert!(memory_capability.bandwidth_score > 0.0);
    }

    #[test]
    fn test_storage_capability_creation() {
        let storage_info = vec![StorageInfo {
            name: "Samsung SSD 970 EVO".to_string(),
            storage_type: "NVMe SSD".to_string(),
            total_capacity: 500,
            available_capacity: 400,
            read_speed: Some(3500),
            write_speed: Some(3200),
        }];

        let storage_capability = StorageCapabilities::from_storage_info(&storage_info);
        
        assert!(storage_capability.total_capacity_gb > 0.0);
        assert!(storage_capability.available_capacity_gb > 0.0);
        assert!(storage_capability.nvme_support);
        assert!(storage_capability.performance_score > 0.0);
    }

    #[test]
    fn test_network_capability_creation() {
        let network_info = NetworkInfo {
            interfaces: vec![
                NetworkInterface {
                    name: "Ethernet".to_string(),
                    interface_type: "Ethernet".to_string(),
                    mac_address: "00:11:22:33:44:55".to_string(),
                    ip_addresses: vec!["192.168.1.100".to_string()],
                    speed: Some(1000),
                }
            ],
            internet_connected: true,
            estimated_bandwidth: Some(1000),
        };

        let network_capability = NetworkCapabilities::from_network_info(&network_info);
        
        assert!(network_capability.internet_connected);
        assert!(network_capability.estimated_bandwidth_mbps.unwrap_or(0) > 0);
        assert!(network_capability.performance_score > 0.0);
    }

    #[test]
    fn test_capability_profile_creation() {
        let cpu_info = CpuInfo {
            brand: "Intel Core i5-8400".to_string(),
            physical_cores: 6,
            logical_cores: 6,
            base_frequency: 2800,
            max_frequency: Some(4000),
            cache_size: Some(9216),
            architecture: "x86_64".to_string(),
        };

        let memory_info = MemoryInfo {
            total_ram: 8_000_000_000, // 8GB in bytes
            available_ram: 6_000_000_000, // 6GB in bytes
            memory_speed: Some(2666),
            memory_type: Some("DDR4".to_string()),
        };

        let storage_info = vec![StorageInfo {
            name: "Generic SSD".to_string(),
            storage_type: "SSD".to_string(),
            total_capacity: 256,
            available_capacity: 200,
            read_speed: Some(500),
            write_speed: Some(450),
        }];

        let network_info = NetworkInfo {
            interfaces: vec![],
            internet_connected: false,
            estimated_bandwidth: None,
        };

        let gpu_info = vec![];

        let system_info = crate::types::SystemInfo {
            os_name: "Windows".to_string(),
            os_version: "11".to_string(),
            cpu_info,
            gpu_info,
            memory_info,
            storage_info,
            network_info,
            npu_info: vec![], // No NPUs in test
            tpu_info: vec![], // No TPUs in test  
            fpga_info: vec![], // No FPGAs in test
            arm_info: None, // Not ARM system in test
        };

        let capability_profile = CapabilityProfile::from_system_info(&system_info);
        
        assert!(capability_profile.scores.overall_score >= 0.0);
        assert!(capability_profile.scores.overall_score <= 10.0);
    }

    #[test]
    fn test_ai_capability_levels() {
        // Test low-end system
        let low_end_cpu = CpuInfo {
            brand: "Intel Celeron".to_string(),
            physical_cores: 2,
            logical_cores: 2,
            base_frequency: 1600,
            max_frequency: Some(2400),
            cache_size: Some(2048),
            architecture: "x86_64".to_string(),
        };

        let low_capability = CpuCapabilities::from_cpu_info(&low_end_cpu);
        let low_ai_level = low_capability.ai_capability_level();
        
        // Should be Very Low or Low for a Celeron
        assert!(matches!(low_ai_level, CapabilityLevel::VeryLow | CapabilityLevel::Low));

        // Test high-end system
        let high_end_cpu = CpuInfo {
            brand: "Intel Core i9-12900K".to_string(),
            physical_cores: 16,
            logical_cores: 24,
            base_frequency: 3200,
            max_frequency: Some(5200),
            cache_size: Some(30720),
            architecture: "x86_64".to_string(),
        };

        let high_capability = CpuCapabilities::from_cpu_info(&high_end_cpu);
        let high_ai_level = high_capability.ai_capability_level();
        
        // Should be High or Very High for an i9
        assert!(matches!(high_ai_level, CapabilityLevel::High | CapabilityLevel::VeryHigh));
    }
}
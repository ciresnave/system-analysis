//! Utility functions and helpers for system analysis.

use crate::error::Result;
use crate::resources::{ResourceType, ResourceAmount, CapabilityLevel};
use crate::types::{SystemProfile, WorkloadRequirements};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Benchmark runner for performance testing
pub struct BenchmarkRunner {
    timeout: Duration,
    #[allow(dead_code)]
    iterations: u32,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(timeout: Duration, iterations: u32) -> Self {
        Self { timeout, iterations }
    }

    /// Run CPU benchmark
    pub fn run_cpu_benchmark(&self) -> Result<BenchmarkResult> {
        let start = Instant::now();
        let mut operations = 0u64;

        while start.elapsed() < self.timeout {
            // Simple CPU-intensive operation
            for _ in 0..10000 {
                operations += 1;
                let _ = (operations as f64).sqrt();
            }
        }

        Ok(BenchmarkResult {
            benchmark_type: BenchmarkType::CPU,
            duration: start.elapsed(),
            operations,
            score: self.calculate_cpu_score(operations, start.elapsed()),
        })
    }

    /// Run memory benchmark
    pub fn run_memory_benchmark(&self) -> Result<BenchmarkResult> {
        let start = Instant::now();
        let mut operations = 0u64;
        let mut data = vec![0u8; 1024 * 1024]; // 1MB buffer

        while start.elapsed() < self.timeout {
            // Memory-intensive operations
            for chunk in data.chunks_mut(1024) {
                chunk.fill(operations as u8);
                operations += 1;
            }
        }

        Ok(BenchmarkResult {
            benchmark_type: BenchmarkType::Memory,
            duration: start.elapsed(),
            operations,
            score: self.calculate_memory_score(operations, start.elapsed()),
        })
    }

    /// Calculate CPU score from benchmark results
    fn calculate_cpu_score(&self, operations: u64, duration: Duration) -> f64 {
        let ops_per_sec = operations as f64 / duration.as_secs_f64();
        // Normalize to 0-10 scale (this would be calibrated against known hardware)
        (ops_per_sec / 100_000.0).min(10.0)
    }

    /// Calculate memory score from benchmark results
    fn calculate_memory_score(&self, operations: u64, duration: Duration) -> f64 {
        let ops_per_sec = operations as f64 / duration.as_secs_f64();
        // Normalize to 0-10 scale
        (ops_per_sec / 50_000.0).min(10.0)
    }
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_type: BenchmarkType,
    pub duration: Duration,
    pub operations: u64,
    pub score: f64,
}

/// Types of benchmarks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BenchmarkType {
    CPU,
    Memory,
    Storage,
    Network,
    GPU,
}

/// System fingerprinting utilities
pub struct SystemFingerprinter;

impl SystemFingerprinter {
    /// Generate a unique fingerprint for the system
    pub fn generate_fingerprint() -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Include system-specific information
        std::env::consts::OS.hash(&mut hasher);
        std::env::consts::ARCH.hash(&mut hasher);
        
        // Include hostname if available
        if let Ok(hostname) = std::env::var("COMPUTERNAME") {
            hostname.hash(&mut hasher);
        } else if let Ok(hostname) = std::env::var("HOSTNAME") {
            hostname.hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Check if system configuration has changed
    pub fn has_system_changed(last_fingerprint: &str) -> bool {
        let current_fingerprint = Self::generate_fingerprint();
        current_fingerprint != last_fingerprint
    }
}

/// Performance optimization suggestions
pub struct PerformanceOptimizer;

impl PerformanceOptimizer {
    /// Suggest optimizations for a specific workload
    pub fn suggest_optimizations(
        system_profile: &SystemProfile,
        _workload_requirements: &WorkloadRequirements,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // CPU optimizations
        if system_profile.cpu_score() < 7.0 {
            suggestions.push(OptimizationSuggestion {
                resource_type: ResourceType::CPU,
                suggestion: "Consider enabling CPU performance mode".to_string(),
                expected_improvement: "5-15% performance increase".to_string(),
                difficulty: OptimizationDifficulty::Easy,
            });
        }

        // Memory optimizations
        if system_profile.memory_score() < 6.0 {
            suggestions.push(OptimizationSuggestion {
                resource_type: ResourceType::Memory,
                suggestion: "Close unnecessary applications to free memory".to_string(),
                expected_improvement: "10-20% memory availability increase".to_string(),
                difficulty: OptimizationDifficulty::Easy,
            });
        }

        // Storage optimizations
        if system_profile.storage_score() < 5.0 {
            suggestions.push(OptimizationSuggestion {
                resource_type: ResourceType::Storage,
                suggestion: "Enable storage optimization and defragmentation".to_string(),
                expected_improvement: "10-30% I/O performance increase".to_string(),
                difficulty: OptimizationDifficulty::Medium,
            });
        }

        // GPU optimizations
        if system_profile.gpu_score() < 4.0 {
            suggestions.push(OptimizationSuggestion {
                resource_type: ResourceType::GPU,
                suggestion: "Update GPU drivers and enable GPU scheduling".to_string(),
                expected_improvement: "5-25% graphics performance increase".to_string(),
                difficulty: OptimizationDifficulty::Medium,
            });
        }

        suggestions
    }
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub resource_type: ResourceType,
    pub suggestion: String,
    pub expected_improvement: String,
    pub difficulty: OptimizationDifficulty,
}

/// Difficulty levels for optimizations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Resource comparison utilities
pub struct ResourceComparator;

impl ResourceComparator {
    /// Compare two systems and highlight differences
    pub fn compare_systems(
        system_a: &SystemProfile,
        system_b: &SystemProfile,
    ) -> SystemComparison {
        let cpu_diff = system_b.cpu_score() - system_a.cpu_score();
        let gpu_diff = system_b.gpu_score() - system_a.gpu_score();
        let memory_diff = system_b.memory_score() - system_a.memory_score();
        let storage_diff = system_b.storage_score() - system_a.storage_score();
        let network_diff = system_b.network_score() - system_a.network_score();

        let overall_diff = system_b.overall_score() - system_a.overall_score();

        SystemComparison {
            cpu_difference: cpu_diff,
            gpu_difference: gpu_diff,
            memory_difference: memory_diff,
            storage_difference: storage_diff,
            network_difference: network_diff,
            overall_difference: overall_diff,
            winner: if overall_diff > 0.0 {
                ComparisonResult::SystemB
            } else if overall_diff < 0.0 {
                ComparisonResult::SystemA
            } else {
                ComparisonResult::Tie
            },
        }
    }

    /// Compare resource amounts
    pub fn compare_resource_amounts(
        amount_a: &ResourceAmount,
        amount_b: &ResourceAmount,
    ) -> Option<f64> {
        match (amount_a, amount_b) {
            (ResourceAmount::Score(a), ResourceAmount::Score(b)) => Some(b - a),
            (ResourceAmount::Gigabytes(a), ResourceAmount::Gigabytes(b)) => Some(b - a),
            (ResourceAmount::Megahertz(a), ResourceAmount::Megahertz(b)) => Some(b - a),
            (ResourceAmount::Units(a), ResourceAmount::Units(b)) => Some(*b as f64 - *a as f64),
            (ResourceAmount::Percentage(a), ResourceAmount::Percentage(b)) => Some(b - a),
            (ResourceAmount::Level(a), ResourceAmount::Level(b)) => {
                let a_score: f64 = (*a).into();
                let b_score: f64 = (*b).into();
                Some(b_score - a_score)
            }
            _ => None, // Different types, cannot compare directly
        }
    }
}

/// System comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemComparison {
    pub cpu_difference: f64,
    pub gpu_difference: f64,
    pub memory_difference: f64,
    pub storage_difference: f64,
    pub network_difference: f64,
    pub overall_difference: f64,
    pub winner: ComparisonResult,
}

/// Comparison result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonResult {
    SystemA,
    SystemB,
    Tie,
}

/// Configuration validation utilities
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate workload requirements for consistency
    pub fn validate_workload_requirements(
        workload_requirements: &WorkloadRequirements,
    ) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        // Check for conflicting requirements
        let mut resource_types = std::collections::HashSet::new();
        for req in &workload_requirements.resource_requirements {
            if resource_types.contains(&req.resource_type) {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: format!("Duplicate resource requirement for {:?}", req.resource_type),
                    suggestion: "Merge duplicate requirements".to_string(),
                });
            }
            resource_types.insert(req.resource_type);
        }

        // Check for unrealistic requirements
        for req in &workload_requirements.resource_requirements {
            match &req.minimum {
                ResourceAmount::Gigabytes(gb) if *gb > 1000.0 => {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Warning,
                        message: format!("Very high {} requirement: {:.1}GB", req.resource_type, gb),
                        suggestion: "Verify this requirement is realistic".to_string(),
                    });
                }
                ResourceAmount::Score(score) if *score > 10.0 => {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Invalid score for {}: {:.1} (max 10.0)", req.resource_type, score),
                        suggestion: "Adjust score to be within 0-10 range".to_string(),
                    });
                }
                _ => {}
            }
        }

        // Validate workload-specific requirements
        if let Some(workload) = &workload_requirements.workload {
            if let Err(e) = workload.validate() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    message: format!("Workload validation failed: {e}"),
                    suggestion: "Fix workload configuration".to_string(),
                });
            }
        }

        Ok(issues)
    }

    /// Validate system profile for consistency
    pub fn validate_system_profile(system_profile: &SystemProfile) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        // Check for unrealistic scores
        let scores = [
            ("CPU", system_profile.cpu_score()),
            ("GPU", system_profile.gpu_score()),
            ("Memory", system_profile.memory_score()),
            ("Storage", system_profile.storage_score()),
            ("Network", system_profile.network_score()),
        ];

        for (name, score) in scores {
            if !(0.0..=10.0).contains(&score) {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    message: format!("Invalid {name} score: {score:.1} (should be 0-10)"),
                    suggestion: "Recalibrate scoring algorithm".to_string(),
                });
            }
        }

        // Check for inconsistencies in system info
        let memory_gb = system_profile.system_info.memory_info.total_ram as f64 / 1024.0;
        if memory_gb > 1000.0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: format!("Very high memory amount detected: {memory_gb:.1}GB"),
                suggestion: "Verify memory detection is accurate".to_string(),
            });
        }

        Ok(issues)
    }
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub message: String,
    pub suggestion: String,
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
}

/// Hardware detection utilities
pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect hardware changes since last analysis
    pub fn detect_hardware_changes(
        last_system_info: &crate::types::SystemInfo,
        current_system_info: &crate::types::SystemInfo,
    ) -> Vec<HardwareChange> {
        let mut changes = Vec::new();

        // Check CPU changes
        if last_system_info.cpu_info.brand != current_system_info.cpu_info.brand {
            changes.push(HardwareChange {
                component: HardwareComponent::CPU,
                change_type: HardwareChangeType::Replaced,
                details: format!("CPU changed from '{}' to '{}'", 
                    last_system_info.cpu_info.brand, 
                    current_system_info.cpu_info.brand),
            });
        }

        // Check memory changes
        if last_system_info.memory_info.total_ram != current_system_info.memory_info.total_ram {
            changes.push(HardwareChange {
                component: HardwareComponent::Memory,
                change_type: if current_system_info.memory_info.total_ram > last_system_info.memory_info.total_ram {
                    HardwareChangeType::Upgraded
                } else {
                    HardwareChangeType::Downgraded
                },
                details: format!("Memory changed from {}MB to {}MB", 
                    last_system_info.memory_info.total_ram, 
                    current_system_info.memory_info.total_ram),
            });
        }

        // Check GPU changes
        if last_system_info.gpu_info.len() != current_system_info.gpu_info.len() {
            changes.push(HardwareChange {
                component: HardwareComponent::GPU,
                change_type: if current_system_info.gpu_info.len() > last_system_info.gpu_info.len() {
                    HardwareChangeType::Added
                } else {
                    HardwareChangeType::Removed
                },
                details: format!("GPU count changed from {} to {}", 
                    last_system_info.gpu_info.len(), 
                    current_system_info.gpu_info.len()),
            });
        }

        // Check storage changes
        if last_system_info.storage_info.len() != current_system_info.storage_info.len() {
            changes.push(HardwareChange {
                component: HardwareComponent::Storage,
                change_type: if current_system_info.storage_info.len() > last_system_info.storage_info.len() {
                    HardwareChangeType::Added
                } else {
                    HardwareChangeType::Removed
                },
                details: format!("Storage device count changed from {} to {}", 
                    last_system_info.storage_info.len(), 
                    current_system_info.storage_info.len()),
            });
        }

        changes
    }
}

/// Hardware change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareChange {
    pub component: HardwareComponent,
    pub change_type: HardwareChangeType,
    pub details: String,
}

/// Hardware components
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HardwareComponent {
    CPU,
    GPU,
    Memory,
    Storage,
    Network,
}

/// Types of hardware changes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HardwareChangeType {
    Added,
    Removed,
    Replaced,
    Upgraded,
    Downgraded,
}

/// Performance trend analysis
pub struct TrendAnalyzer;

impl TrendAnalyzer {
    /// Analyze performance trends over time
    pub fn analyze_trends(
        historical_profiles: &[SystemProfile],
    ) -> PerformanceTrends {
        if historical_profiles.is_empty() {
            return PerformanceTrends::default();
        }

        let mut cpu_scores = Vec::new();
        let mut gpu_scores = Vec::new();
        let mut memory_scores = Vec::new();
        let mut storage_scores = Vec::new();
        let mut network_scores = Vec::new();

        for profile in historical_profiles {
            cpu_scores.push(profile.cpu_score());
            gpu_scores.push(profile.gpu_score());
            memory_scores.push(profile.memory_score());
            storage_scores.push(profile.storage_score());
            network_scores.push(profile.network_score());
        }

        PerformanceTrends {
            cpu_trend: Self::calculate_trend(&cpu_scores),
            gpu_trend: Self::calculate_trend(&gpu_scores),
            memory_trend: Self::calculate_trend(&memory_scores),
            storage_trend: Self::calculate_trend(&storage_scores),
            network_trend: Self::calculate_trend(&network_scores),
            overall_trend: Self::calculate_trend(&historical_profiles.iter()
                .map(|p| p.overall_score())
                .collect::<Vec<_>>()),
        }
    }

    /// Calculate trend for a series of values
    fn calculate_trend(values: &[f64]) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_half = &values[..values.len() / 2];
        let second_half = &values[values.len() / 2..];

        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let diff = second_avg - first_avg;

        if diff > 0.5 {
            TrendDirection::Improving
        } else if diff < -0.5 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        }
    }
}

/// Performance trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub cpu_trend: TrendDirection,
    pub gpu_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub storage_trend: TrendDirection,
    pub network_trend: TrendDirection,
    pub overall_trend: TrendDirection,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            cpu_trend: TrendDirection::Stable,
            gpu_trend: TrendDirection::Stable,
            memory_trend: TrendDirection::Stable,
            storage_trend: TrendDirection::Stable,
            network_trend: TrendDirection::Stable,
            overall_trend: TrendDirection::Stable,
        }
    }
}

/// Trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
}

/// Utility functions for common operations
pub mod common {
    use super::*;

    /// Convert capability level to human-readable string
    pub fn capability_level_to_string(level: CapabilityLevel) -> &'static str {
        match level {
            CapabilityLevel::VeryLow => "Very Low",
            CapabilityLevel::Low => "Low",
            CapabilityLevel::Medium => "Medium",
            CapabilityLevel::High => "High",
            CapabilityLevel::VeryHigh => "Very High",
            CapabilityLevel::Exceptional => "Exceptional",
        }
    }

    /// Format resource amount for display
    pub fn format_resource_amount(amount: &ResourceAmount) -> String {
        match amount {
            ResourceAmount::Level(level) => capability_level_to_string(*level).to_string(),
            ResourceAmount::Gigabytes(gb) => format!("{gb:.1} GB"),
            ResourceAmount::Megahertz(mhz) => format!("{mhz:.0} MHz"),
            ResourceAmount::Score(score) => format!("{score:.1}/10"),
            ResourceAmount::Units(units) => format!("{units} units"),
            ResourceAmount::Percentage(pct) => format!("{pct:.1}%"),
            ResourceAmount::Custom { value, unit } => format!("{value:.1} {unit}"),
        }
    }

    /// Calculate percentage difference between two values
    pub fn percentage_difference(old_value: f64, new_value: f64) -> f64 {
        if old_value == 0.0 {
            if new_value == 0.0 { 0.0 } else { 100.0 }
        } else {
            ((new_value - old_value) / old_value) * 100.0
        }
    }

    /// Round to specified decimal places
    pub fn round_to_decimal(value: f64, decimal_places: u32) -> f64 {
        let factor = 10.0_f64.powi(decimal_places as i32);
        (value * factor).round() / factor
    }
}

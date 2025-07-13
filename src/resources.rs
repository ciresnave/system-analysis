//! Resource management and requirement modeling.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Types of system resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU processing power
    CPU,
    /// GPU processing power
    GPU,
    /// System memory (RAM)
    Memory,
    /// Storage (disk space and I/O)
    Storage,
    /// Network bandwidth and latency
    Network,
    /// Custom resource type
    Custom(u32),
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::CPU => write!(f, "CPU"),
            ResourceType::GPU => write!(f, "GPU"),
            ResourceType::Memory => write!(f, "Memory"),
            ResourceType::Storage => write!(f, "Storage"),
            ResourceType::Network => write!(f, "Network"),
            ResourceType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Capability levels for resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CapabilityLevel {
    /// Very low capability
    VeryLow,
    /// Low capability
    Low,
    /// Medium capability
    Medium,
    /// High capability
    High,
    /// Very high capability
    VeryHigh,
    /// Exceptional capability
    Exceptional,
}

impl fmt::Display for CapabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapabilityLevel::VeryLow => write!(f, "Very Low"),
            CapabilityLevel::Low => write!(f, "Low"),
            CapabilityLevel::Medium => write!(f, "Medium"),
            CapabilityLevel::High => write!(f, "High"),
            CapabilityLevel::VeryHigh => write!(f, "Very High"),
            CapabilityLevel::Exceptional => write!(f, "Exceptional"),
        }
    }
}

impl From<f64> for CapabilityLevel {
    fn from(score: f64) -> Self {
        match score {
            score if score >= 9.0 => CapabilityLevel::Exceptional,
            score if score >= 7.5 => CapabilityLevel::VeryHigh,
            score if score >= 6.0 => CapabilityLevel::High,
            score if score >= 4.0 => CapabilityLevel::Medium,
            score if score >= 2.0 => CapabilityLevel::Low,
            _ => CapabilityLevel::VeryLow,
        }
    }
}

impl From<CapabilityLevel> for f64 {
    fn from(level: CapabilityLevel) -> Self {
        match level {
            CapabilityLevel::Exceptional => 10.0,
            CapabilityLevel::VeryHigh => 9.0,
            CapabilityLevel::High => 7.0,
            CapabilityLevel::Medium => 5.0,
            CapabilityLevel::Low => 3.0,
            CapabilityLevel::VeryLow => 1.0,
        }
    }
}

impl CapabilityLevel {
    /// Convert to numeric score
    pub fn to_numeric(&self) -> f64 {
        (*self).into()
    }
    
    /// Create from numeric score
    pub fn from_numeric(score: f64) -> Self {
        score.into()
    }
}

/// Resource requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    /// Type of resource
    pub resource_type: ResourceType,
    /// Minimum required level/amount
    pub minimum: ResourceAmount,
    /// Recommended level/amount
    pub recommended: Option<ResourceAmount>,
    /// Maximum useful level/amount
    pub maximum: Option<ResourceAmount>,
    /// Preferred vendor/technology
    pub preferred_vendor: Option<String>,
    /// Additional constraints
    pub constraints: Vec<ResourceConstraint>,
    /// Whether this requirement is critical
    pub is_critical: bool,
    /// Importance weight (0.0 to 1.0)
    pub weight: f64,
}

impl ResourceRequirement {
    /// Create a new resource requirement
    pub fn new(resource_type: ResourceType) -> Self {
        Self {
            resource_type,
            minimum: ResourceAmount::Level(CapabilityLevel::Medium),
            recommended: None,
            maximum: None,
            preferred_vendor: None,
            constraints: Vec::new(),
            is_critical: true,
            weight: 1.0,
        }
    }

    /// Set minimum amount in GB (for memory/storage)
    pub fn minimum_gb(mut self, gb: f64) -> Self {
        self.minimum = ResourceAmount::Gigabytes(gb);
        self
    }

    /// Set recommended amount in GB (for memory/storage)
    pub fn recommended_gb(mut self, gb: f64) -> Self {
        self.recommended = Some(ResourceAmount::Gigabytes(gb));
        self
    }

    /// Set maximum amount in GB (for memory/storage)
    pub fn maximum_gb(mut self, gb: f64) -> Self {
        self.maximum = Some(ResourceAmount::Gigabytes(gb));
        self
    }

    /// Set minimum capability level
    pub fn minimum_level(mut self, level: CapabilityLevel) -> Self {
        self.minimum = ResourceAmount::Level(level);
        self
    }

    /// Set recommended capability level
    pub fn recommended_level(mut self, level: CapabilityLevel) -> Self {
        self.recommended = Some(ResourceAmount::Level(level));
        self
    }

    /// Set maximum capability level
    pub fn maximum_level(mut self, level: CapabilityLevel) -> Self {
        self.maximum = Some(ResourceAmount::Level(level));
        self
    }

    /// Set minimum score (0-10)
    pub fn minimum_score(mut self, score: f64) -> Self {
        self.minimum = ResourceAmount::Score(score);
        self
    }

    /// Set recommended score (0-10)
    pub fn recommended_score(mut self, score: f64) -> Self {
        self.recommended = Some(ResourceAmount::Score(score));
        self
    }

    /// Set preferred vendor
    pub fn preferred_vendor(mut self, vendor: Option<impl Into<String>>) -> Self {
        self.preferred_vendor = vendor.map(|v| v.into());
        self
    }

    /// Add a constraint
    pub fn add_constraint(mut self, constraint: ResourceConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Mark as critical requirement
    pub fn critical(mut self) -> Self {
        self.is_critical = true;
        self
    }

    /// Mark as required
    pub fn required(self) -> Self {
        // Already critical by default if not set
        self
    }
    
    /// Set minimum GHz (for CPU)
    pub fn minimum_ghz(mut self, ghz: f64) -> Self {
        self.minimum = ResourceAmount::Megahertz(ghz * 1000.0);
        self
    }
    
    /// Set GPU memory requirement
    pub fn gpu_memory_gb(self, gb: f64) -> Self {
        // For GPU memory we can use a custom constraint
        self.add_constraint(ResourceConstraint::RequiredFeature(format!("GPU Memory: {} GB", gb)))
    }
    
    /// Set storage type
    pub fn storage_type(self, storage_type: String) -> Self {
        self.add_constraint(ResourceConstraint::RequiredFeature(storage_type))
    }
    
    /// Set minimum Mbps (for network)
    pub fn minimum_mbps(mut self, mbps: f64) -> Self {
        self.minimum = ResourceAmount::Custom { value: mbps, unit: "Mbps".to_string() };
        self
    }
    
    /// Set number of cores
    pub fn cores(mut self, cores: u32) -> Self {
        self.minimum = ResourceAmount::Units(cores);
        self
    }

    /// Check if a resource amount meets this requirement
    pub fn is_satisfied_by(&self, amount: &ResourceAmount) -> bool {
        amount >= &self.minimum
    }

    /// Get the gap between available and required resources
    pub fn get_gap(&self, available: &ResourceAmount) -> Option<ResourceGap> {
        if self.is_satisfied_by(available) {
            None
        } else {
            Some(ResourceGap {
                resource_type: self.resource_type,
                required: self.minimum.clone(),
                available: available.clone(),
                severity: if self.is_critical {
                    crate::types::RequirementSeverity::Critical
                } else {
                    crate::types::RequirementSeverity::High
                },
            })
        }
    }
}

/// Different ways to specify resource amounts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ResourceAmount {
    /// Capability level
    Level(CapabilityLevel),
    /// Amount in gigabytes
    Gigabytes(f64),
    /// Amount in megahertz
    Megahertz(f64),
    /// Score from 0 to 10
    Score(f64),
    /// Number of units
    Units(u32),
    /// Percentage (0.0 to 100.0)
    Percentage(f64),
    /// Custom amount with label
    Custom { value: f64, unit: String },
}

impl fmt::Display for ResourceAmount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceAmount::Level(level) => write!(f, "{}", level),
            ResourceAmount::Gigabytes(gb) => write!(f, "{:.1} GB", gb),
            ResourceAmount::Megahertz(mhz) => write!(f, "{:.0} MHz", mhz),
            ResourceAmount::Score(score) => write!(f, "{:.1}/10", score),
            ResourceAmount::Units(units) => write!(f, "{} units", units),
            ResourceAmount::Percentage(pct) => write!(f, "{:.1}%", pct),
            ResourceAmount::Custom { value, unit } => write!(f, "{:.1} {}", value, unit),
        }
    }
}

impl ResourceAmount {
    /// Create a gigabyte amount
    pub fn new_gb(gb: f64) -> Self {
        ResourceAmount::Gigabytes(gb)
    }
    
    /// Create a megabyte amount (converted to GB)
    pub fn new_mb(mb: f64) -> Self {
        ResourceAmount::Gigabytes(mb / 1024.0)
    }
    
    /// Create a megahertz amount
    pub fn new_mhz(mhz: f64) -> Self {
        ResourceAmount::Megahertz(mhz)
    }
    
    /// Create a gigahertz amount (converted to MHz)
    pub fn new_ghz(ghz: f64) -> Self {
        ResourceAmount::Megahertz(ghz * 1000.0)
    }
    
    /// Get value as gigabytes if possible
    pub fn as_gb(&self) -> Option<f64> {
        match self {
            ResourceAmount::Gigabytes(gb) => Some(*gb),
            _ => None,
        }
    }
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceConstraint {
    /// Minimum frequency requirement
    MinimumFrequency(f64),
    /// Maximum power consumption in watts
    MaxPowerConsumption(f64),
    /// Required feature support
    RequiredFeature(String),
    /// Vendor restriction
    VendorRestriction(Vec<String>),
    /// Architecture requirement
    ArchitectureRequirement(String),
    /// Temperature constraint
    MaxTemperature(f64),
    /// Custom constraint
    Custom { name: String, value: String },
}

/// Resource gap information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceGap {
    /// Resource type with gap
    pub resource_type: ResourceType,
    /// Required amount
    pub required: ResourceAmount,
    /// Available amount
    pub available: ResourceAmount,
    /// Severity of the gap
    pub severity: crate::types::RequirementSeverity,
}

/// Resource pool for tracking available resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    /// Available resources by type
    pub resources: std::collections::HashMap<ResourceType, ResourceAmount>,
    /// Resource utilization tracking
    pub utilization: std::collections::HashMap<ResourceType, f64>,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl ResourcePool {
    /// Create a new empty resource pool
    pub fn new() -> Self {
        Self {
            resources: std::collections::HashMap::new(),
            utilization: std::collections::HashMap::new(),
            last_updated: chrono::Utc::now(),
        }
    }

    /// Add or update a resource
    pub fn set_resource(&mut self, resource_type: ResourceType, amount: ResourceAmount) {
        self.resources.insert(resource_type, amount);
        self.last_updated = chrono::Utc::now();
    }

    /// Add a resource (alias for set_resource)
    pub fn add_resource(&mut self, resource_type: ResourceType, amount: ResourceAmount) {
        self.set_resource(resource_type, amount);
    }
    
    /// Get available amount for a resource type
    pub fn get_resource(&self, resource_type: &ResourceType) -> Option<&ResourceAmount> {
        self.resources.get(resource_type)
    }

    /// Set resource utilization
    pub fn set_utilization(&mut self, resource_type: ResourceType, utilization: f64) {
        self.utilization.insert(resource_type, utilization.clamp(0.0, 100.0));
        self.last_updated = chrono::Utc::now();
    }

    /// Get resource utilization
    pub fn get_utilization(&self, resource_type: &ResourceType) -> f64 {
        self.utilization.get(resource_type).copied().unwrap_or(0.0)
    }

    /// Check if a resource type is available
    pub fn has_resource(&self, resource_type: &ResourceType) -> bool {
        self.resources.contains_key(resource_type)
    }
    
    /// Calculate utilization for a specific resource
    pub fn calculate_utilization(&self, resource_type: &ResourceType, used: &ResourceAmount) -> f64 {
        if let Some(available) = self.get_resource(resource_type) {
            match (used, available) {
                (ResourceAmount::Gigabytes(used_gb), ResourceAmount::Gigabytes(avail_gb)) => {
                    (used_gb / avail_gb * 100.0).min(100.0)
                }
                _ => 0.0, // Default for unsupported combinations
            }
        } else {
            0.0
        }
    }
    
    /// Check if a specific amount is available
    pub fn is_available(&self, resource_type: &ResourceType, required: &ResourceAmount) -> bool {
        if let Some(available) = self.get_resource(resource_type) {
            match (required, available) {
                (ResourceAmount::Gigabytes(req_gb), ResourceAmount::Gigabytes(avail_gb)) => {
                    avail_gb >= req_gb
                }
                _ => false, // Conservative approach for unsupported combinations
            }
        } else {
            false
        }
    }

    /// Check if requirements can be satisfied
    pub fn can_satisfy(&self, requirements: &[ResourceRequirement]) -> Vec<ResourceGap> {
        let mut gaps = Vec::new();

        for req in requirements {
            if let Some(available) = self.get_resource(&req.resource_type) {
                if let Some(gap) = req.get_gap(available) {
                    gaps.push(gap);
                }
            } else {
                // Resource not available at all
                gaps.push(ResourceGap {
                    resource_type: req.resource_type,
                    required: req.minimum.clone(),
                    available: ResourceAmount::Score(0.0),
                    severity: if req.is_critical {
                        crate::types::RequirementSeverity::Critical
                    } else {
                        crate::types::RequirementSeverity::High
                    },
                });
            }
        }

        gaps
    }

    /// Calculate overall satisfaction score (0-10)
    pub fn satisfaction_score(&self, requirements: &[ResourceRequirement]) -> f64 {
        if requirements.is_empty() {
            return 10.0;
        }

        let mut total_weight = 0.0;
        let mut weighted_satisfaction = 0.0;

        for req in requirements {
            total_weight += req.weight;

            let satisfaction = if let Some(available) = self.get_resource(&req.resource_type) {
                self.calculate_satisfaction_ratio(&req.minimum, available)
            } else {
                0.0
            };

            weighted_satisfaction += satisfaction * req.weight;
        }

        if total_weight > 0.0 {
            (weighted_satisfaction / total_weight) * 10.0
        } else {
            10.0
        }
    }

    /// Calculate satisfaction ratio for a requirement
    fn calculate_satisfaction_ratio(&self, required: &ResourceAmount, available: &ResourceAmount) -> f64 {
        match (required, available) {
            (ResourceAmount::Score(req), ResourceAmount::Score(avail)) => (avail / req).min(1.0),
            (ResourceAmount::Gigabytes(req), ResourceAmount::Gigabytes(avail)) => (avail / req).min(1.0),
            (ResourceAmount::Level(req), ResourceAmount::Level(avail)) => {
                let req_score: f64 = (*req).into();
                let avail_score: f64 = (*avail).into();
                (avail_score / req_score).min(1.0)
            }
            (ResourceAmount::Units(req), ResourceAmount::Units(avail)) => {
                (*avail as f64 / *req as f64).min(1.0)
            }
            (ResourceAmount::Percentage(req), ResourceAmount::Percentage(avail)) => {
                (avail / req).min(1.0)
            }
            _ => 0.5, // Different types, assume partial satisfaction
        }
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_level_conversion() {
        assert_eq!(CapabilityLevel::VeryLow.to_numeric(), 1.0);
        assert_eq!(CapabilityLevel::Low.to_numeric(), 3.0);
        assert_eq!(CapabilityLevel::Medium.to_numeric(), 5.0);
        assert_eq!(CapabilityLevel::High.to_numeric(), 7.0);
        assert_eq!(CapabilityLevel::VeryHigh.to_numeric(), 9.0);
        assert_eq!(CapabilityLevel::Exceptional.to_numeric(), 10.0);

        assert_eq!(CapabilityLevel::from_numeric(1.0), CapabilityLevel::VeryLow);
        assert_eq!(CapabilityLevel::from_numeric(5.0), CapabilityLevel::Medium);
        assert_eq!(CapabilityLevel::from_numeric(10.0), CapabilityLevel::Exceptional);
        assert_eq!(CapabilityLevel::from_numeric(6.5), CapabilityLevel::High);
    }

    #[test]
    fn test_resource_amount_creation() {
        let amount_gb = ResourceAmount::new_gb(16.0);
        assert_eq!(amount_gb.as_gb(), Some(16.0));

        let amount_mb = ResourceAmount::new_mb(1024.0);
        assert_eq!(amount_mb.as_gb(), Some(1.0));

        let amount_mhz = ResourceAmount::new_mhz(3200.0);
        let amount_ghz = ResourceAmount::new_ghz(2.5);

        // Basic test that they were created correctly
        assert!(matches!(amount_mhz, ResourceAmount::Megahertz(_)));
        assert!(matches!(amount_ghz, ResourceAmount::Megahertz(_)));
    }

    #[test]
    fn test_resource_requirement_basic() {
        let requirement = ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(8.0)
            .recommended_gb(16.0);

        assert_eq!(requirement.resource_type, ResourceType::Memory);
        assert_eq!(requirement.minimum, ResourceAmount::Gigabytes(8.0));
        assert!(requirement.recommended.is_some());
        assert!(requirement.is_critical);
    }

    #[test]
    fn test_resource_pool_basic() {
        let mut pool = ResourcePool::new();
        
        let memory_resource = ResourceAmount::new_gb(32.0);
        pool.add_resource(ResourceType::Memory, memory_resource);

        assert!(pool.has_resource(&ResourceType::Memory));
        assert!(!pool.has_resource(&ResourceType::GPU));

        let retrieved = pool.get_resource(&ResourceType::Memory);
        assert_eq!(retrieved.unwrap().as_gb(), Some(32.0));
    }
}
//! Feedback system for real-time monitoring integration
//! 
//! This module provides hooks and interfaces for external monitoring systems
//! to provide feedback about actual performance, allowing the system analysis
//! to improve its predictions over time.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use crate::error::SystemAnalysisError;
use crate::types::SystemProfile;

/// Performance feedback from external monitoring systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    /// Unique identifier for this feedback session
    pub session_id: String,
    /// Timestamp when feedback was recorded
    pub timestamp: u64,
    /// System profile at time of measurement
    pub system_profile: SystemProfile,
    /// Workload identifier that was running
    pub workload_id: String,
    /// Actual performance metrics
    pub metrics: PerformanceMetrics,
    /// Resource utilization during execution
    pub resource_utilization: ResourceUtilization,
    /// Any errors or warnings encountered
    pub issues: Vec<PerformanceIssue>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Actual performance metrics measured during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Inference latency in milliseconds
    pub latency_ms: Option<f64>,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: Option<f64>,
    /// Memory peak usage in GB
    pub peak_memory_gb: Option<f64>,
    /// Average memory usage in GB
    pub avg_memory_gb: Option<f64>,
    /// CPU utilization percentage (0-100)
    pub cpu_utilization_percent: Option<f64>,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: Option<f64>,
    /// Power consumption in watts
    pub power_consumption_watts: Option<f64>,
    /// Thermal measurements
    pub thermal: Option<ThermalMetrics>,
    /// Quality metrics (accuracy, loss, etc.)
    pub quality: Option<QualityMetrics>,
}

/// Thermal performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// CPU temperature in Celsius
    pub cpu_temp_celsius: Option<f64>,
    /// GPU temperature in Celsius
    pub gpu_temp_celsius: Option<f64>,
    /// Whether thermal throttling occurred
    pub thermal_throttling: bool,
}

/// Quality metrics for ML workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Model accuracy (0-1)
    pub accuracy: Option<f64>,
    /// Training loss
    pub loss: Option<f64>,
    /// Inference confidence scores
    pub confidence_scores: Vec<f64>,
    /// Custom quality metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Resource utilization measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization breakdown
    pub cpu: CpuUtilization,
    /// GPU utilization breakdown
    pub gpu: Vec<GpuUtilization>,
    /// Memory utilization
    pub memory: MemoryUtilization,
    /// Storage I/O metrics
    pub storage: StorageUtilization,
    /// Network utilization
    pub network: NetworkUtilization,
}

/// CPU utilization details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUtilization {
    /// Overall CPU usage percentage
    pub overall_percent: f64,
    /// Per-core usage percentages
    pub per_core_percent: Vec<f64>,
    /// CPU frequency during execution
    pub frequency_mhz: Option<f64>,
    /// CPU power draw
    pub power_watts: Option<f64>,
}

/// GPU utilization details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUtilization {
    /// GPU index
    pub gpu_index: u32,
    /// GPU usage percentage
    pub utilization_percent: f64,
    /// Memory usage in GB
    pub memory_used_gb: f64,
    /// Memory usage percentage
    pub memory_percent: f64,
    /// GPU temperature
    pub temperature_celsius: Option<f64>,
    /// GPU power draw
    pub power_watts: Option<f64>,
    /// GPU clock speeds
    pub clock_speeds: Option<GpuClocks>,
}

/// GPU clock speed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuClocks {
    /// Core clock in MHz
    pub core_mhz: f64,
    /// Memory clock in MHz
    pub memory_mhz: f64,
}

/// Memory utilization details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUtilization {
    /// Used system memory in GB
    pub used_gb: f64,
    /// Available system memory in GB
    pub available_gb: f64,
    /// Memory usage percentage
    pub usage_percent: f64,
    /// Swap usage in GB
    pub swap_used_gb: Option<f64>,
}

/// Storage I/O utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageUtilization {
    /// Read throughput in MB/s
    pub read_mbps: f64,
    /// Write throughput in MB/s
    pub write_mbps: f64,
    /// I/O operations per second
    pub iops: f64,
    /// Storage utilization percentage
    pub usage_percent: f64,
}

/// Network utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUtilization {
    /// Download throughput in Mbps
    pub download_mbps: f64,
    /// Upload throughput in Mbps
    pub upload_mbps: f64,
    /// Network latency in milliseconds
    pub latency_ms: f64,
    /// Packet loss percentage
    pub packet_loss_percent: f64,
}

/// Performance issues encountered during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Issue description
    pub description: String,
    /// Timestamp when issue occurred
    pub timestamp: u64,
    /// Suggested resolution
    pub suggestion: Option<String>,
}

/// Severity levels for performance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational message
    Info,
    /// Warning that may affect performance
    Warning,
    /// Error that significantly impacts performance
    Error,
    /// Critical error that prevents execution
    Critical,
}

/// Categories of performance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    /// Memory-related issues
    Memory,
    /// CPU performance issues
    Cpu,
    /// GPU performance issues
    Gpu,
    /// Storage I/O issues
    Storage,
    /// Network connectivity issues
    Network,
    /// Thermal throttling
    Thermal,
    /// Power consumption issues
    Power,
    /// Software configuration issues
    Configuration,
    /// Model-specific issues
    Model,
}

/// Feedback aggregation for improving predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackAggregation {
    /// System profile hash for grouping similar systems
    pub system_hash: String,
    /// Workload identifier
    pub workload_id: String,
    /// Number of feedback samples
    pub sample_count: u32,
    /// Aggregated performance metrics
    pub avg_metrics: PerformanceMetrics,
    /// Performance variance (standard deviation)
    pub variance: PerformanceMetrics,
    /// Most common issues
    pub common_issues: Vec<(PerformanceIssue, u32)>,
    /// Prediction accuracy
    pub prediction_accuracy: PredictionAccuracy,
}

/// Accuracy of system predictions vs actual performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionAccuracy {
    /// Predicted vs actual latency accuracy (0-1)
    pub latency_accuracy: Option<f64>,
    /// Predicted vs actual throughput accuracy (0-1)
    pub throughput_accuracy: Option<f64>,
    /// Predicted vs actual memory usage accuracy (0-1)
    pub memory_accuracy: Option<f64>,
    /// Overall prediction accuracy (0-1)
    pub overall_accuracy: f64,
}

/// Trait for feedback collectors that external monitoring systems can implement
pub trait FeedbackCollector: Send + Sync {
    /// Submit performance feedback
    fn submit_feedback(&mut self, feedback: PerformanceFeedback) -> Result<(), SystemAnalysisError>;
    
    /// Get aggregated feedback for a system/workload combination
    fn get_aggregation(&self, system_hash: &str, workload_id: &str) -> Option<FeedbackAggregation>;
    
    /// Get prediction accuracy statistics
    fn get_prediction_accuracy(&self, system_hash: &str) -> Option<PredictionAccuracy>;
    
    /// Clear old feedback data
    fn cleanup_old_data(&mut self, retention_days: u32) -> Result<u32, SystemAnalysisError>;
}

/// In-memory feedback collector for development and testing
pub struct InMemoryFeedbackCollector {
    feedback_history: Vec<PerformanceFeedback>,
    aggregations: HashMap<String, FeedbackAggregation>,
}

impl InMemoryFeedbackCollector {
    /// Create a new in-memory feedback collector
    pub fn new() -> Self {
        Self {
            feedback_history: Vec::new(),
            aggregations: HashMap::new(),
        }
    }
    
    /// Get current timestamp in seconds since Unix epoch
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Generate a hash key for system/workload combination
    fn generate_key(system_hash: &str, workload_id: &str) -> String {
        format!("{system_hash}:{workload_id}")
    }
    
    /// Update aggregations when new feedback is received
    fn update_aggregations(&mut self, feedback: &PerformanceFeedback) {
        let key = Self::generate_key(&feedback.system_profile.system_hash(), &feedback.workload_id);
        
        // For now, just store the latest feedback as aggregation
        // In a real implementation, you'd calculate proper aggregations
        let aggregation = FeedbackAggregation {
            system_hash: feedback.system_profile.system_hash(),
            workload_id: feedback.workload_id.clone(),
            sample_count: 1,
            avg_metrics: feedback.metrics.clone(),
            variance: feedback.metrics.clone(), // Placeholder
            common_issues: feedback.issues.iter()
                .map(|issue| (issue.clone(), 1))
                .collect(),
            prediction_accuracy: PredictionAccuracy {
                latency_accuracy: Some(0.8),    // Placeholder
                throughput_accuracy: Some(0.8), // Placeholder
                memory_accuracy: Some(0.8),     // Placeholder
                overall_accuracy: 0.8,          // Placeholder
            },
        };
        
        self.aggregations.insert(key, aggregation);
    }
}

impl FeedbackCollector for InMemoryFeedbackCollector {
    fn submit_feedback(&mut self, feedback: PerformanceFeedback) -> Result<(), SystemAnalysisError> {
        // Validate feedback
        if feedback.session_id.is_empty() {
            return Err(SystemAnalysisError::invalid_workload(
                "Session ID cannot be empty".to_string()
            ));
        }
        
        if feedback.workload_id.is_empty() {
            return Err(SystemAnalysisError::invalid_workload(
                "Workload ID cannot be empty".to_string()
            ));
        }
        
        // Update aggregations
        self.update_aggregations(&feedback);
        
        // Store feedback
        self.feedback_history.push(feedback);
        
        Ok(())
    }
    
    fn get_aggregation(&self, system_hash: &str, workload_id: &str) -> Option<FeedbackAggregation> {
        let key = Self::generate_key(system_hash, workload_id);
        self.aggregations.get(&key).cloned()
    }
    
    fn get_prediction_accuracy(&self, system_hash: &str) -> Option<PredictionAccuracy> {
        // Find aggregations for this system
        let system_aggregations: Vec<_> = self.aggregations.values()
            .filter(|agg| agg.system_hash == system_hash)
            .collect();
        
        if system_aggregations.is_empty() {
            return None;
        }
        
        // Calculate average accuracy across all workloads for this system
        let total_accuracy: f64 = system_aggregations.iter()
            .map(|agg| agg.prediction_accuracy.overall_accuracy)
            .sum();
        
        Some(PredictionAccuracy {
            latency_accuracy: Some(0.8),    // Placeholder
            throughput_accuracy: Some(0.8), // Placeholder
            memory_accuracy: Some(0.8),     // Placeholder
            overall_accuracy: total_accuracy / system_aggregations.len() as f64,
        })
    }
    
    fn cleanup_old_data(&mut self, retention_days: u32) -> Result<u32, SystemAnalysisError> {
        let cutoff_timestamp = Self::current_timestamp() - (retention_days as u64 * 24 * 60 * 60);
        let initial_count = self.feedback_history.len();
        
        self.feedback_history.retain(|feedback| feedback.timestamp >= cutoff_timestamp);
        
        let removed_count = initial_count - self.feedback_history.len();
        Ok(removed_count as u32)
    }
}

impl Default for InMemoryFeedbackCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Feedback integration for the SystemAnalyzer
pub struct FeedbackIntegration {
    collector: Box<dyn FeedbackCollector>,
    enabled: bool,
}

impl FeedbackIntegration {
    /// Create a new feedback integration with the specified collector
    pub fn new(collector: Box<dyn FeedbackCollector>) -> Self {
        Self {
            collector,
            enabled: true,
        }
    }
    
    /// Create feedback integration with in-memory collector
    pub fn with_memory_collector() -> Self {
        Self::new(Box::new(InMemoryFeedbackCollector::new()))
    }
    
    /// Enable or disable feedback collection
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Check if feedback collection is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Submit performance feedback
    pub fn submit_feedback(&mut self, feedback: PerformanceFeedback) -> Result<(), SystemAnalysisError> {
        if !self.enabled {
            return Ok(());
        }
        
        self.collector.submit_feedback(feedback)
    }
    
    /// Get feedback aggregation for improving predictions
    pub fn get_aggregation(&self, system_hash: &str, workload_id: &str) -> Option<FeedbackAggregation> {
        if !self.enabled {
            return None;
        }
        
        self.collector.get_aggregation(system_hash, workload_id)
    }
    
    /// Get prediction accuracy for a system
    pub fn get_prediction_accuracy(&self, system_hash: &str) -> Option<PredictionAccuracy> {
        if !self.enabled {
            return None;
        }
        
        self.collector.get_prediction_accuracy(system_hash)
    }
    
    /// Clean up old feedback data
    pub fn cleanup_old_data(&mut self, retention_days: u32) -> Result<u32, SystemAnalysisError> {
        if !self.enabled {
            return Ok(0);
        }
        
        self.collector.cleanup_old_data(retention_days)
    }
}

/// Builder for creating performance feedback
pub struct FeedbackBuilder {
    session_id: String,
    workload_id: String,
    system_profile: Option<SystemProfile>,
    metrics: PerformanceMetrics,
    resource_utilization: Option<ResourceUtilization>,
    issues: Vec<PerformanceIssue>,
    metadata: HashMap<String, String>,
}

impl FeedbackBuilder {
    /// Create a new feedback builder
    pub fn new(session_id: impl Into<String>, workload_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            workload_id: workload_id.into(),
            system_profile: None,
            metrics: PerformanceMetrics {
                latency_ms: None,
                throughput_ops_per_sec: None,
                peak_memory_gb: None,
                avg_memory_gb: None,
                cpu_utilization_percent: None,
                gpu_utilization_percent: None,
                power_consumption_watts: None,
                thermal: None,
                quality: None,
            },
            resource_utilization: None,
            issues: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set the system profile
    pub fn system_profile(mut self, profile: SystemProfile) -> Self {
        self.system_profile = Some(profile);
        self
    }
    
    /// Set latency measurement
    pub fn latency_ms(mut self, latency: f64) -> Self {
        self.metrics.latency_ms = Some(latency);
        self
    }
    
    /// Set throughput measurement
    pub fn throughput_ops_per_sec(mut self, throughput: f64) -> Self {
        self.metrics.throughput_ops_per_sec = Some(throughput);
        self
    }
    
    /// Set memory usage measurements
    pub fn memory_usage(mut self, peak_gb: f64, avg_gb: f64) -> Self {
        self.metrics.peak_memory_gb = Some(peak_gb);
        self.metrics.avg_memory_gb = Some(avg_gb);
        self
    }
    
    /// Set CPU utilization
    pub fn cpu_utilization(mut self, percent: f64) -> Self {
        self.metrics.cpu_utilization_percent = Some(percent);
        self
    }
    
    /// Set GPU utilization
    pub fn gpu_utilization(mut self, percent: f64) -> Self {
        self.metrics.gpu_utilization_percent = Some(percent);
        self
    }
    
    /// Add a performance issue
    pub fn add_issue(mut self, severity: IssueSeverity, category: IssueCategory, description: impl Into<String>) -> Self {
        self.issues.push(PerformanceIssue {
            severity,
            category,
            description: description.into(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            suggestion: None,
        });
        self
    }
    
    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Build the performance feedback
    pub fn build(self) -> Result<PerformanceFeedback, SystemAnalysisError> {
        let system_profile = self.system_profile.ok_or_else(|| {
            SystemAnalysisError::invalid_workload(
                "System profile is required for feedback".to_string()
            )
        })?;
        
        Ok(PerformanceFeedback {
            session_id: self.session_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            system_profile,
            workload_id: self.workload_id,
            metrics: self.metrics,
            resource_utilization: self.resource_utilization.unwrap_or_else(|| {
                // Default empty resource utilization
                ResourceUtilization {
                    cpu: CpuUtilization {
                        overall_percent: 0.0,
                        per_core_percent: Vec::new(),
                        frequency_mhz: None,
                        power_watts: None,
                    },
                    gpu: Vec::new(),
                    memory: MemoryUtilization {
                        used_gb: 0.0,
                        available_gb: 0.0,
                        usage_percent: 0.0,
                        swap_used_gb: None,
                    },
                    storage: StorageUtilization {
                        read_mbps: 0.0,
                        write_mbps: 0.0,
                        iops: 0.0,
                        usage_percent: 0.0,
                    },
                    network: NetworkUtilization {
                        download_mbps: 0.0,
                        upload_mbps: 0.0,
                        latency_ms: 0.0,
                        packet_loss_percent: 0.0,
                    },
                }
            }),
            issues: self.issues,
            metadata: self.metadata,
        })
    }
}

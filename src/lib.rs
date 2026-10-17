//! # System Analysis
//!
//! A comprehensive Rust library for analyzing system capabilities, workload requirements,
//! and optimal resource allocation. This crate provides tools for determining if a system
//! can run specific workloads, scoring hardware capabilities, and recommending optimal
//! configurations.
//!
//! ## Features
//!
//! - Comprehensive system capability analysis
//! - Workload requirement modeling and matching
//! - Resource utilization prediction
//! - Performance benchmarking framework
//! - Bottleneck detection and analysis
//! - Hardware capability scoring for different workload types
//! - Resource allocation optimization
//! - System compatibility checking
//! - AI/ML workload specialization
//! - Cross-platform support (Windows, Linux, macOS)
//!
//! ## Quick Start
//!
//! ```rust
//! use system_analysis::{SystemAnalyzer, WorkloadRequirements};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut analyzer = SystemAnalyzer::new();
//!     let system_profile = analyzer.analyze_system().await?;
//!     
//!     println!("System Overall Score: {}/10", system_profile.overall_score());
//!     Ok(())
//! }
//! ```

pub mod analyzer;
pub mod capabilities;
pub mod dynamic_models;
pub mod error;
pub mod feedback;
pub mod models;
pub mod resources;
pub mod types;
pub mod utils;
pub mod workloads;

// Re-export main types and functions
pub use analyzer::{AnalyzerConfig, SystemAnalyzer};
pub use dynamic_models::{DynamicModelDatabase, ModelDatabaseStats, ModelFetchConfig};
pub use error::{Result, SystemAnalysisError};
pub use feedback::{FeedbackBuilder, FeedbackCollector, FeedbackIntegration, PerformanceFeedback};
pub use models::{ModelDatabase, ModelDefinition, ModelRunner, get_model_database};
pub use types::{
    AIAcceleratorType, AICapabilities, AIUpgradeRecommendations, AIWorkloadRequirements,
    AcceleratorCompatibility, AcceleratorDevice, AcceleratorRecommendation, BottleneckSeverity,
    CapabilityLevel, CompatibilityResult, GPUUpgrade, LLMCapability, MemoryUpgrade,
    ModelBottleneck, ModelBottleneckType, ModelCompatibilityResult, PerformanceEstimate,
    PerformanceGainEstimate, PerformanceImpact, PerformanceLevel, PerformanceTargets,
    PerformanceTier, QuantizationSuggestion, StorageRecommendation, SystemProfile,
    WorkloadPriority, WorkloadRequirements,
};
pub use workloads::{
    AIInferenceWorkload, AIModel, AITaskType, AITrainingWorkload, CustomWorkload, InferenceConfig,
    ModelParameters, QuantizationLevel, TrainingConfig, Workload, WorkloadMetadata,
    WorkloadRegistry, WorkloadType,
};

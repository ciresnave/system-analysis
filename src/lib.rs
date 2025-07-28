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
pub mod error;
pub mod models;
pub mod resources;
pub mod types;
pub mod utils;
pub mod workloads;
pub mod feedback;
pub mod dynamic_models;

// Re-export main types and functions
pub use analyzer::{SystemAnalyzer, AnalyzerConfig};
pub use error::{SystemAnalysisError, Result};
pub use models::{ModelDatabase, ModelDefinition, ModelRunner, get_model_database};
pub use types::{
    SystemProfile, WorkloadRequirements, WorkloadPriority, PerformanceTargets,
    CompatibilityResult, PerformanceEstimate, PerformanceTier,
    AIAcceleratorType, AIWorkloadRequirements, ModelCompatibilityResult,
    AcceleratorCompatibility, AcceleratorDevice, QuantizationSuggestion,
    PerformanceLevel, ModelBottleneck, ModelBottleneckType, BottleneckSeverity,
    AIUpgradeRecommendations, MemoryUpgrade, GPUUpgrade, AcceleratorRecommendation,
    StorageRecommendation, PerformanceGainEstimate, PerformanceImpact,
    CapabilityLevel, LLMCapability, AICapabilities
};
pub use workloads::{
    Workload, WorkloadType, AIInferenceWorkload, AITrainingWorkload,
    CustomWorkload, WorkloadRegistry, AIModel, AITaskType, QuantizationLevel,
    ModelParameters, InferenceConfig, TrainingConfig, WorkloadMetadata
};
pub use feedback::{FeedbackIntegration, PerformanceFeedback, FeedbackBuilder, FeedbackCollector};
pub use dynamic_models::{DynamicModelDatabase, ModelFetchConfig, ModelDatabaseStats};

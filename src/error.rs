//! Error types for the system analysis crate.

use thiserror::Error;

/// The main error type for the system analysis crate.
#[derive(Error, Debug)]
pub enum SystemAnalysisError {
    /// Error occurred while analyzing system hardware
    #[error("System analysis failed: {message}")]
    AnalysisError { message: String },

    /// Error occurred while accessing system information
    #[error("System information access failed: {source}")]
    SystemInfoError { 
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>
    },

    /// Error occurred while calculating compatibility
    #[error("Compatibility calculation failed: {message}")]
    CompatibilityError { message: String },

    /// Invalid workload requirements
    #[error("Invalid workload requirements: {message}")]
    InvalidWorkload { message: String },

    /// Resource requirement not found
    #[error("Resource requirement not found: {resource_type}")]
    ResourceNotFound { resource_type: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// GPU detection error (when feature is enabled)
    #[cfg(feature = "gpu-detection")]
    #[error("GPU detection error: {message}")]
    GpuError { message: String },
}

impl SystemAnalysisError {
    /// Create a new analysis error
    pub fn analysis(message: impl Into<String>) -> Self {
        Self::AnalysisError {
            message: message.into(),
        }
    }

    /// Create a new system info error
    pub fn system_info(message: impl Into<String>) -> Self {
        Self::SystemInfoError {
            source: Box::new(std::io::Error::other(
                message.into()
            )),
        }
    }

    /// Create a new compatibility error
    pub fn compatibility(message: impl Into<String>) -> Self {
        Self::CompatibilityError {
            message: message.into(),
        }
    }

    /// Create a new invalid workload error
    pub fn invalid_workload(message: impl Into<String>) -> Self {
        Self::InvalidWorkload {
            message: message.into(),
        }
    }

    /// Create a new resource not found error
    pub fn resource_not_found(resource_type: impl Into<String>) -> Self {
        Self::ResourceNotFound {
            resource_type: resource_type.into(),
        }
    }

    /// Create a new config error
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    #[cfg(feature = "gpu-detection")]
    /// Create a new GPU error
    pub fn gpu(message: impl Into<String>) -> Self {
        Self::GpuError {
            message: message.into(),
        }
    }
}

/// Convenience type alias for Results with SystemAnalysisError
pub type Result<T> = std::result::Result<T, SystemAnalysisError>;

//! Tests for the system analysis crate

use system_analysis::{
    SystemAnalyzer, WorkloadRequirements, 
    resources::{ResourceRequirement, ResourceType, CapabilityLevel},
    workloads::{AIInferenceWorkload, ModelParameters, Workload},
};

#[tokio::test]
async fn test_basic_system_analysis() {
    let mut analyzer = SystemAnalyzer::new();
    let result = analyzer.analyze_system().await;
    
    assert!(result.is_ok(), "System analysis should succeed");
    
    let profile = result.unwrap();
    assert!(profile.overall_score() >= 0.0 && profile.overall_score() <= 10.0);
    assert!(profile.cpu_score() >= 0.0 && profile.cpu_score() <= 10.0);
    assert!(profile.gpu_score() >= 0.0 && profile.gpu_score() <= 10.0);
    assert!(profile.memory_score() >= 0.0 && profile.memory_score() <= 10.0);
    assert!(profile.storage_score() >= 0.0 && profile.storage_score() <= 10.0);
}

#[test]
fn test_workload_requirements_creation() {
    let mut requirements = WorkloadRequirements::new("test-workload");
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(8.0)
            .recommended_gb(16.0)
    );
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::CPU)
            .minimum_level(CapabilityLevel::Medium)
    );
    
    assert_eq!(requirements.name, "test-workload");
    assert_eq!(requirements.resource_requirements.len(), 2);
    
    let memory_req = requirements.get_resource_requirement(&ResourceType::Memory);
    assert!(memory_req.is_some());
    
    let cpu_req = requirements.get_resource_requirement(&ResourceType::CPU);
    assert!(cpu_req.is_some());
}

#[test]
fn test_ai_workload_creation() {
    let model_params = ModelParameters::new()
        .parameters(7_000_000_000)
        .memory_required(16.0)
        .compute_required(6.0)
        .prefer_gpu(true);
    
    let workload = AIInferenceWorkload::new(model_params.clone());
    
    assert_eq!(workload.model_params.parameters, 7_000_000_000);
    assert_eq!(workload.model_params.memory_required, 16.0);
    assert!(workload.model_params.prefer_gpu);
    
    // Test workload validation
    let validation_result = workload.validate();
    assert!(validation_result.is_ok());
    
    // Test resource requirements generation
    let requirements = workload.resource_requirements();
    assert!(!requirements.is_empty());
    
    // Should have memory, GPU, CPU, and storage requirements
    let memory_req = requirements.iter().find(|r| r.resource_type == ResourceType::Memory);
    assert!(memory_req.is_some());
    
    let gpu_req = requirements.iter().find(|r| r.resource_type == ResourceType::GPU);
    assert!(gpu_req.is_some());
    
    let cpu_req = requirements.iter().find(|r| r.resource_type == ResourceType::CPU);
    assert!(cpu_req.is_some());
}

#[tokio::test]
async fn test_compatibility_checking() {
    let mut analyzer = SystemAnalyzer::new();
    let system_profile = analyzer.analyze_system().await.unwrap();
    
    // Create a simple workload
    let mut requirements = WorkloadRequirements::new("simple-test");
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(1.0) // Very low requirement, should be satisfied
    );
    
    let compatibility = analyzer.check_compatibility(&system_profile, &requirements);
    assert!(compatibility.is_ok());
    
    let compat_result = compatibility.unwrap();
    assert!(compat_result.score >= 0.0 && compat_result.score <= 10.0);
}

#[tokio::test]
async fn test_resource_utilization_prediction() {
    let mut analyzer = SystemAnalyzer::new();
    let system_profile = analyzer.analyze_system().await.unwrap();
    
    let mut requirements = WorkloadRequirements::new("test-utilization");
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::CPU)
            .minimum_level(CapabilityLevel::Low)
    );
    
    let utilization = analyzer.predict_utilization(&system_profile, &requirements);
    assert!(utilization.is_ok());
    
    let util_result = utilization.unwrap();
    assert!(util_result.cpu_percent() >= 0.0 && util_result.cpu_percent() <= 100.0);
    assert!(util_result.memory_percent() >= 0.0 && util_result.memory_percent() <= 100.0);
}

#[test]
fn test_model_parameters_builder() {
    let params = ModelParameters::new()
        .parameters(1_000_000_000)
        .memory_required(8.0)
        .compute_required(5.0)
        .prefer_gpu(false)
        .context_length(2048)
        .batch_size(4);
    
    assert_eq!(params.parameters, 1_000_000_000);
    assert_eq!(params.memory_required, 8.0);
    assert_eq!(params.compute_required, 5.0);
    assert!(!params.prefer_gpu);
    assert_eq!(params.context_length, Some(2048));
    assert_eq!(params.batch_size, 4);
}

#[test]
fn test_workload_requirements_clone() {
    let model_params = ModelParameters::new()
        .parameters(1_000_000_000)
        .memory_required(4.0);
    
    let workload = AIInferenceWorkload::new(model_params);
    
    let mut requirements = WorkloadRequirements::new("clone-test");
    requirements.set_workload(Box::new(workload));
    
    // Test that cloning works
    let cloned = requirements.clone();
    assert_eq!(cloned.name, requirements.name);
    assert!(cloned.workload.is_some());
}

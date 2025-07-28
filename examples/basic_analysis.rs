use system_analysis::{SystemAnalyzer, WorkloadRequirements};
use system_analysis::workloads::{AIInferenceWorkload, ModelParameters, WorkloadType};
use system_analysis::resources::{ResourceRequirement, ResourceType, CapabilityLevel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create a system analyzer
    let mut analyzer = SystemAnalyzer::new();
    
    // Analyze the current system
    let system_profile = analyzer.analyze_system().await?;
    
    println!("System Capability Profile:");
    println!("  CPU Score: {}/10", system_profile.cpu_score());
    println!("  GPU Score: {}/10", system_profile.gpu_score());
    println!("  Memory Score: {}/10", system_profile.memory_score());
    println!("  Storage Score: {}/10", system_profile.storage_score());
    println!("  Overall Score: {}/10", system_profile.overall_score());
    
    // Check AI inference capabilities
    let ai_capabilities = system_profile.ai_capabilities();
    println!("AI Inference Capability: {:?}", ai_capabilities.inference_capability);
    println!("AI Training Capability: {:?}", ai_capabilities.training_capability);
    
    // Define a workload (e.g., running a specific AI model)
    let model_requirements = ModelParameters::new()
        .parameters(7_000_000_000)
        .memory_required(16.0)
        .compute_required(5.0)
        .prefer_gpu(true);
    
    let _inference_workload = AIInferenceWorkload::new(model_requirements);
    
    // Define workload requirements
    let mut workload_requirements = WorkloadRequirements::new("llama2-7b-inference");
    
    workload_requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(16.0)
            .recommended_gb(24.0)
    );
    
    workload_requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::GPU)
            .minimum_level(CapabilityLevel::Medium)
            .recommended_level(CapabilityLevel::High)
            .preferred_vendor(Some("NVIDIA"))
    );
    
    workload_requirements.workload = Some(WorkloadType::AIInference);
    
    // Check if the system can run the workload
    let compatibility = analyzer.check_compatibility(&system_profile, &workload_requirements)?;
    
    if compatibility.is_compatible {
        println!("System can run the workload!");
        println!("Compatibility score: {}/10", compatibility.score);
        println!("Expected performance: {:?}", compatibility.performance_estimate);
    } else {
        println!("System cannot run the workload!");
        println!("Missing requirements:");
        
        for missing in &compatibility.missing_requirements {
            println!("  - {}: required {}, available {}",
                missing.resource_type,
                missing.required,
                missing.available
            );
        }
        
        // Get upgrade recommendations
        let recommendations = analyzer.recommend_upgrades(&system_profile, &workload_requirements)?;
        
        println!("Recommended upgrades:");
        for upgrade in recommendations {
            println!("  - {}: {} (estimated improvement: {})",
                upgrade.resource_type,
                upgrade.recommendation,
                upgrade.estimated_improvement
            );
        }
    }
    
    // Predict resource utilization
    let utilization = analyzer.predict_utilization(&system_profile, &workload_requirements)?;
    
    println!("Predicted resource utilization:");
    println!("  CPU: {}%", utilization.cpu_percent);
    println!("  GPU: {}%", utilization.gpu_percent);
    println!("  Memory: {}%", utilization.memory_percent);
    
    // Find optimal configuration
    let optimal_config = analyzer.find_optimal_configuration(&workload_requirements)?;
    
    println!("Optimal hardware configuration:");
    println!("  CPU: {}", optimal_config.cpu_recommendation);
    println!("  GPU: {}", optimal_config.gpu_recommendation.as_deref().unwrap_or("Not required"));
    println!("  Memory: {}", optimal_config.memory_recommendation);
    println!("  Storage: {}", optimal_config.storage_recommendation);
    
    Ok(())
}

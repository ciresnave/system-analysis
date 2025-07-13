use system_analysis::{SystemAnalyzer, WorkloadRequirements};
use system_analysis::workloads::{AIInferenceWorkload, ModelParameters, QuantizationLevel, Workload};
use system_analysis::types::WorkloadPriority;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== AI Workload Analysis Demo ===\n");

    // Create a system analyzer
    let mut analyzer = SystemAnalyzer::new();
    
    // Analyze the current system
    println!("Analyzing system capabilities...");
    let system_profile = analyzer.analyze_system().await?;
    
    println!("System Analysis Results:");
    println!("  Overall Score: {:.1}/10", system_profile.overall_score());
    println!("  CPU Score: {:.1}/10", system_profile.cpu_score());
    println!("  GPU Score: {:.1}/10", system_profile.gpu_score());
    println!("  Memory: {:.1} GB", system_profile.system_info.memory_info.total_ram as f64 / 1024.0);
    println!("  GPUs: {}", system_profile.system_info.gpu_info.len());
    
    for (i, gpu) in system_profile.system_info.gpu_info.iter().enumerate() {
        println!("    GPU {}: {} ({})", i, gpu.name, gpu.vendor);
        if let Some(vram) = gpu.vram_size {
            println!("      VRAM: {:.1} GB", vram as f64 / 1024.0);
        }
    }
    println!();

    // Test different AI models
    let models = vec![
        ("GPT-3.5 (1.3B)", 1_300_000_000, 4.0, QuantizationLevel::None),
        ("LLaMA-7B", 7_000_000_000, 16.0, QuantizationLevel::None),
        ("LLaMA-7B Quantized", 7_000_000_000, 8.0, QuantizationLevel::Int4),
        ("LLaMA-13B", 13_000_000_000, 26.0, QuantizationLevel::None),
        ("LLaMA-70B", 70_000_000_000, 140.0, QuantizationLevel::None),
        ("GPT-4 Class (175B)", 175_000_000_000, 350.0, QuantizationLevel::None),
    ];

    for (model_name, params, memory_gb, quantization) in models {
        println!("--- Testing {} ---", model_name);
        
        // Create model parameters
        let model_params = ModelParameters::new()
            .parameters(params)
            .memory_required(memory_gb)
            .compute_required(6.0)
            .prefer_gpu(true)
            .quantization(quantization)
            .context_length(2048)
            .batch_size(1);

        // Create AI inference workload
        let ai_workload = AIInferenceWorkload::new(model_params);

        // Create workload requirements
        let mut workload_requirements = WorkloadRequirements::new(
            format!("{}-inference", model_name.replace(" ", "-").to_lowercase())
        );
        
        // Add resource requirements from the workload
        for req in ai_workload.resource_requirements() {
            workload_requirements.add_resource_requirement(req);
        }
        
        workload_requirements.set_workload(Box::new(ai_workload));
        workload_requirements.set_priority(WorkloadPriority::High);

        // Check compatibility
        let compatibility = analyzer.check_compatibility(&system_profile, &workload_requirements)?;
        
        println!("  Compatibility: {}", if compatibility.is_compatible() { "✓ Compatible" } else { "✗ Not Compatible" });
        println!("  Score: {:.1}/10", compatibility.score());
        println!("  Performance: {}", compatibility.performance_estimate());
        
        if !compatibility.missing_requirements().is_empty() {
            println!("  Missing requirements:");
            for missing in compatibility.missing_requirements() {
                println!("    - {}: need {}, have {}", 
                    missing.resource_type(), 
                    missing.required(), 
                    missing.available()
                );
            }
        }

        // Get resource utilization prediction
        let utilization = analyzer.predict_utilization(&system_profile, &workload_requirements)?;
        println!("  Resource utilization:");
        println!("    CPU: {:.1}%", utilization.cpu_percent());
        println!("    GPU: {:.1}%", utilization.gpu_percent());
        println!("    Memory: {:.1}%", utilization.memory_percent());

        // Check for bottlenecks
        if !compatibility.bottlenecks.is_empty() {
            println!("  Bottlenecks:");
            for bottleneck in &compatibility.bottlenecks {
                println!("    - {}: {}", bottleneck.resource_type, bottleneck.description);
            }
        }

        println!();
    }

    // Generate recommendations for the most demanding model that doesn't work
    println!("--- Upgrade Recommendations for Large Models ---");
    
    let demanding_model = ModelParameters::new()
        .parameters(70_000_000_000)
        .memory_required(140.0)
        .compute_required(8.0)
        .prefer_gpu(true);

    let demanding_workload = AIInferenceWorkload::new(demanding_model);
    let mut demanding_requirements = WorkloadRequirements::new("large-ai-model");
    
    for req in demanding_workload.resource_requirements() {
        demanding_requirements.add_resource_requirement(req);
    }
    demanding_requirements.set_workload(Box::new(demanding_workload));

    let upgrades = analyzer.recommend_upgrades(&system_profile, &demanding_requirements)?;
    
    if upgrades.is_empty() {
        println!("No upgrades needed - system can handle large AI models!");
    } else {
        println!("Recommended upgrades:");
        for upgrade in upgrades {
            println!("  - {}: {}", upgrade.resource_type(), upgrade.recommendation());
            println!("    Expected improvement: {}", upgrade.estimated_improvement());
            println!("    Priority: {:?}", upgrade.priority);
        }
    }

    println!("\n--- Optimal Configuration for AI Workloads ---");
    let optimal_config = analyzer.find_optimal_configuration(&demanding_requirements)?;
    
    println!("For optimal AI inference performance:");
    println!("  CPU: {}", optimal_config.cpu_recommendation());
    println!("  GPU: {}", optimal_config.gpu_recommendation());
    println!("  Memory: {}", optimal_config.memory_recommendation());
    println!("  Storage: {}", optimal_config.storage_recommendation());
    println!("  Network: {}", optimal_config.network_recommendation());
    
    if let Some(cost) = &optimal_config.total_cost {
        println!("  Estimated cost: ${:.0} - ${:.0} {}", 
            cost.min_cost, 
            cost.max_cost, 
            cost.currency
        );
    }

    Ok(())
}

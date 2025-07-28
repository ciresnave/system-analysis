use system_analysis::{SystemAnalyzer, error::Result};
use system_analysis::workloads::{AIModel, AITaskType, QuantizationLevel};
use system_analysis::types::{AIAcceleratorType, AIWorkloadRequirements};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Advanced AI Hardware Compatibility Analysis ===\n");

    // Create a system analyzer
    let mut analyzer = SystemAnalyzer::new();
    
    // Analyze the current system
    println!("Analyzing system capabilities and AI accelerators...");
    let system_profile = analyzer.analyze_system().await?;
    
    println!("System Analysis Results:");
    println!("  Overall Score: {:.1}/10", system_profile.overall_score());
    println!("  AI Workload Score: {:.1}/10", system_profile.ai_workload_score());
    println!("  AI Accelerator Score: {:.1}/10", system_profile.ai_accelerator_score());
    
    // Check for AI accelerators
    println!("\nAI Accelerator Detection:");
    if system_profile.has_ai_accelerators() {
        println!("  AI Accelerators Detected!");
        println!("  Total TOPS Performance: {:.1}", system_profile.total_tops_performance());
        
        // Print NPU information
        if !system_profile.system_info.npu_info.is_empty() {
            println!("  NPUs Detected:");
            for npu in &system_profile.system_info.npu_info {
                println!("    - {} ({})", npu.model_name, npu.vendor);
                if let Some(tops) = npu.tops_performance {
                    println!("      Performance: {tops:.1} TOPS");
                }
                println!("      Frameworks: {}", npu.supported_frameworks.join(", "));
            }
        }
        
        // Print TPU information
        if !system_profile.system_info.tpu_info.is_empty() {
            println!("  TPUs Detected:");
            for tpu in &system_profile.system_info.tpu_info {
                println!("    - {} ({})", tpu.model_name, tpu.vendor);
                if let Some(tops) = tpu.tops_performance {
                    println!("      Performance: {tops:.1} TOPS");
                }
                println!("      Architecture: {}", tpu.architecture);
            }
        }
    } else {
        println!("  No specialized AI accelerators detected");
    }
    
    // Check for ARM system
    if system_profile.is_arm_system() {
        if let Some(system_type) = system_profile.arm_system_type() {
            println!("  ARM System Detected: {system_type}");
            if let Some(arm_info) = &system_profile.system_info.arm_info {
                println!("    Board: {}", arm_info.board_model);
                println!("    Architecture: {}", arm_info.cpu_architecture);
                println!("    ML Capabilities: {:?}", arm_info.ml_capabilities);
            }
        }
    }
    
    // Test AI capabilities
    println!("\nAI Capabilities Assessment:");
    let ai_capabilities = system_profile.ai_capabilities();
    println!("  Inference Capability: {:?}", ai_capabilities.inference_capability);
    println!("  Training Capability: {:?}", ai_capabilities.training_capability);
    println!("  Edge Capability: {:?}", ai_capabilities.edge_capability);
    println!("  Max Model Size: {:.1} GB", ai_capabilities.max_model_size);
    println!("  LLM Capability: {:?}", ai_capabilities.llm_capability);
    println!("  Computer Vision: {:?}", ai_capabilities.computer_vision_capability);
    println!("  Supported Frameworks: {}", ai_capabilities.supported_frameworks.join(", "));
    
    println!("\n=== Testing AI Model Compatibility ===");
    
    // Define various AI models to test
    let models = vec![
        AIModel::new("MobileNet v2", "CNN", 3_500_000)
            .with_framework("TensorFlow Lite")
            .with_task(AITaskType::ImageClassification)
            .with_quantization(QuantizationLevel::Int8),
            
        AIModel::new("BERT Base", "Transformer", 110_000_000)
            .with_framework("PyTorch")
            .with_task(AITaskType::NLP)
            .with_context_length(512),
            
        AIModel::new("LLaMA 7B", "Transformer", 7_000_000_000)
            .with_framework("GGML")
            .with_task(AITaskType::TextGeneration)
            .with_context_length(2048)
            .with_quantization(QuantizationLevel::Int4),
            
        AIModel::new("Stable Diffusion XL", "Diffusion", 2_600_000_000)
            .with_framework("PyTorch")
            .with_task(AITaskType::TextGeneration) // Using TextGeneration as placeholder
            .with_quantization(QuantizationLevel::None),
            
        AIModel::new("GPT-4", "Transformer", 175_000_000_000)
            .with_framework("ONNX")
            .with_task(AITaskType::TextGeneration)
            .with_context_length(8192),
    ];
    
    // Test each model
    for model in &models {
        println!("\n--- Testing Model: {} ---", model.name);
        println!("  Architecture: {} ({} parameters)", model.architecture, format_params(model.parameters));
        println!("  Framework: {}", model.framework);
        println!("  Task: {:?}", model.task);
        println!("  Memory Required: {:.1} GB", model.memory_required);
        println!("  Quantization: {:?}", model.quantization);
        
        // Check model compatibility
        let compatibility = analyzer.check_model_compatibility(model)?;
        
        println!("  Compatibility Result: {}", if compatibility.can_run { "✓ Can Run" } else { "✗ Cannot Run" });
        
        if compatibility.memory_sufficient {
            println!("  Memory: ✓ Sufficient");
        } else {
            println!("  Memory: ✗ Insufficient");
        }
        
        // Print accelerator compatibility
        println!("  Compatible Devices:");
        for device in &compatibility.accelerator_compatibility.compatible_devices {
            println!("    - {device:?}");
        }
        
        if let Some(recommended) = &compatibility.accelerator_compatibility.recommended_device {
            println!("  Recommended Device: {recommended:?}");
            println!("  Expected Performance: {:?}", compatibility.accelerator_compatibility.expected_performance);
        }
        
        // Print optimal quantization
        println!("  Quantization Recommendation: {:?}", compatibility.optimal_quantization.recommended_level);
        println!("  Reasoning: {}", compatibility.optimal_quantization.reasoning);
        
        // Print bottlenecks
        if !compatibility.bottlenecks.is_empty() {
            println!("  Bottlenecks:");
            for bottleneck in &compatibility.bottlenecks {
                println!("    - {:?} ({:?}): {}", bottleneck.bottleneck_type, bottleneck.severity, bottleneck.description);
            }
        }
        
        println!("  Expected Inference Speed: {:.1} samples/sec", compatibility.expected_inference_speed);
        println!("  Recommended Batch Size: {}", compatibility.recommended_batch_size);
    }
    
    // Show hardware upgrade recommendations for the largest model
    println!("\n=== AI Hardware Upgrade Recommendations ===");
    
    let largest_model = &models[4]; // GPT-4
    
    // Create AI workload requirements
    let ai_requirements = AIWorkloadRequirements {
        base_requirements: "Large Language Model".to_string(),
        required_accelerator_types: vec![AIAcceleratorType::GPU, AIAcceleratorType::TPU],
        required_tops: Some(250.0),
        preferred_accelerator: Some(AIAcceleratorType::TPU),
        required_model_memory: largest_model.memory_required,
        supported_quantization: vec![QuantizationLevel::None, QuantizationLevel::Int8],
        min_inference_speed: Some(10.0),
        required_frameworks: vec!["ONNX".to_string(), "PyTorch".to_string()],
    };
    
    // Get upgrade recommendations
    let upgrades = analyzer.recommend_ai_hardware_upgrades(&ai_requirements)?;
    
    println!("Recommendations for running large language models:");
    println!("  Priority: {:?}", upgrades.priority);
    
    if let Some(memory) = &upgrades.memory_upgrade {
        println!("  Memory Upgrade:");
        println!("    {}", memory.description);
        println!("    Estimated Cost: ${:.2}", memory.estimated_cost_usd);
    }
    
    if let Some(gpu) = &upgrades.gpu_upgrade {
        println!("  GPU Upgrade:");
        println!("    {}", gpu.description);
        println!("    Estimated Cost: ${:.2}", gpu.estimated_cost_usd);
    }
    
    if let Some(accelerator) = &upgrades.accelerator_recommendation {
        println!("  Specialized Accelerator:");
        println!("    {}", accelerator.description);
        println!("    Estimated Cost: ${:.2}", accelerator.estimated_cost_usd);
    }
    
    if let Some(cost) = &upgrades.estimated_cost {
        println!("  Total Estimated Cost: ${:.2} - ${:.2}", cost.min_cost_usd, cost.max_cost_usd);
    }
    
    if let Some(perf) = &upgrades.performance_gain {
        println!("  Performance Improvements:");
        println!("    Latency: {:.1}% improvement", perf.latency_improvement_percent);
        println!("    Throughput: {:.1}% improvement", perf.throughput_improvement_percent);
        println!("    Energy Efficiency: {:.1}% improvement", perf.energy_efficiency_improvement_percent);
        println!("    {}", perf.description);
    }
    
    // Performance benefit estimation
    println!("\n=== AI Acceleration Performance Benefit ===");
    let acceleration_benefit = analyzer.estimate_ai_acceleration_benefit(&ai_requirements)?;
    
    println!("Specialized AI hardware would provide:");
    println!("  Speed: {:.1}x improvement", acceleration_benefit.speed_improvement_factor);
    println!("  Power Efficiency: {:.1}% improvement", acceleration_benefit.power_efficiency_improvement * 100.0);
    println!("  Cost per Performance: {:.2}", acceleration_benefit.cost_per_performance);
    println!("  Confidence: {:.0}%", acceleration_benefit.confidence_level * 100.0);
    println!("  {}", acceleration_benefit.description);
    
    Ok(())
}

// Helper function to format parameter counts in a readable way
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        params.to_string()
    }
}

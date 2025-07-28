use system_analysis::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing basic system analysis functionality...");
    
    // Create a system analyzer
    let mut analyzer = SystemAnalyzer::new();
    
    // Analyze the system
    match analyzer.analyze_system().await {
        Ok(system_profile) => {
            println!("System analysis completed successfully!");
            println!("Overall score: {:.2}", system_profile.overall_score());
            println!("CPU score: {:.2}", system_profile.cpu_score);
            println!("Memory score: {:.2}", system_profile.memory_score);
            
            // Test AI capability assessment
            let ai_capabilities = system_profile.ai_capabilities();
            println!("AI inference capability: {:?}", ai_capabilities.inference_capability);
            println!("Max model size: {:.1} GB", ai_capabilities.max_model_size);
        }
        Err(e) => {
            eprintln!("System analysis failed: {e}");
            return Err(e);
        }
    }
    
    Ok(())
}

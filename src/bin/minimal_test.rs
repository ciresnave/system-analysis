use system_analysis::SystemAnalyzer;

fn main() {
    println!("Testing basic analyzer creation...");
    
    // Just test that we can create an analyzer
    let _analyzer = SystemAnalyzer::new();
    
    println!("✓ Basic analyzer creation works!");
}

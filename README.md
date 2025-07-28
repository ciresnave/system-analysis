# System Analysis

[![Crates.io](https://img.shields.io/crates/v/system-analysis.svg)](https://crates.io/crates/system-analysis)
[![Documentation](https://docs.rs/system-analysis/badge.svg)](https://docs.rs/system-analysis)
[![License](https://img.shields.io/crates/l/system-analysis.svg)](https://github.com/yourusername/system-analysis#license)
[![Build Status](https://github.com/yourusername/system-analysis/workflows/CI/badge.svg)](https://github.com/yourusername/system-analysis/actions)

A comprehensive Rust library for analyzing system capabilities, workload requirements, and optimal resource allocation. This crate provides tools for determining if a system can run specific workloads, scoring hardware capabilities, and recommending optimal configurations with a focus on AI/ML workloads.

## 🚀 Features

- **🔍 Comprehensive System Analysis**: Detailed hardware capability assessment including CPU, GPU, memory, storage, and network
- **🤖 AI/ML Specialization**: Built-in support for AI inference and training workloads with model parameter analysis
- **⚖️ Workload Modeling**: Flexible framework for defining workload requirements and characteristics
- **✅ Compatibility Checking**: Determine if a system can run specific workloads with detailed compatibility scoring
- **📊 Resource Utilization Prediction**: Estimate resource usage patterns for workloads
- **🔴 Bottleneck Detection**: Identify system bottlenecks and performance limitations
- **⬆️ Upgrade Recommendations**: Suggest specific hardware upgrades for better performance
- **⚙️ Optimal Configuration**: Find the best hardware configuration for specific workloads
- **🌐 Cross-Platform Support**: Works on Windows, Linux, and macOS
- **🏃‍♂️ Performance Benchmarking**: Built-in benchmarking tools for capability validation
- **📈 Trend Analysis**: Track performance trends over time

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
system-analysis = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## 🏁 Quick Start

### Basic System Analysis

```rust
use system_analysis::SystemAnalyzer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = SystemAnalyzer::new();
    let system_profile = analyzer.analyze_system().await?;
    
    println!("System Overall Score: {}/10", system_profile.overall_score());
    println!("CPU Score: {}/10", system_profile.cpu_score());
    println!("GPU Score: {}/10", system_profile.gpu_score());
    println!("Memory Score: {}/10", system_profile.memory_score());
    
    Ok(())
}
```

### AI Workload Analysis

```rust
use system_analysis::{SystemAnalyzer, WorkloadRequirements};
use system_analysis::workloads::{AIInferenceWorkload, ModelParameters};
use system_analysis::resources::{ResourceRequirement, ResourceType, CapabilityLevel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = SystemAnalyzer::new();
    let system_profile = analyzer.analyze_system().await?;
    
    // Define an AI model workload
    let model_params = ModelParameters::new()
        .parameters(7_000_000_000)  // 7B parameter model
        .memory_required(16.0)      // 16GB memory requirement
        .compute_required(6.0)      // High compute requirement
        .prefer_gpu(true);          // Prefer GPU acceleration
    
    let ai_workload = AIInferenceWorkload::new(model_params);
    
    let mut workload_requirements = WorkloadRequirements::new("llama2-7b-inference");
    workload_requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(16.0)
            .recommended_gb(24.0)
    );
    workload_requirements.set_workload(Box::new(ai_workload));
    
    // Check compatibility
    let compatibility = analyzer.check_compatibility(&system_profile, &workload_requirements)?;
    
    if compatibility.is_compatible() {
        println!("✓ System can run the AI model!");
        println!("Performance estimate: {}", compatibility.performance_estimate());
    } else {
        println!("✗ System cannot run the AI model");
        for missing in compatibility.missing_requirements() {
            println!("Missing: {}", missing.resource_type());
        }
    }
    
    Ok(())
}
```

## Core Concepts

### System Profile

The `SystemProfile` represents a comprehensive analysis of system capabilities:

```rust
let system_profile = analyzer.analyze_system().await?;

// Access individual scores
println!("CPU Score: {}", system_profile.cpu_score());
println!("GPU Score: {}", system_profile.gpu_score());
println!("Memory Score: {}", system_profile.memory_score());

// Get AI capabilities assessment
let ai_capabilities = system_profile.ai_capabilities();
println!("AI Inference Level: {}", ai_capabilities.inference_level());
```

### Workload Requirements

Define what your workload needs to run effectively:

```rust
let mut requirements = WorkloadRequirements::new("my-workload");

// Add resource requirements
requirements.add_resource_requirement(
    ResourceRequirement::new(ResourceType::Memory)
        .minimum_gb(8.0)
        .recommended_gb(16.0)
        .critical()
);

requirements.add_resource_requirement(
    ResourceRequirement::new(ResourceType::GPU)
        .minimum_level(CapabilityLevel::High)
        .preferred_vendor(Some("NVIDIA"))
);
```

### Compatibility Analysis

Check if a system can run your workload:

```rust
let compatibility = analyzer.check_compatibility(&system_profile, &requirements)?;

println!("Compatible: {}", compatibility.is_compatible());
println!("Score: {}/10", compatibility.score());
println!("Performance: {}", compatibility.performance_estimate());

// Get missing requirements
for missing in compatibility.missing_requirements() {
    println!("Need {} {}, have {}", 
        missing.resource_type(), 
        missing.required(), 
        missing.available()
    );
}
```

### Resource Utilization Prediction

Estimate how your workload will use system resources:

```rust
let utilization = analyzer.predict_utilization(&system_profile, &requirements)?;

println!("Expected CPU usage: {}%", utilization.cpu_percent());
println!("Expected GPU usage: {}%", utilization.gpu_percent());
println!("Expected memory usage: {}%", utilization.memory_percent());
```

### Upgrade Recommendations

Get specific recommendations for improving system performance:

```rust
let upgrades = analyzer.recommend_upgrades(&system_profile, &requirements)?;

for upgrade in upgrades {
    println!("Upgrade {}: {}", 
        upgrade.resource_type(), 
        upgrade.recommendation()
    );
    println!("Expected improvement: {}", upgrade.estimated_improvement());
}
```

## Advanced Usage

### Custom Workloads

Implement the `Workload` trait for custom workload types:

```rust
use system_analysis::workloads::{Workload, WorkloadType, PerformanceCharacteristics};

struct CustomWorkload {
    // Your workload parameters
}

impl Workload for CustomWorkload {
    fn workload_type(&self) -> WorkloadType {
        WorkloadType::Custom("my-workload".to_string())
    }
    
    fn resource_requirements(&self) -> Vec<ResourceRequirement> {
        // Define your resource requirements
        vec![]
    }
    
    fn estimated_utilization(&self) -> HashMap<ResourceType, f64> {
        // Define expected resource utilization
        HashMap::new()
    }
    
    // ... implement other required methods
}
```

### GPU Detection

Enable GPU detection for NVIDIA GPUs:

```toml
[dependencies]
system-analysis = { version = "0.1.0", features = ["gpu-detection"] }
```

### Performance Benchmarking

Run performance benchmarks to validate system capabilities:

```rust
use system_analysis::utils::BenchmarkRunner;
use std::time::Duration;

let benchmark = BenchmarkRunner::new(Duration::from_secs(5), 100);
let cpu_result = benchmark.run_cpu_benchmark()?;
let memory_result = benchmark.run_memory_benchmark()?;

println!("CPU Benchmark Score: {}", cpu_result.score);
println!("Memory Benchmark Score: {}", memory_result.score);
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_analysis.rs`: Basic system analysis and workload compatibility
- `ai_workload_analysis.rs`: AI/ML workload analysis with multiple models

Run examples with:

```bash
cargo run --example basic_analysis
cargo run --example ai_workload_analysis
```

## Configuration

Customize the analyzer behavior:

```rust
use system_analysis::analyzer::AnalyzerConfig;

let config = AnalyzerConfig {
    enable_gpu_detection: true,
    enable_detailed_cpu_analysis: true,
    enable_network_testing: false,
    cache_duration_seconds: 300,
    enable_benchmarking: false,
    benchmark_timeout_seconds: 30,
};

let analyzer = SystemAnalyzer::with_config(config);
```

## API Reference

### Main Types

- `SystemAnalyzer`: Main analyzer for system capabilities
- `SystemProfile`: Comprehensive system analysis results
- `WorkloadRequirements`: Specification of workload needs
- `CompatibilityResult`: Results of compatibility analysis
- `ResourceUtilization`: Resource usage predictions
- `UpgradeRecommendation`: Hardware upgrade suggestions

### Resource Types

- `ResourceType`: CPU, GPU, Memory, Storage, Network
- `CapabilityLevel`: VeryLow, Low, Medium, High, VeryHigh, Exceptional
- `ResourceAmount`: Different ways to specify resource amounts

### Workload Types

- `AIInferenceWorkload`: AI model inference workloads
- `ModelParameters`: AI model parameter specifications
- Custom workloads via the `Workload` trait

## Error Handling

The crate uses the `SystemAnalysisError` enum for comprehensive error handling:

```rust
use system_analysis::error::SystemAnalysisError;

match result {
    Ok(profile) => println!("Analysis successful: {}", profile.overall_score()),
    Err(SystemAnalysisError::AnalysisError { message }) => {
        eprintln!("Analysis failed: {}", message);
    }
    Err(SystemAnalysisError::SystemInfoError { source }) => {
        eprintln!("System information error: {}", source);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Platform Support

- **Windows**: Full support with Windows-specific optimizations
- **Linux**: Full support with Linux-specific hardware detection
- **macOS**: Full support with macOS-specific APIs

## Performance Considerations

- System analysis results are cached for 5 minutes by default
- GPU detection can be disabled to reduce startup time
- Network testing is disabled by default as it can be slow
- Benchmarking is optional and disabled by default

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under either of

- Apache License, Version 2.0
- MIT License

at your option.

## Changelog

### 0.1.0 (Initial Release)

- Basic system analysis functionality
- AI workload specialization
- Compatibility checking
- Resource utilization prediction
- Upgrade recommendations
- Cross-platform support
- Comprehensive examples and documentation

## 🔧 Advanced Usage

### Custom Workload Definition

```rust
use system_analysis::{
    workloads::{Workload, WorkloadType},
    resources::{ResourceRequirement, ResourceType, CapabilityLevel},
};

struct CustomWorkload {
    name: String,
    cpu_intensive: bool,
    memory_gb: f64,
}

impl Workload for CustomWorkload {
    fn workload_type(&self) -> WorkloadType {
        WorkloadType::Custom(self.name.clone())
    }
    
    fn resource_requirements(&self) -> Vec<ResourceRequirement> {
        let mut requirements = Vec::new();
        
        requirements.push(
            ResourceRequirement::new(ResourceType::Memory)
                .minimum_gb(self.memory_gb)
        );
        
        if self.cpu_intensive {
            requirements.push(
                ResourceRequirement::new(ResourceType::CPU)
                    .minimum_level(CapabilityLevel::High)
                    .cores(8)
            );
        }
        
        requirements
    }
    
    fn validate(&self) -> system_analysis::Result<()> {
        if self.memory_gb <= 0.0 {
            return Err(system_analysis::SystemAnalysisError::invalid_workload(
                "Memory requirement must be positive"
            ));
        }
        Ok(())
    }
    
    fn clone_workload(&self) -> Box<dyn Workload> {
        Box::new(CustomWorkload {
            name: self.name.clone(),
            cpu_intensive: self.cpu_intensive,
            memory_gb: self.memory_gb,
        })
    }
}
```

### System Capability Profiling

```rust
use system_analysis::{SystemAnalyzer, capabilities::CapabilityProfile};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = SystemAnalyzer::new();
    let system_profile = analyzer.analyze_system().await?;
    let capabilities = analyzer.get_capability_profile().await?;
    
    // CPU Analysis
    println!("CPU: {} ({} cores)", 
        capabilities.cpu.brand, 
        capabilities.cpu.physical_cores
    );
    println!("CPU AI Score: {}/10", capabilities.cpu.ai_score);
    
    // GPU Analysis
    for (i, gpu) in capabilities.gpu.iter().enumerate() {
        println!("GPU {}: {} ({}GB VRAM)", 
            i + 1, 
            gpu.name, 
            gpu.memory_gb
        );
        println!("GPU AI Score: {}/10", gpu.ai_score);
    }
    
    // Memory Analysis
    println!("Memory: {:.1}GB {} @ {}MHz", 
        capabilities.memory.total_gb,
        capabilities.memory.memory_type,
        capabilities.memory.frequency_mhz.unwrap_or(0)
    );
    
    Ok(())
}
```

### Advanced Performance Benchmarking

```rust
use system_analysis::{SystemAnalyzer, utils::BenchmarkRunner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = SystemAnalyzer::new();
    let mut benchmark_runner = BenchmarkRunner::new();
    
    // Run CPU benchmark
    let cpu_score = benchmark_runner.benchmark_cpu().await?;
    println!("CPU Benchmark Score: {}/10", cpu_score);
    
    // Run memory benchmark
    let memory_score = benchmark_runner.benchmark_memory().await?;
    println!("Memory Benchmark Score: {}/10", memory_score);
    
    // Run comprehensive system benchmark
    let overall_score = benchmark_runner.benchmark_system().await?;
    println!("Overall System Score: {}/10", overall_score);
    
    Ok(())
}
```

## 🏗️ Architecture

The crate is organized into several key modules:

- **`analyzer`**: Main system analysis logic and orchestration
- **`capabilities`**: Hardware capability assessment and scoring
- **`workloads`**: Workload definitions and AI/ML specializations
- **`resources`**: Resource management and requirement modeling
- **`types`**: Core data types and structures
- **`utils`**: Utility functions and helper tools
- **`error`**: Comprehensive error handling

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test modules
cargo test integration_tests
cargo test edge_case_tests

# Run benchmarks
cargo bench
```

## 📊 Benchmarking

The crate includes comprehensive benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench -- system_analysis
cargo bench -- workload_creation
cargo bench -- compatibility_checking

# Generate HTML benchmark reports
cargo bench -- --output-format html
```

## 🎯 Use Cases

### AI/ML Model Deployment

- **Model Compatibility**: Check if your system can run specific AI models
- **Performance Estimation**: Predict inference speed and resource usage
- **Hardware Recommendations**: Get specific upgrade suggestions for AI workloads
- **Batch Processing**: Optimize batch sizes based on available resources

### System Planning

- **Hardware Procurement**: Make informed decisions about hardware purchases
- **Capacity Planning**: Understand current and future resource needs
- **Bottleneck Analysis**: Identify and resolve system limitations
- **Cost Optimization**: Balance performance and cost requirements

### DevOps and Infrastructure

- **Container Sizing**: Right-size containers based on workload requirements
- **Auto-scaling**: Make intelligent scaling decisions based on system capabilities
- **Resource Allocation**: Optimize resource distribution across workloads
- **Performance Monitoring**: Track system capability trends over time

## 🔌 Feature Flags

Enable optional features based on your needs:

```toml
[dependencies]
system-analysis = { version = "0.1.0", features = ["gpu-detection"] }
```

Available features:

- **`gpu-detection`**: Enable NVIDIA GPU detection and CUDA capabilities
- **`advanced-benchmarks`**: Include additional benchmarking tools
- **`ml-models`**: Extended AI/ML model support and optimizations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/system-analysis.git
cd system-analysis
cargo build
cargo test
```

### Contribution Areas

- **Hardware Support**: Add support for new hardware types
- **Workload Types**: Implement new workload categories
- **Platform Support**: Enhance cross-platform compatibility
- **Performance**: Optimize analysis algorithms
- **Documentation**: Improve docs and examples

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## 🙏 Acknowledgments

- The Rust community for excellent system information crates
- Contributors and testers who help improve this project
- AI/ML community for workload insights and requirements

## 📈 Roadmap

- [ ] Support for ARM and Apple Silicon optimization
- [ ] Integration with cloud provider APIs
- [ ] Real-time system monitoring capabilities
- [ ] Machine learning-based performance prediction
- [ ] Web-based dashboard for system analysis
- [ ] Integration with container orchestration platforms
- [ ] Support for specialized AI hardware (TPUs, FPGAs)

---

For more detailed documentation, please visit [docs.rs/system-analysis](https://docs.rs/system-analysis).

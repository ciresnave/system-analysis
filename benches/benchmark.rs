//! Benchmarks for the system analysis crate

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use system_analysis::{
    SystemAnalyzer, WorkloadRequirements, Workload,
    resources::{ResourceRequirement, ResourceType, CapabilityLevel},
    workloads::{AIInferenceWorkload, ModelParameters, QuantizationLevel},
};
use tokio::runtime::Runtime;

fn bench_system_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut analyzer = SystemAnalyzer::new();
    
    c.bench_function("system_analysis", |b| {
        b.iter(|| {
            rt.block_on(analyzer.analyze_system()).unwrap()
        })
    });
}

fn bench_workload_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload_creation");
    
    for &param_count in &[1_000_000, 1_000_000_000, 7_000_000_000, 70_000_000_000] {
        group.bench_with_input(
            BenchmarkId::new("ai_inference", param_count),
            &param_count,
            |b, &param_count| {
                b.iter(|| {
                    let model_params = ModelParameters::new()
                        .parameters(param_count)
                        .quantization(QuantizationLevel::Int8)
                        .context_length(2048);
                    AIInferenceWorkload::new(model_params)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_compatibility_checking(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut analyzer = SystemAnalyzer::new();
    let system_profile = rt.block_on(analyzer.analyze_system()).unwrap();
    
    let mut group = c.benchmark_group("compatibility_checking");
    
    // Create workloads of different complexity
    let workloads = vec![
        ("simple", create_simple_workload()),
        ("medium", create_medium_workload()),
        ("complex", create_complex_workload()),
    ];
    
    for (name, workload) in workloads {
        group.bench_with_input(
            BenchmarkId::new("compatibility", name),
            &workload,
            |b, workload| {
                b.iter(|| {
                    analyzer.check_compatibility(&system_profile, workload).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_resource_requirements(c: &mut Criterion) {
    let mut group = c.benchmark_group("resource_requirements");
    
    for &param_count in &[1_000_000, 1_000_000_000, 7_000_000_000] {
        group.bench_with_input(
            BenchmarkId::new("generate_requirements", param_count),
            &param_count,
            |b, &param_count| {
                let model_params = ModelParameters::new()
                    .parameters(param_count)
                    .quantization(QuantizationLevel::Int8)
                    .context_length(2048);
                let workload = AIInferenceWorkload::new(model_params);
                
                b.iter(|| workload.resource_requirements())
            },
        );
    }
    
    group.finish();
}

fn create_simple_workload() -> WorkloadRequirements {
    let mut requirements = WorkloadRequirements::new("simple-workload");
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(4.0)
            .recommended_gb(8.0)
    );
    requirements
}

fn create_medium_workload() -> WorkloadRequirements {
    let mut requirements = WorkloadRequirements::new("medium-workload");
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(16.0)
            .recommended_gb(32.0)
    );
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::CPU)
            .minimum_level(CapabilityLevel::Medium)
            .recommended_level(CapabilityLevel::High)
    );
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::GPU)
            .minimum_level(CapabilityLevel::High)
            .gpu_memory_gb(8.0)
    );
    requirements
}

fn create_complex_workload() -> WorkloadRequirements {
    let mut requirements = WorkloadRequirements::new("complex-workload");
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Memory)
            .minimum_gb(64.0)
            .recommended_gb(128.0)
            .required()
    );
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::CPU)
            .minimum_level(CapabilityLevel::High)
            .recommended_level(CapabilityLevel::VeryHigh)
            .cores(16)
            .minimum_ghz(3.0)
    );
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::GPU)
            .minimum_level(CapabilityLevel::VeryHigh)
            .recommended_level(CapabilityLevel::Exceptional)
            .gpu_memory_gb(24.0)
            .required()
    );
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Storage)
            .minimum_gb(1000.0)
            .recommended_gb(2000.0)
            .storage_type("NVMe SSD".to_string())
    );
    
    requirements.add_resource_requirement(
        ResourceRequirement::new(ResourceType::Network)
            .minimum_mbps(1000.0)
            .recommended_gb(10000.0)  // Note: using recommended_gb as there's no recommended_mbps
    );
    
    requirements
}

criterion_group!(
    benches,
    bench_system_analysis,
    bench_workload_creation,
    bench_compatibility_checking,
    bench_resource_requirements
);
criterion_main!(benches);
// Test file to check hardware-query API with complete HardwareInfo

use hardware_query::HardwareInfo;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing complete hardware-query integration...");

    // Get complete system information
    let hw_info = HardwareInfo::query()?;

    // Access CPU information
    let cpu = hw_info.cpu();
    println!("\n🖥️  CPU Information:");
    println!(
        "CPU: {} {} with {} cores, {} threads",
        cpu.vendor(),
        cpu.model_name(),
        cpu.physical_cores(),
        cpu.logical_cores()
    );

    // Check for specific CPU features
    if cpu.has_feature("avx2") {
        println!("✅ CPU supports AVX2 instructions");
    }
    if cpu.has_feature("avx512") {
        println!("✅ CPU supports AVX512 instructions");
    }

    // Get GPU information
    println!("\n🎮 GPU Information:");
    for (i, gpu) in hw_info.gpus().iter().enumerate() {
        println!(
            "GPU {}: {} {} with {} GB VRAM",
            i + 1,
            gpu.vendor(),
            gpu.model_name(),
            gpu.memory_gb()
        );
    }

    // Check specialized hardware
    println!("\n🤖 AI Accelerators:");
    if !hw_info.npus().is_empty() {
        println!("✅ NPUs detected: {} units", hw_info.npus().len());
        for (i, npu) in hw_info.npus().iter().enumerate() {
            println!("  NPU {}: {} {}", i + 1, npu.vendor(), npu.model_name());
        }
    } else {
        println!("❌ No NPUs detected");
    }

    if !hw_info.tpus().is_empty() {
        println!("✅ TPUs detected: {} units", hw_info.tpus().len());
        for (i, tpu) in hw_info.tpus().iter().enumerate() {
            println!("  TPU {}: {} {}", i + 1, tpu.vendor(), tpu.model_name());
        }
    } else {
        println!("❌ No TPUs detected");
    }

    // Check ARM-specific hardware (Raspberry Pi, Jetson, etc.)
    println!("\n🔧 Specialized Hardware:");
    if let Some(arm) = hw_info.arm_hardware() {
        println!("✅ ARM System: {}", arm.system_type);
    } else {
        println!("❌ Not an ARM system");
    }

    // Check FPGA hardware
    if !hw_info.fpgas().is_empty() {
        println!("✅ FPGAs detected: {} units", hw_info.fpgas().len());
        for (i, fpga) in hw_info.fpgas().iter().enumerate() {
            println!("  FPGA {}: {} {}", i + 1, fpga.vendor, fpga.family);
        }
    } else {
        println!("❌ No FPGAs detected");
    }

    // Get memory information
    let memory = hw_info.memory();
    println!("\n💾 Memory Information:");
    println!("Total RAM: {:.1} GB", memory.total_gb());
    println!("Available RAM: {:.1} GB", memory.available_gb());

    // Get system summary
    println!("\n📊 System Summary:");
    let summary = hw_info.summary();
    println!("{}", summary);

    Ok(())
}

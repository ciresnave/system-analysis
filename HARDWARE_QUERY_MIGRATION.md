# Hardware-Query Migration Guide

## Full Migration Plan from sysinfo to hardware-query

This document outlines the complete migration strategy for replacing sysinfo with your hardware-query crate in the system-analysis library.

## ✅ Completed Modifications

### 1. Updated Cargo.toml

- Added hardware-query dependency placeholder
- Updated feature flags for GPU vendor detection
- Removed nvml-wrapper (now included in hardware-query)

### 2. Enhanced Type System

Added new types in `src/types.rs`:

- `NpuInfo` - Neural Processing Unit information
- `TpuInfo` - Tensor Processing Unit information  
- `FpgaInfo` - FPGA accelerator information
- `ArmInfo` - ARM-specific hardware details

### 3. Updated SystemInfo Structure

Extended `SystemInfo` to include:

```rust
pub struct SystemInfo {
    // ...existing fields...
    pub npu_info: Vec<NpuInfo>,
    pub tpu_info: Vec<TpuInfo>, 
    pub fpga_info: Vec<FpgaInfo>,
    pub arm_info: Option<ArmInfo>,
}
```

### 4. Created Conversion Methods

Added hardware-query conversion methods in `src/analyzer.rs`:

- `convert_cpu_info()` - Enhanced CPU detection
- `convert_gpu_info()` - Professional GPU detection
- `convert_npu_info()` - NPU accelerator detection
- `convert_tpu_info()` - TPU accelerator detection
- `convert_fpga_info()` - FPGA accelerator detection
- `convert_arm_info()` - ARM system detection
- `convert_memory_info()` - Enhanced memory info
- `convert_storage_info()` - Storage device detection
- `convert_network_info()` - Network interface detection

## 🔄 Migration Steps to Complete

### 1. Set Hardware-Query Path

Update `Cargo.toml` with correct path:

```toml
[dependencies]
hardware-query = { path = "path/to/hardware-query" }
```

### 2. Enable Hardware-Query Import

Uncomment in `src/analyzer.rs`:

```rust
use hardware_query::{HardwareInfo, CPUFeature};
```

### 3. Update get_system_info Method

The method is already updated to use hardware-query:

```rust
let hw_info = HardwareInfo::query()?;
```

### 4. API Compatibility Fixes

Some method calls need adjustment based on actual hardware-query API:

- CPU cache methods
- GPU compute capability format
- Memory type detection
- Network speed handling

## 🚀 Expected Benefits After Migration

### AI/ML Workload Analysis

- **NPU Detection**: Intel Movidius, Apple Neural Engine, Qualcomm Hexagon
- **TPU Detection**: Google Cloud TPU, Edge TPU, Intel Habana with TOPS ratings
- **FPGA Support**: Intel/Altera, Xilinx with logic element counts
- **Performance Metrics**: Direct TOPS performance for accelerator scoring

### Enhanced Hardware Detection

- **CPU Features**: AVX, AVX2, AVX512, FMA instruction sets
- **GPU Capabilities**: CUDA/ROCm/DirectML support detection
- **Memory Details**: Speed, type, configuration
- **ARM Systems**: Raspberry Pi, Jetson, Apple Silicon specifics

### Improved Compatibility Analysis

- **AI Accelerator Scoring**: TOPS-based performance estimates
- **Specialized Hardware Matching**: NPU/TPU requirements for workloads
- **Platform-Specific Optimizations**: ARM-aware workload placement
- **Enhanced Bottleneck Detection**: AI-specific hardware limitations

## 📊 Migration Impact Assessment

| Component | Before (sysinfo) | After (hardware-query) | Improvement |
|-----------|------------------|------------------------|-------------|
| CPU Analysis | Basic info | Detailed features + cache | ⭐⭐⭐⭐⭐ |
| GPU Detection | NVML only | Multi-vendor APIs | ⭐⭐⭐⭐⭐ |
| AI Accelerators | None | NPU/TPU/FPGA with TOPS | 🚀 **Game Changer** |
| Memory Info | Total/available | Speed, type, modules | ⭐⭐⭐⭐ |
| ARM Support | None | Pi/Jetson/Apple Silicon | ⭐⭐⭐⭐⭐ |
| Cross-Platform | Generic | Platform-specific APIs | ⭐⭐⭐⭐ |

## 🛠️ Final Steps

1. **Clone hardware-query** to the correct path
2. **Update dependency path** in Cargo.toml
3. **Enable imports** in analyzer.rs
4. **Fix API compatibility** issues
5. **Test comprehensive detection** across platforms
6. **Update examples** to showcase AI hardware
7. **Add benchmarks** for performance comparison

## 🎯 Post-Migration Enhancements

### New Capabilities to Implement

1. **AI Workload Optimizer**: Match models to optimal hardware
2. **TOPS-based Scoring**: Performance estimation using accelerator metrics
3. **Multi-accelerator Workloads**: Distribute across NPU/TPU/GPU
4. **ARM-optimized Pipelines**: Leverage Pi/Jetson-specific features
5. **Hardware-aware Model Selection**: Recommend models based on capabilities

This migration transforms system-analysis from basic hardware detection to comprehensive AI-focused system analysis.

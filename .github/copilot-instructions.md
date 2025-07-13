# Copilot Instructions for System Analysis Crate

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a comprehensive Rust library for analyzing system capabilities, workload requirements, and optimal resource allocation. The crate provides tools for determining if a system can run specific workloads, scoring hardware capabilities, and recommending optimal configurations.

## Key Principles
- Use async/await patterns for I/O operations
- Implement comprehensive error handling with `thiserror`
- Design for extensibility and modularity
- Focus on AI/ML workload specialization
- Provide cross-platform support (Windows, Linux, macOS)
- Use strong typing and clear API design
- Include comprehensive documentation and examples

## Architecture Guidelines
- Core types in `src/types.rs`
- System analysis logic in `src/analyzer.rs`
- Workload definitions in `src/workloads/`
- Resource management in `src/resources/`
- Hardware capabilities in `src/capabilities/`
- Utilities and helpers in `src/utils/`

## Code Style
- Use descriptive error types
- Implement `Debug`, `Clone`, `Serialize`, `Deserialize` where appropriate
- Prefer composition over inheritance
- Use builder patterns for complex configurations
- Include comprehensive unit tests
- Add integration tests for full workflows

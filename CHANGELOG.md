# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2025-10-03

### Updated

- Updated criterion from 0.6.0 to 0.7.0 for improved benchmarking
- Updated scraper from 0.21.0 to 0.24.0 for latest web scraping capabilities  
- Updated sysinfo from 0.33.1 to 0.37.2 for enhanced system information gathering
- Updated MSRV to Rust 1.88 to support latest sysinfo requirements

### Fixed

- Fixed sysinfo API compatibility issue with `physical_core_count()` method (now static)

## [0.2.0] - Previous Release

### Added
- Initial implementation of system analysis capabilities
- Hardware capability profiling (CPU, GPU, Memory, Storage, Network)
- AI/ML workload modeling and compatibility checking
- Resource requirement specification system
- Performance estimation and bottleneck detection
- System upgrade recommendations
- Cross-platform support (Windows, Linux, macOS)

### Changed
- Updated to sysinfo 0.32 for improved system information gathering

### Fixed
- Fixed compilation issues with sysinfo API changes
- Resolved trait import issues in examples and tests

## [0.1.0] - 2025-07-10

### Added
- Initial release
- Basic system analysis framework
- AI workload specialization
- Comprehensive documentation
- Example applications
- Integration test suite

### Security
- No known security issues

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
 * MIT license ([LICENSE-MIT](LICENSE-MIT))
at your option.
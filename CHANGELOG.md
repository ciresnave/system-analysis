# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-17

### Breaking

- Raised MSRV from 1.88 to 1.95, to take hardware-query 0.3.0 and the
  other dependency bumps below.

### Updated

- hardware-query 0.2.0 -> 0.3.0 (pulls in its Linux-build fix,
  dependency-freshness pass, and clippy/fmt cleanup; this crate only
  calls `HardwareInfo::query()` and its stable accessor methods,
  none of which were renamed)
- reqwest 0.12 -> 0.13 (the `rustls-tls` feature was renamed to
  `rustls`; no other API usage affected)
- scraper 0.24 -> 0.27
- nvml-wrapper 0.11 -> 0.13
- sysinfo 0.37.2 -> 0.39
- criterion 0.7 -> 0.8 (dev-dependency)
- tempfile 3.8 -> 3.27 (dev-dependency)
- rayon 1.7 -> 1.12, dashmap 6.1 -> 6.2, tokio 1.0 -> 1.53 (no
  breaking usage encountered)

### Fixed

- Collapsed several nested if-let/match patterns into if-let-chains
  and replaced two reverse-order `sort_by` closures with
  `sort_by_key(Reverse(..))`, both newly suggested by clippy once the
  MSRV bump unlocked let-chain syntax. No behavior change.
- Ran `cargo fmt` for the first time on this repo (formatting only).

### CI

- Bumped the `msrv` job's pinned toolchain from 1.70.0 to 1.95.0 to
  match the new `rust-version`. The existing `test` job's
  ubuntu-latest/windows-latest/macos-latest matrix already covers
  Linux (and Windows/macOS) builds, so no separate Linux-only job
  was needed.

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
# Changelog

## 0.1.0 - Unreleased

- Stabilized the standalone CellShard package contract around installed CMake targets and documented include usage.
- Added an install-consumer check that validates both inspect-only and CUDA runtime consumers.
- Hardened Python packaging so Linux wheel and source-install paths build the Python extension without forcing the standalone export app into wheel builds.
- Restored downstream Cellerator compatibility for the dense-reduce surface after the `part` to `partition` API migration.
- Added initial support-matrix documentation for the standalone release.

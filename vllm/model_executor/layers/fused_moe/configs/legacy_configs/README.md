# Legacy Configurations

This directory contains older MoE kernel configurations that are maintained for backward compatibility.

## Purpose

These configurations serve as a fallback when:
- No version-specific configuration is available
- No default configuration exists in the main directory
- Running on older hardware or with older Triton versions

## Migration

When updating configurations:
1. Test new configurations thoroughly
2. Place updated configs in the appropriate version-specific folder or main directory
3. Keep the legacy version here for compatibility

## Note

These configurations may not be optimal for newer hardware or Triton versions. Consider running benchmarks to generate optimized configurations for your specific setup.
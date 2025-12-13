# Changelog

All notable changes to FluxFlow UI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.3.1] - 2025-12-13

### Note
- v0.3.0 was skipped due to release coordination issues with fluxflow-training
- All features from v0.3.0 are included in this release
- This is the first public release of fluxflow-ui on PyPI with CFG support

### Changed
- **Updated fluxflow-training dependency** from `>=0.3.0` to `>=0.3.1,<0.4.0`
  - Aligns with fluxflow-training v0.3.1 release
  - Ensures compatibility with latest core package (fluxflow>=0.3.1)

## [0.3.0] - 2025-12-12

### Added
- **Classifier-Free Guidance (CFG) Training Support**
  - New CFG accordion in training tab
  - `cfg_dropout_prob` slider (0.0-0.20, default 0.10) for null conditioning
  - Enables training models for guided generation
  - Recommended: 10% dropout for balanced CFG control
- **Classifier-Free Guidance (CFG) Generation Support**
  - CFG accordion in generation tab
  - `guidance_scale` slider (1.0-15.0, default 5.0) for prompt adherence control
  - Optional negative prompt support for better control over unwanted features
  - Requires model trained with `cfg_dropout_prob > 0`
  - Recommended guidance scale: 3-7 for most cases
- **Pipeline Training Mode**
  - Dual-tab interface: "Simple Mode" vs "Pipeline Mode"
  - YAML editor with syntax highlighting for pipeline configuration
  - Load/validate/save pipeline configs
  - Multi-step sequential training support
  - Per-step component freeze/unfreeze capabilities
  - Loss-threshold based transitions between steps
  - `pipeline_yaml_content` config parameter
  - Saves YAML to `outputs/<project>/pipeline_config.yaml`
  - Passes `--config` flag to fluxflow-train CLI
  - Full resume support mid-pipeline
- **Multi-Resolution Training**
  - New "Multi-Resolution Training" accordion
  - Comma-separated resolution stages (e.g., "256, 512, 768, 1024")
  - Progressive resolution training for improved convergence
  - Reduces initial VRAM requirements
  - `reduced_min_sizes` textbox parameter
- **Advanced Training Parameters**
  - `use_gradient_checkpointing` checkbox (saves VRAM at cost of speed)
  - `use_lpips` checkbox (perceptual loss, default ON)
  - `lambda_lpips` slider (0.0-1.0, default 0.1) for perceptual loss weight
  - Better control over memory vs. quality trade-offs
- **WebDataset Support**
  - `use_webdataset` parameter for streaming datasets
  - `webdataset_token` parameter for authentication
  - `webdataset_url` textbox for remote dataset URLs
  - `webdataset_image_key` parameter (default "jpg")
  - `webdataset_caption_key` parameter (default "prompt")
  - Enables efficient training on large remote datasets

### Changed
- **Updated dependencies**:
  - `fluxflow-training` from `>=0.1.0` to `>=0.3.0,<0.4.0`
  - Added `pyyaml>=6.0.0` for pipeline config parsing
- Enhanced config manager with all new v0.3.0 parameters
- Enhanced training runner with pipeline mode and YAML config support
- Enhanced generation worker with CFG-guided dual-pass sampling
- Improved UI layout with collapsible accordions for advanced features
- Training runner: New `_build_pipeline_command()` helper
- Training runner: Enhanced `_build_training_args()` with advanced params
- Training runner: Enhanced `_build_data_args()` with WebDataset params
- Generation worker: New `_generate_with_cfg()` method for dual-pass sampling

### Deprecated
- `use_tt2m` parameter (use `use_webdataset` instead)
- `tt2m_token` parameter (use `webdataset_token` instead)
- Legacy parameters still supported for backward compatibility with v0.1.x configs

### Fixed
- Training runner now properly handles pipeline YAML configuration files
- Generation worker correctly implements dual-pass CFG sampling
- Multi-resolution size parsing validates comma-separated integer lists
- Config manager properly serializes all new parameters to JSON

## [0.1.1] - 2024-11-XX

### Fixed
- Minor bug fixes and stability improvements

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of FluxFlow UI
- Basic training interface with VAE, GAN, and Flow modes
- Generation interface with prompt-based image synthesis
- Configuration management system (JSON-based)
- Training progress monitoring with live logs
- Sample generation during training checkpoints
- File browser for model/dataset selection
- Flask-based web server with Gradio integration

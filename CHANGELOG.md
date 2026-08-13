# Changelog

All notable changes to FluxFlow UI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Not yet released — work in progress toward v0.10.0.

### Added (Experimental)
- **AMD ROCm/gfx1151 support (experimental, unvalidated)**: `_get_device()`
  now delegates to `fluxflow.utils.device.get_device()`, which distinguishes
  ROCm from real NVIDIA CUDA. No UI-visible behavior change — ROCm-build
  PyTorch already routed through the existing CUDA code path. See
  `docs/ROCM.md` in fluxflow-core.
- NPU (XDNA) acceleration was evaluated and deferred; not implemented.

### Added
- `SELENIUM_SHORTCUTS.md` at repo root with happy-path QA flows for the v0.10.0
  generation worker (SH-01 smoke, SH-02 generation regression, SH-03 API-only).
- Narrow `assert X is not None` asserts in `app_flask.py` after each
  `request.json` access to satisfy mypy without changing runtime behavior
  (the `@require_json` decorator already guarantees a non-None JSON body).
- Optional "Text Encoder (optional override)" field on the Generate tab and
  a matching `text_encoder_path` field on `/api/generation/load`, letting
  you point at an explicit `text_encoder.safetensors` instead of the
  auto-discovered sibling file or the checkpoint's bundled copy.

### Changed
- Updated `generation_worker.py` to unpack `(text_seq, text_mask)` from the
  per-token `BertTextEncoder` and thread the tuple end-to-end into the flow
  processor, tracking the v0.10.0 bezier-coupled redesign in `fluxflow-core`.
- Built the CFG null branch via `fluxflow.utils.build_cfg_null_pair` (encoded
  empty prompt) instead of `torch.zeros_like` on the pooled vector, preventing
  NaN / black-image artifacts on the null pass.
- Threaded the per-token tuple end-to-end through Flask routes that pass
  conditioning to the flow processor; the external HTTP API was unchanged
  (text in, image out — same request and response shape) from v0.8.0.
- Documented that the salvage path for old checkpoints lives in
  `fluxflow-core`; see
  [MIGRATION-v0.10.0-redesign.md](https://github.com/danny-mio/fluxflow-core/blob/develop/docs/MIGRATION-v0.10.0-redesign.md).

### Fixed
- **v0.10.0 checkpoints were misdetected as v0.7.0** in the generation
  worker's legacy-loading fallback, causing them to load with mismatched
  v0.7.0 model classes (`strict=False` silently dropped most trained
  weights). Added `GenerationWorker._load_v100_fallback`, and the version
  heuristic now uses `fluxflow.models.detect_architecture_version`.
- The "Version X model" status message always showed a stale/default value
  (`getattr(pipeline, "version", ...)` — no such attribute is ever set) —
  now re-detected from the checkpoint's own state-dict keys for display.

## [0.8.0] - 2026-02-21

### Changed
- **Updated `fluxflow-training` dependency** to `>=0.8.0`
  - Enables use of the v0.8.0 pillar-attention flow architecture
  - No UI changes required; version routing handled by `load_versioned_checkpoint()`
- Version bumped to 0.8.0

### Fixed
- **Type annotation** for `GenerationWorker.tokenizer`: changed from `Optional[AutoTokenizer]` to `Any` to match the dynamic type returned by `AutoTokenizer.from_pretrained()`
- **Flaky test** in `TestStopTraining::test_terminates_process`: fixed race condition where the test assertion ran before the background thread had a chance to set `_is_running = False`
- **Black formatting** applied to `app.py`, `tabs/generation.py`, `tabs/training.py`

## [0.4.0] - 2025-12-17

### Fixed
- **CFG image dimension bug** (CRITICAL)
  - Fixed CFG-generated images always being 256x256 regardless of user-specified dimensions
  - Root cause: `_generate_with_cfg()` wasn't properly separating `hw_vec` (dimension info) from image latent
  - Solution: Properly extract and preserve `hw_vec` during CFG denoising loop
  - **Impact**: CFG now correctly respects user-chosen width and height settings
  - **Files**: `src/fluxflow_ui/utils/generation_worker.py`
  - Users can now generate CFG images at any resolution (512x512, 768x512, 1024x1024, etc.)

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

## [0.1.1] - 2024-11-01

### Fixed
- Minor bug fixes and stability improvements

## [0.1.0] - 2024-10-01

### Added
- Initial release of FluxFlow UI
- Basic training interface with VAE, GAN, and Flow modes
- Generation interface with prompt-based image synthesis
- Configuration management system (JSON-based)
- Training progress monitoring with live logs
- Sample generation during training checkpoints
- File browser for model/dataset selection
- Flask-based web server with Gradio integration

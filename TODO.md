# FluxFlow UI v0.3.0 Update Tasks

## Objective
Update fluxflow-ui to support all features from fluxflow-training v0.3.0

## Tasks

### 1. Update Dependencies (pyproject.toml)
- [ ] Bump version to 0.3.0
- [ ] Update fluxflow-training dependency to >=0.3.0,<0.4.0
- [ ] Add pyyaml>=6.0.0 for pipeline config support

### 2. Add CFG Training Support (tabs/training.py)
- [ ] Add CFG accordion with cfg_dropout_prob slider (0.0-0.20, default 0.10)
- [ ] Update config manager defaults
- [ ] Update training runner to pass cfg_dropout_prob

### 3. Add Pipeline Mode (tabs/training.py)
- [ ] Add Tabs: "Simple Mode" vs "Pipeline Mode (Advanced)"
- [ ] Add YAML editor with syntax highlighting
- [ ] Add load/validate/save YAML buttons
- [ ] Create example pipeline config helper function
- [ ] Update training runner to handle --config flag

### 4. Add Multi-Resolution Training (tabs/training.py)
- [ ] Add "Multi-Resolution Training" accordion
- [ ] Add reduced_min_sizes textbox (comma-separated)
- [ ] Parse and validate sizes in training runner

### 5. Add Advanced Parameters (tabs/training.py)
- [ ] Add use_gradient_checkpointing checkbox
- [ ] Add use_lpips checkbox
- [ ] Add lambda_lpips slider (0.0-1.0, default 0.1)
- [ ] Update training runner to pass new params

### 6. Fix Deprecated Parameters (tabs/training.py)
- [ ] Rename use_tt2m → use_webdataset
- [ ] Rename tt2m_token → webdataset_token
- [ ] Add webdataset_url textbox
- [ ] Add webdataset_image_key textbox (default "jpg")
- [ ] Add webdataset_caption_key textbox (default "prompt")
- [ ] Update config manager defaults

### 7. Add CFG Generation Support (tabs/generation.py + utils/generation_worker.py)
- [ ] Add CFG accordion in generation tab
- [ ] Add guidance_scale slider (1.0-15.0, default 5.0)
- [ ] Add negative_prompt textbox
- [ ] Implement cfg_guided_prediction in generation worker
- [ ] Import cfg_utils from fluxflow_training

### 8. Update Config Manager (utils/config_manager.py)
- [ ] Add cfg_dropout_prob to defaults (0.0)
- [ ] Add use_lpips, lambda_lpips to defaults
- [ ] Add use_gradient_checkpointing to defaults
- [ ] Add reduced_min_sizes to defaults
- [ ] Add webdataset params to defaults

### 9. Update Training Runner (utils/training_runner.py)
- [ ] Add _build_cfg_args() helper
- [ ] Add _build_advanced_args() helper for new params
- [ ] Add pipeline mode support (--config flag)
- [ ] Update _build_data_args() for webdataset params

### 10. Testing
- [ ] Test simple mode (existing workflow)
- [ ] Test pipeline mode (load/validate YAML)
- [ ] Test CFG training (verify param passed)
- [ ] Test multi-res (parse comma-separated sizes)
- [ ] Test CFG generation (guidance_scale)
- [ ] Run pytest tests/ -v

### 11. Documentation
- [ ] Update README.md with v0.3.0 features
- [ ] Update USER_GUIDE.md with pipeline mode instructions
- [ ] Add PIPELINE_EXAMPLES.md with sample YAML configs

## Validation Checklist
- [ ] All tests passing
- [ ] Simple mode works (backward compatible)
- [ ] Pipeline mode works (YAML config)
- [ ] CFG training works (dropout applied)
- [ ] CFG generation works (guidance applied)
- [ ] No lint errors (black, flake8)

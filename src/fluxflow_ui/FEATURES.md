# FluxFlow UI - Feature Documentation

## Overview

FluxFlow UI is a comprehensive web-based interface for training and generating images with the FluxFlow text-to-image model. Built with Gradio 4.0+, it provides an intuitive alternative to command-line training and generation.

## Architecture

### Core Components

```
ui/
├── app.py                       # Main Gradio application
├── tabs/
│   ├── training.py             # Training interface and controls
│   └── generation.py           # Image generation interface
├── utils/
│   ├── config_manager.py       # Configuration persistence
│   ├── training_runner.py      # Background training process manager
│   └── generation_worker.py    # Image generation worker
└── components/
    └── (future: custom Gradio components)
```

## Phase 1 Features (Implemented)

### Training Interface

#### Dataset Configuration
- **Data Path Input**: Select directory containing training images
- **Captions File Input**: Tab-separated file mapping images to captions
- **Output Path**: Directory for checkpoints and samples
- **Resume Checkpoint**: Optional path to continue training

#### Model Architecture
- **VAE Dimension**: Latent space dimension (32-256)
- **Feature Dimension**: Flow processor dimension (32-256)
- **Discriminator Dimension**: PatchDiscriminator features (32-256)

#### Training Parameters
- **Epochs**: Number of training epochs
- **Batch Size**: Samples per batch (usually 1 for limited VRAM)
- **Learning Rate**: Base learning rate for optimizers
- **Min LR Multiplier**: Minimum LR as fraction of base LR

#### Training Modes
- **Train VAE**: Enable VAE (compressor + expander) training
- **Disable GAN**: Train VAE without discriminator
- **Use SPADE**: Enable SPADE spatial conditioning
- **Train Flow**: Train diffusion model
- **Train Flow (Full)**: Full diffusion training schedule
- **Preserve LR**: Keep learning rates when resuming

#### Advanced Settings (Collapsible)
- **Adversarial Weight**: GAN loss weight (0.0-2.0)
- **KL Beta**: KL divergence weight (0.0-5.0)
- **KL Warmup Steps**: Gradual KL annealing steps
- **Sample Interval**: Batches between sample generation
- **Log Interval**: Batches between console logs
- **Workers**: Data loading worker threads

#### Controls
- **Start Training**: Launch training with current config
- **Stop Training**: Gracefully terminate training
- **Status Display**: Real-time status messages
- **Console Output**: Live streaming training logs (auto-scrolling)

#### Features
- ✅ Configuration persistence (auto-save/restore)
- ✅ Real-time log streaming
- ✅ Validation of inputs before training
- ✅ Background process management
- ✅ Graceful shutdown with checkpoint saving

### Generation Interface

#### Model Loading
- **Checkpoint Path**: Path to `.safetensors` model file
- **VAE Dimension**: Must match training config
- **Feature Dimension**: Must match training config
- **Text Embedding Dimension**: Usually 1024
- **Load Button**: Initialize model with config
- **Status Display**: Model loading feedback

#### Generation Parameters
- **Prompt Input**: Multi-line text prompt
- **Image Size**: Output resolution (256-1024, step 64)
- **Sampling Steps**: Diffusion steps (10-100)
- **Use Fixed Seed**: Enable reproducible generation
- **Seed Value**: Random seed when enabled

#### Controls
- **Generate Button**: Create image from prompt
- **Image Display**: Live preview of generated image
- **Status Display**: Generation progress/errors

#### Features
- ✅ Model caching (load once, generate many)
- ✅ Device auto-detection (CUDA/MPS/CPU)
- ✅ Dimension validation
- ✅ Error handling with user feedback
- ✅ Configuration persistence

### Global Features

#### Configuration Management
- Auto-save on every training/generation run
- Restore last-used settings on startup
- JSON-based storage (`.ui_configs/`)
- Separate configs for training and generation

#### User Experience
- Modern Gradio Soft theme
- Responsive layout (split panels)
- Helpful tooltips and guides
- Error messages with actionable feedback
- Real-time updates without page refresh

## Phase 2 Features (Planned)

### Training Enhancements
- [ ] Live metrics visualization (loss curves)
- [ ] Training graph integration
- [ ] Resource monitoring (GPU/CPU/RAM)
- [ ] Automatic best checkpoint selection
- [ ] Email/webhook notifications on completion

### Generation Enhancements
- [ ] Batch prompt generation with queue
- [ ] Image-to-image support
- [ ] Scheduler selection (DPM++, Euler, etc.)
- [ ] Negative prompts
- [ ] Generation history browser
- [ ] Side-by-side checkpoint comparison

### Model Management
- [ ] Checkpoint browser with previews
- [ ] Auto-config detection from checkpoints
- [ ] Model merging tools
- [ ] Checkpoint pruning/optimization

### Dataset Tools
- [ ] Dataset browser with thumbnails
- [ ] Caption editor
- [ ] Dimension cache viewer
- [ ] Data augmentation preview
- [ ] Dataset statistics dashboard

## Technical Details

### Training Runner
- Uses `subprocess.Popen` for background execution
- Streams stdout/stderr in real-time
- Thread-safe output queue
- Graceful termination with SIGTERM
- Command-line argument builder

### Generation Worker
- Lazy model loading (on-demand)
- PyTorch model caching
- Automatic device selection
- Memory-efficient inference
- Numpy array output for Gradio

### Configuration Manager
- JSON-based persistence
- Type-safe defaults
- Validation on load
- Atomic file writes
- Backward compatibility

## Performance Considerations

### Memory Optimization
- Models loaded only when needed
- Single model instance for multiple generations
- Output buffer size limits (500 lines)
- Automatic garbage collection

### UI Responsiveness
- Background training (non-blocking)
- Async log updates (1s interval)
- Lazy loading of components
- Minimal client-side JavaScript

### Scalability
- Support for long training runs (days)
- Handle large log outputs
- Multiple concurrent generations (future)
- Queue management for batch jobs (future)

## Browser Compatibility

Tested with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## API Endpoints (Gradio Auto-Generated)

Gradio automatically creates REST API endpoints:
- `/api/predict` - Generate images
- `/api/queue/join` - Join generation queue
- WebSocket for real-time updates

Access API docs at: `http://localhost:7860/docs`

## Security Considerations

- No authentication in current version (localhost only)
- File path validation to prevent directory traversal
- Process isolation for training
- No code execution from user input
- Recommend firewall rules for public deployment

## Future Integrations

- [ ] ComfyUI node compatibility
- [ ] Automatic1111 extension
- [ ] Discord bot integration
- [ ] REST API for external tools
- [ ] MLflow experiment tracking

---

**Version**: 1.0.0 (Phase 1 MVP)
**Last Updated**: November 2025

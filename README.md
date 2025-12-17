# FluxFlow UI

Web interface for FluxFlow text-to-image generation and training.

## 🚧 Model Availability Notice

**Training In Progress**: FluxFlow models are currently being trained. The UI is fully functional, but trained model checkpoints are not yet available for download.

**When Available**: Trained checkpoints will be published to [MODEL_ZOO.md](https://github.com/danny-mio/fluxflow-core/blob/main/MODEL_ZOO.md) upon completion of training validation.

**Current Capabilities**: You can use this UI to:
- Configure and launch training runs with your own datasets
- Monitor training progress in real-time
- Test the architecture with your own trained checkpoints

---

## Installation

> **Note**: This documentation describes **v0.3.0** (upcoming release). For the current stable version (v0.1.1), see [v0.1.1 documentation](https://github.com/danny-mio/fluxflow-ui/tree/v0.1.1).

### Prerequisites

**Required:**
- Python 3.10 or higher
- pip package manager
- 8GB+ RAM (16GB+ recommended)

**For GPU Training (Recommended):**
- **NVIDIA GPU**: CUDA 11.8+ with compatible drivers
- **Apple Silicon**: macOS 12.3+ (for MPS support)
- **GPU VRAM**: 8GB minimum, 16GB+ for high quality training

**Verify Prerequisites:**
```bash
python --version  # Should show 3.10 or higher
pip --version     # Should be installed
nvidia-smi        # (NVIDIA only) Should show GPU info
```

### Production Install (v0.1.1 - Current Stable)

```bash
pip install fluxflow-ui
```

**What gets installed:**
- `fluxflow-ui` - Web interface for training and generation
- `fluxflow-training` - Training capabilities (automatically installed as dependency)
- `fluxflow` core package (transitively installed)
- CLI command: `fluxflow-ui`

**Package available on PyPI**: [fluxflow-ui v0.1.1](https://pypi.org/project/fluxflow-ui/)

⚠️ **Note**: v0.1.1 does NOT include CFG (Classifier-Free Guidance) features described below. For CFG support, use development install.

### Development Install

```bash
git clone https://github.com/danny-mio/fluxflow-ui.git
cd fluxflow-ui
pip install -e ".[dev]"
```

## ⚠️ Security Warning

**FluxFlow UI is designed for local development use only.**

- No authentication or authorization
- File browser can access entire filesystem
- Not hardened for production deployment

See [SECURITY.md](SECURITY.md) for details on security measures, limitations, and production deployment warnings.

**Do not expose this application to the internet without additional security hardening.**

---

## Features

- **Training Interface**: Configure and monitor training runs
- **Generation Interface**: Generate images with various parameters
- **Real-time Progress**: Monitor training progress with live updates
- **Model Management**: Load and manage checkpoints
- **Interactive Controls**: Adjust generation parameters in real-time

## Quick Start

### Launch the Web UI

FluxFlow UI supports two interfaces:

**Flask (Primary - Recommended):**
```bash
fluxflow-ui
```

**Gradio (Alternative):**
```bash
python -m fluxflow_ui.app
```

Then open your browser to `http://localhost:7860`

**Note:** Flask is the primary interface with full features. Gradio is provided as an alternative but may have limited functionality.

### Features

#### Training Tab
- Configure training parameters
- Start/stop training runs
- Monitor loss curves and metrics
- View sample generations during training

#### Generation Tab
- Load trained models
- Generate images from text prompts
- Adjust sampling parameters
- Batch generation support

## Classifier-Free Guidance (CFG)

**✨ New in v0.3.0** (upcoming release): FluxFlow UI supports training and generation with Classifier-Free Guidance.

### Training with CFG

To train models with CFG support:

1. Navigate to the **Training** tab
2. Expand the **CFG Training** section
3. Set `cfg_dropout_prob` between 0.0-0.20 (recommended: 0.10-0.15)
   - This randomly drops text conditioning during training
   - Higher values = stronger CFG effect but may reduce unconditional quality
   - Set to 0.0 to disable CFG training

### Generating with CFG

To use CFG during generation:

1. Navigate to the **Generation** tab
2. Load a checkpoint trained with `cfg_dropout_prob > 0`
3. Expand the **CFG Settings** section
4. Enable CFG and set parameters:
   - **Enable CFG**: Toggle on
   - **Guidance Scale**: 1.0-15.0 (recommended: 3.0-7.0)
     - 1.0 = no guidance
     - 3.0-7.0 = balanced quality/creativity
     - 7.0-15.0 = strong guidance (may oversaturate)
   - **Negative Prompt** (optional): Text to avoid in generation

**Note**: CFG requires 2× forward passes per sampling step, doubling generation time.

### CFG Benefits

- **Better prompt adherence**: Images follow text descriptions more closely
- **Higher quality**: Improved coherence and detail
- **Negative prompts**: Ability to steer away from unwanted features
- **Flexible control**: Adjust guidance strength per generation

## Package Contents

- `fluxflow_ui.tabs` - UI tab implementations
- `fluxflow_ui.utils` - Config management and training runners
- `fluxflow_ui.templates` - HTML templates
- `fluxflow_ui.static` - CSS and JavaScript assets

## Configuration

The UI runs on `http://0.0.0.0:7860` by default. To customize the host and port, modify the `main()` function in `src/fluxflow_ui/app_flask.py`.

## Development

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-ui)
- [Security Policy](SECURITY.md)
- [User Guide](docs/USER_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

MIT License - see LICENSE file for details.

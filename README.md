# FluxFlow UI

Web interface for FluxFlow text-to-image generation and training.

## 🚧 Model Availability Notice

**Training In Progress**: FluxFlow models are currently being trained. The UI is fully functional, but trained model checkpoints are not yet available for download.

**When Available**: Trained checkpoints will be published to [MODEL_ZOO.md](../MODEL_ZOO.md) upon completion of the [TRAINING_VALIDATION_PLAN.md](../TRAINING_VALIDATION_PLAN.md).

**Current Capabilities**: You can use this UI to:
- Configure and launch training runs with your own datasets
- Monitor training progress in real-time
- Test the architecture with your own trained checkpoints

---

## Installation

### Development Installation (Recommended)

```bash
# Clone and install in editable mode with development dependencies
pip install -e ".[dev]"
```

**Note:** This package is not published to PyPI. Install from source only.

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

## Package Contents

- `fluxflow_ui.tabs` - UI tab implementations
- `fluxflow_ui.utils` - Config management and training runners
- `fluxflow_ui.templates` - HTML templates
- `fluxflow_ui.static` - CSS and JavaScript assets

## Configuration

The UI runs on `http://0.0.0.0:7860` by default. To customize the host and port, modify the `main()` function in `app_flask.py`.

## Security Warning

⚠️ **FluxFlow UI is designed for local development use only.**

- No authentication or authorization
- File browser can access entire filesystem
- Not hardened for production deployment

See [SECURITY.md](SECURITY.md) for details on security measures, limitations, and production deployment warnings.

**Do not expose this application to the internet without additional security hardening.**

## Development

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-ui)
- [Security Policy](SECURITY.md)
- [User Guide](USER_GUIDE.md)

## License

MIT License - see LICENSE file for details.

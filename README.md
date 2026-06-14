# FluxFlow UI

Web interface for FluxFlow text-to-image generation and training.

## v0.10.0 — Internal text-path rewiring (no API change)

v0.10.0 tracks the **bezier-coupled redesign** shipped in `fluxflow-core` /
`fluxflow-training`. The change is **internal-only** — your requests do not
change:

- `BertTextEncoder` now returns a per-token tuple `(text_seq, text_mask)`; the
  generation worker unpacks it and threads the per-token signal end-to-end into
  the flow processor.
- The CFG null branch uses an encoded empty prompt
  (`build_cfg_null_pair`) instead of `torch.zeros_like` on the pooled vector,
  preventing NaN / black-image artifacts.

The Flask HTTP API (`/api/generation/load`, `/api/generation/generate`, etc.)
is unchanged from v0.8.0 — same request payload, same response shape. End users
have nothing to migrate on the UI side.

For full context on the redesign and a checkpoint salvage path, see
[fluxflow-core MIGRATION-v0.10.0-redesign.md](https://github.com/danny-mio/fluxflow-core/blob/develop/docs/MIGRATION-v0.10.0-redesign.md).

## Model Availability Notice

**Training In Progress**: FluxFlow models are currently being trained. The UI is fully functional, but trained model checkpoints are not yet available for download.

**When Available**: Trained checkpoints will be published to [MODEL_ZOO.md](https://github.com/danny-mio/fluxflow-core/blob/main/MODEL_ZOO.md) upon completion of training validation.

**Current Capabilities**: You can use this UI to:
- Configure and launch training runs with your own datasets
- Monitor training progress in real-time
- Test the architecture with your own trained checkpoints

---

## Installation

> **Note**: This documentation describes **v0.10.0** (current development line).
> For the previous PyPI-published stable version (v0.8.0), see
> [v0.8.0 documentation](https://github.com/danny-mio/fluxflow-ui/tree/v0.8.0).

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

### Production Install (latest PyPI release)

```bash
pip install fluxflow-ui
```

**What gets installed:**
- `fluxflow-ui` - Web interface for training and generation
- `fluxflow-training` - Training capabilities (automatically installed as dependency)
- `fluxflow` core package (transitively installed)
- CLI command: `fluxflow-ui`

**Package on PyPI**: [fluxflow-ui](https://pypi.org/project/fluxflow-ui/) —
the v0.8.0 release is the current PyPI line and supports the pillar-attention
flow architecture introduced in v0.8.0. v0.10.0 (this branch) is still on the
development line; it tracks `fluxflow-training` from the matching
`feature/model-v0.10.0` branch and will move to `>=0.10.0` once that core
release lands on PyPI. The change is internal-API only relative to v0.8.0
(no HTTP surface change).

### Development Install

```bash
git clone https://github.com/danny-mio/fluxflow-ui.git
cd fluxflow-ui
pip install -e ".[dev]"
```

## Security Warning

**FluxFlow UI is designed for local development use only.**

- No authentication or authorization
- File browser can access entire filesystem
- Not hardened for production deployment

See [SECURITY.md](SECURITY.md) for details on security measures, limitations, and production deployment warnings.

**Do not expose this application to the internet without additional security hardening.**

---

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

## Classifier-Free Guidance (CFG)

**Available since v0.3.0**: FluxFlow UI supports training and generation with Classifier-Free Guidance.

### Training with CFG

To train models with CFG support:

1. Navigate to the **Training Pipeline** tab
2. Expand the **CFG Training** section
3. Set `cfg_dropout_prob` between 0.0-0.20 (recommended: 0.10-0.15)
   - This randomly drops text conditioning during training
   - Higher values = stronger CFG effect but may reduce unconditional quality
   - Set to 0.0 to disable CFG training

### Generating with CFG

To use CFG during generation:

1. Navigate to the **Generation Studio** tab
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
- [User Guide](docs/USER_GUIDE.md) (includes a Troubleshooting section)

## License

MIT License - see LICENSE file for details.

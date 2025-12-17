# FluxFlow UI

A user-friendly web interface for training and generating images with FluxFlow.

## Features

### Training Interface
- **Dataset Configuration**: Easy selection of images and captions
- **Model Architecture**: Configure VAE dimensions, feature maps, discriminator settings
- **Training Modes**: Toggle VAE, GAN, SPADE, and Flow training
- **Real-time Monitoring**: Live console output with training progress
- **Resume Support**: Continue training from checkpoints with preserved learning rates
- **Advanced Settings**: Control adversarial weight, KL divergence, sampling intervals

### Generation Interface
- **Model Loading**: Load trained checkpoints with auto-configuration
- **Text-to-Image**: Generate images from text prompts
- **Parameter Control**: Adjust image size, sampling steps, and random seeds
- **Reproducibility**: Fixed seed support for consistent results
- **Live Preview**: Instant image display after generation

## Installation

1. Install UI dependencies:
```bash
pip install -r ui/requirements_ui.txt
```text
1. Ensure FluxFlow base requirements are installed:
```bash
pip install -r requirements.txt
```text
1. Download tokenizer cache (first time only):
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./_cache')"
```text
## Quick Start

### Launch the UI

```bash
python ui/app.py
```text
The interface will open at `http://localhost:7860`

### Training a Model

1. Go to the **Train** tab
1. Configure dataset:
   - **Data Path**: Directory containing your images
   - **Captions File**: Tab-separated file with image filenames and captions
1. Set training parameters:
   - **Epochs**: Number of training epochs (start with 10)
   - **Batch Size**: Usually 1 for limited VRAM
   - **Learning Rate**: Default 1e-5 works well
1. Choose training modes:
   - **Train VAE**: Enable for initial training (recommended)
   - **Use SPADE**: Enable for better quality
   - **Train Flow**: Enable after VAE is trained
1. Click **Start Training**
1. Monitor progress in the console output

### Generating Images

1. Go to the **Generate** tab
1. Load a trained model:
   - **Model Checkpoint**: Path to `.safetensors` file
   - **VAE Dimension**: Match your training config (e.g., 64)
   - **Feature Dimension**: Match your training config (e.g., 64)
1. Click **Load Model**
1. Enter a text prompt (e.g., "A beautiful sunset over mountains")
1. Adjust generation parameters:
   - **Image Size**: 512 is recommended
   - **Sampling Steps**: 50 for good quality
   - **Seed**: Enable for reproducible results
1. Click **Generate Image**

## Tips and Best Practices

### Training

**For Limited VRAM (8GB):**
- VAE Dim: 32-64
- Feature Dim: 32-64
- Batch Size: 1

**For Better Quality (24GB+ VRAM):**
- VAE Dim: 128-256
- Feature Dim: 128-256
- Batch Size: 2-4

**Recommended Training Flow:**
1. Train VAE only for 50-100 epochs with SPADE enabled
1. Then train Flow model with `--train_diff` or `--train_diff_full`
1. Monitor loss values in console

**Resume Training:**
- Enter checkpoint path in "Resume from Checkpoint"
- Enable "Preserve LR" to keep learning rates
- Training will automatically resume from last saved position

### Generation

**Fast Generation:**
- Use smaller image sizes (256-384)
- Reduce sampling steps to 20-30

**High Quality:**
- Use 512+ image size
- Use 50+ sampling steps
- Enable fixed seed for iteration

**Troubleshooting:**
- If generation fails, check model dimensions match training config
- Ensure checkpoint contains both diffuser and text_encoder weights
- Try reloading the model

## Configuration Persistence

The UI automatically saves your last used configuration:
- Training config: `.ui_configs/training_config.json`
- Generation config: `.ui_configs/generation_config.json`

Configs are restored when you reopen the UI.

## Advanced Features

### Console Output
- Live streaming of training logs
- Auto-scrolls to latest output
- Keeps last 500 lines

### Training Control
- **Start Training**: Begins training with current config
- **Stop Training**: Gracefully stops training (saves checkpoint)

### Model Checkpoints
Training automatically saves:
- `flxflow_final.safetensors`: Main model
- `text_encoder.safetensors`: Text encoder
- `disc_img.safetensors`: Discriminator
- `training_state.json`: Resume state
- Sample images at intervals

## Architecture

```text
ui/
├── app.py                    # Main application
├── tabs/
│   ├── training.py          # Training interface
│   └── generation.py        # Generation interface
├── utils/
│   ├── config_manager.py    # Config persistence
│   ├── training_runner.py   # Background training
│   └── generation_worker.py # Image generation
└── requirements_ui.txt      # UI dependencies
```text
## Troubleshooting

### UI won't start
- Check Gradio is installed: `pip install gradio`
- Verify port 7860 is available
- Try: `python ui/app.py --server-port 7861`

### Training won't start
- Verify data path and captions file exist
- Check at least one training mode is enabled
- Ensure CUDA/MPS is available for GPU training

### Generation fails
- Verify checkpoint path is correct
- Match dimensions to training config
- Check model loaded successfully (status message)

### Out of Memory
- Reduce batch size to 1
- Reduce model dimensions (32-64)
- Use CPU mode (slower but works)

## Future Enhancements

Planned features:
- Live training metrics visualization
- Batch prompt generation
- Model comparison tools
- Dataset browser
- Training graph integration

## Support

For issues and questions:
- Check `TROUBLESHOOTING.md` in project root
- Review training logs in console output
- Ensure all dependencies are installed

## License

Same as FluxFlow project (MIT)

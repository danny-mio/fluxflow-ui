# FluxFlow UI - User Guide

**Version**: 0.3.0  
**Last Updated**: 2025-12-12

Complete guide for using FluxFlow UI to train and generate images with text-to-image models.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Image Generation](#image-generation)
3. [Model Training](#model-training)
4. [File Browser](#file-browser)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- FluxFlow UI installed (`pip install fluxflow-ui`)
- For GPU training: CUDA or MPS (Apple Silicon) support

### Launching the Interface

```bash
fluxflow-ui
```

Open your browser to: **http://localhost:7860**

You'll see two main tabs:
- **Generation Studio** - Create images from text prompts
- **Training Pipeline** - Train your own models

---

## Image Generation

Generate images from text prompts using trained FluxFlow models.

### Basic Generation Workflow

#### 1. Load a Model

**Navigate to Generation Studio tab** (first tab)

**Load your trained checkpoint:**
1. Click **Browse** button next to "Model Checkpoint"
2. Navigate to your `.safetensors` checkpoint file
3. Select the file (e.g., `flxflow_final.safetensors`)
4. Click **Load Model** button

**What happens:**
- UI automatically detects model dimensions
- Shows confirmation: `"Model loaded (VAE=32, Feature=32)"` or similar
- Takes 2-5 seconds on typical hardware

**Supported checkpoints:**
- `.safetensors` files from FluxFlow training
- Must contain both `diffuser.*` and `text_encoder.*` weights

#### 2. Enter Your Prompt

**In the text prompt field, type what you want to generate:**

```
A serene mountain landscape at sunset, golden hour lighting, photorealistic, 4K
```

**Tips for good prompts:**
- Be descriptive and specific
- Include style keywords (photorealistic, illustration, painting, etc.)
- Mention lighting, mood, composition
- 10-50 words works well

#### 3. Configure Generation Settings

**Image Dimensions:**
- Width/Height: 256-1024 pixels
- Must be multiples of 16
- Recommended: 512×512 (balanced), 256×256 (fast), 1024×1024 (detailed)

**Sampling Steps:**
- Range: 10-100 steps
- Recommended: 20-50 steps
- More steps = higher quality but slower
- 10 steps: Fast preview
- 50 steps: Production quality

**Random Seed:**
- Leave unchecked for random results
- Check and set seed for reproducible generation
- Use same seed + prompt = same image

#### 4. Generate!

Click **Generate Image** button

**What happens:**
- Generation takes 3-30 seconds depending on:
  - Image size (256×256 = fast, 1024×1024 = slow)
  - Sampling steps (10 = fast, 100 = slow)
  - Hardware (GPU > MPS > CPU)
- Status message shows progress
- Generated image appears below

**Example timing** (Apple M1 Max, 512×512, 50 steps):
- Generation: ~8 seconds
- With CFG: ~15 seconds (2× forward passes)

---

### Advanced: Classifier-Free Guidance (CFG)

**CFG improves image quality and prompt adherence.**

#### When to Use CFG

✅ **Use CFG when you want:**
- Better prompt adherence
- Higher quality/detail
- Avoid specific features (negative prompts)
- More control over output

❌ **Skip CFG when:**
- Model wasn't trained with `cfg_dropout_prob > 0`
- Speed is critical (CFG is 2× slower)
- Exploring creative variations

#### Using CFG

1. **Expand CFG Settings** section (click to open)
2. **Enable CFG** checkbox ☑️
3. **Set Guidance Scale:**
   - 1.0 = No guidance (standard generation)
   - 3.0-5.0 = Subtle improvement (recommended start)
   - 5.0-7.0 = Strong guidance (balanced)
   - 7.0-15.0 = Very strong (may oversaturate)
4. **(Optional) Add Negative Prompt:**
   ```
   blurry, low quality, distorted, ugly, bad anatomy, watermark
   ```
5. Click **Generate Image**

#### CFG Examples

**Without CFG** (guidance_scale=1.0):
- Natural, varied results
- Faster generation
- May miss some prompt details

**With CFG** (guidance_scale=5.0):
- Closer to prompt description
- Better detail and coherence
- Slightly slower

**Strong CFG** (guidance_scale=10.0):
- Very literal interpretation
- High detail
- Risk of oversaturation/artifacts

---

## Model Training

Train your own text-to-image models on custom datasets.

### Training Workflow Overview

1. Prepare dataset (images + captions)
2. Configure training parameters
3. Start training
4. Monitor progress
5. Stop when satisfied
6. Use checkpoint for generation

### 1. Prepare Your Dataset

**Image Dataset:**
- Format: JPG, PNG
- Recommended: 10,000+ images for good results
- Minimum: 1,000 images for testing
- All images in one directory

**Captions File:**
- Format: Plain text file (`.txt`)
- One caption per line
- Same order as images (alphabetically sorted)
- Example:
  ```
  A red car on a city street
  Mountain landscape with snow
  Portrait of a person wearing glasses
  ```

**Dataset Structure:**
```
/my-dataset/
  ├── image_0001.jpg
  ├── image_0002.jpg
  ├── image_0003.jpg
  └── ...

/my-dataset/captions.txt  (outside or inside folder)
```

### 2. Configure Training

**Navigate to Training Pipeline tab** (second tab)

#### Dataset Configuration

1. **Dataset Path:**
   - Click **Browse** → navigate to image folder
   - Or manually type path: `/path/to/images`

2. **Captions File:**
   - Click **Browse** → select `.txt` file
   - Or type path: `/path/to/captions.txt`

3. **Output Path:**
   - Click **Browse** → choose output folder
   - Default: `outputs/flux`
   - Checkpoints saved here

#### Model Architecture

**VAE Dimension:**
- Controls latent space size
- Options: 32, 64, 128, 256
- Recommended:
  - 32: Fast training, lower quality (testing)
  - 64: Balanced (recommended start)
  - 128/256: High quality (requires more VRAM)

**Feature Maps Dimension:**
- Controls flow processor capacity
- Options: 32, 64, 128, 256
- Recommended: Same as VAE dimension
- Higher = more capacity but slower

**Text Embedding Dimension:**
- Default: 1024 (DistilBERT)
- Don't change unless using different text encoder

#### Training Parameters

**Epochs:**
- Number of passes through entire dataset
- Recommended:
  - 1-3 epochs: Quick experiment
  - 5-10 epochs: Production training
  - 10+ epochs: Large dataset

**Batch Size:**
- Images processed simultaneously
- Constrained by VRAM
- Recommended:
  - 1: Limited VRAM (<8GB)
  - 2-4: Moderate VRAM (8-16GB)
  - 4-8: High VRAM (16GB+)

**Training Mode Checkboxes:**

☑️ **Train VAE:**
- Trains the encoder/decoder
- Enable for: New models, improving reconstruction
- Disable for: Fine-tuning only flow processor

☑️ **Train SPADE:**
- Enables SPADE normalization training
- Improves quality
- Recommended: Enable

#### Advanced Settings (Optional)

**Learning Rate:** Default 1e-5 (usually good)
**Workers:** Number of data loading threads (default 4)

### 3. Start Training

Click **Start Training** button

**What happens:**
1. **Dimension Scanning** (first time):
   - Scans all images to cache dimensions
   - Builds multi-resolution dataset
   - Progress: `Scanning: 10% | 5000/50000 [00:30<04:15]`
   - Large datasets: 10-20 minutes
   - Cached after first run (much faster next time)

2. **Training Begins:**
   - Console shows real-time progress
   - Metrics: Loss, step time, ETA
   - Example output:
     ```
     Epoch 1/5 | Step 100/5000 | Loss: 0.045 | 0.5s/step
     ```

3. **Checkpoints Saved:**
   - Periodic saves (every N steps)
   - Final checkpoint: `flxflow_final.safetensors`
   - Location: Your output path

### 4. Monitor Progress

**Training Console:**
- Real-time updates every second
- Shows current step, loss, speed
- Scroll to see full history

**What to Watch:**
- Loss decreasing over time = learning
- Steps/second = training speed
- ETA = estimated time remaining

**Normal Progress:**
- Loss starts high (0.1-0.5)
- Gradually decreases
- May plateau then drop again
- Final loss: 0.01-0.05 (typical)

### 5. Stop Training

Click **Stop Training** button

**When to stop:**
- ✅ Reached desired epochs
- ✅ Loss plateaued for long time
- ✅ Sample images look good
- ✅ Out of time/patience

**Checkpoint saved:**
- `{output_path}/flxflow_final.safetensors`
- Use this for generation!

---

## File Browser

Navigate server filesystem to select files and directories.

### Using the File Browser

1. **Click Browse button** next to any path field
2. **Modal opens** showing current directory
3. **Navigate:**
   - Click folders to enter
   - Click `..` to go up one level
   - Scroll to see all items
4. **Select:**
   - Click file/folder to select
   - Path auto-fills in field
   - Modal closes automatically
5. **Close:** Press **ESC** key or click **X**

### File Types Shown

- **Folders:** Blue folder icon
- **SafeTensors:** `.safetensors` (model checkpoints)
- **Images:** `.jpg`, `.png`, `.jpeg`
- **Text files:** `.txt`, `.md`
- **YAML files:** `.yaml`, `.yml`

**Note:** File browser shows **server** filesystem, not your local computer (if running remotely).

---

## Tips & Best Practices

### Generation Tips

**For Best Results:**
1. Use 512×512 or higher resolution
2. 30-50 sampling steps
3. Enable CFG with guidance 5.0-7.0
4. Descriptive prompts (20-40 words)
5. Use negative prompts to avoid artifacts

**Common Prompt Patterns:**
```
[subject], [style], [lighting], [quality modifiers]

Examples:
"Portrait of a woman, oil painting, soft lighting, detailed, high quality"
"Futuristic city, cyberpunk style, neon lights, 4K, photorealistic"
"Mountain landscape, golden hour, misty atmosphere, professional photography"
```

**Negative Prompt Suggestions:**
```
blurry, low quality, distorted, bad anatomy, 
watermark, text, signature, cropped, ugly
```

### Training Tips

**Dataset Quality:**
- Higher quality images = better model
- Consistent style helps
- Diverse subjects help generalization
- Good captions critical for prompt adherence

**Caption Writing:**
- Be descriptive but concise
- Include: Subject, style, setting, mood
- 10-30 words per caption
- Use consistent terminology

**Training Strategy:**

**Quick Test** (1 hour):
- 1,000 images
- 1-2 epochs
- VAE dim 32
- Batch size 1

**Production Training** (1 day):
- 10,000+ images
- 5-10 epochs
- VAE dim 64-128
- Batch size 2-4

**High Quality** (3-7 days):
- 50,000+ images
- 10-20 epochs
- VAE dim 128-256
- Batch size 4-8

### Hardware Recommendations

**Minimum (Testing):**
- 8GB RAM
- 4GB VRAM or Apple Silicon M1
- CPU training possible but slow

**Recommended (Development):**
- 16GB RAM
- 8-12GB VRAM or M1 Pro/Max
- SSD for dataset storage

**Optimal (Production):**
- 32GB+ RAM
- 16GB+ VRAM or M1 Ultra
- NVMe SSD for fast data loading

---

## Troubleshooting

### Generation Issues

**"Model not loaded" error:**
- Make sure you clicked "Load Model" button
- Check checkpoint file exists and is valid
- Verify file is `.safetensors` format

**"Width and height must be multiples of 16":**
- Adjust dimensions to: 256, 512, 768, 1024, etc.
- Not: 500, 600, 750 (invalid)

**Image is all noise:**
- Model may not be trained
- Try different checkpoint
- Check sampling steps (increase to 50)

**CFG not working:**
- Model must be trained with `cfg_dropout_prob > 0`
- Enable CFG checkbox
- Set guidance scale > 1.0

### Training Issues

**"No images found":**
- Check dataset path is correct
- Verify images are JPG/PNG
- Check folder permissions

**"Captions file not found":**
- Verify captions file path
- Check file extension is `.txt`
- Ensure file is readable

**Training crashes:**
- Reduce batch size (try 1)
- Reduce VAE/feature dimension to 32
- Check VRAM usage (may be out of memory)
- Update GPU drivers

**Dimension scanning very slow:**
- Normal for large datasets (100K+ images)
- Progress shown: `Scanning: 5% | 5000/100000`
- Only slow first time (cached after)
- Can't be skipped (required for multi-resolution training)

**"Leaked semaphore" warning:**
- Harmless Python multiprocessing cleanup warning
- Common on macOS (Apple Silicon)
- Does NOT prevent training from working
- Training continues normally

### Performance Issues

**Generation is slow:**
- Normal on CPU (30-60 seconds)
- Use GPU/MPS for 5-10× speedup
- CFG doubles generation time (expected)
- Reduce sampling steps for faster preview

**Training is slow:**
- Check batch size (higher = faster per-image)
- Reduce image resolution in dataset
- Use GPU instead of CPU
- Disable SPADE if not needed (slightly faster)

**Out of memory:**
- Reduce batch size to 1
- Reduce VAE dimension to 32
- Close other applications
- Use smaller images in dataset

### UI Issues

**Can't click buttons:**
- Refresh page (Ctrl+R or Cmd+R)
- Clear browser cache
- Try different browser (Chrome recommended)

**File browser doesn't open:**
- Check JavaScript is enabled
- Try pressing ESC then clicking Browse again
- Refresh page

**Console not updating:**
- Training may have crashed (check terminal)
- Refresh page to reconnect
- Check Flask server is still running

---

## Next Steps

### After Training Your First Model

1. **Test Generation:**
   - Load your checkpoint
   - Try various prompts
   - Experiment with CFG settings
   - Compare with/without CFG

2. **Evaluate Quality:**
   - Do images match prompts?
   - Is quality acceptable?
   - Any common artifacts?

3. **Iterate:**
   - More epochs if underfit
   - Better captions if prompt adherence poor
   - More data if quality insufficient
   - Adjust architecture if needed

### Advanced Topics

**Not Covered in This Guide:**
- Pipeline YAML configuration (multi-step training)
- Custom optimizer/scheduler configs
- LPIPS perceptual loss
- WebDataset streaming
- Distributed training

**See:**
- Advanced training features require YAML configs
- Feature planned for future UI update
- Current UI: Simple single-step training only

---

## Support & Resources

**Documentation:**
- [README.md](../README.md) - Quick start
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed troubleshooting
- [SECURITY.md](../SECURITY.md) - Security considerations

**Examples:**
- Screenshot gallery: [docs/assets/](assets/)
- Example configs: See fluxflow-training repository

**Issues:**
- Report bugs: [GitHub Issues](https://github.com/danny-mio/fluxflow-ui/issues)
- Feature requests welcome

**Community:**
- Discussions: GitHub Discussions (coming soon)

---

**Enjoy creating with FluxFlow!** 🎨

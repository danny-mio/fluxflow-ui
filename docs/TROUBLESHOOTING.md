# FluxFlow UI - Troubleshooting Guide

Common issues and solutions for FluxFlow UI.

---

## Quick Diagnosis

**Problem Category:**
- [Generation/Inference Issues](#generation-issues)
- [Training Issues](#training-issues)
- [Installation/Setup Issues](#installation-issues)
- [Performance Issues](#performance-issues)
- [UI/Browser Issues](#ui-issues)

---

## Generation Issues

### Model Won't Load

**Symptom:** "Model not loaded" error when clicking Generate

**Causes & Solutions:**

1. **Didn't click "Load Model" button**
   - ✅ Click the "Load Model" button after selecting checkpoint
   - ℹ️ Path field fills automatically, but model isn't loaded until you click button

2. **Invalid checkpoint file**
   - ✅ Verify file is `.safetensors` format
   - ✅ Check file isn't corrupted (re-download if needed)
   - ✅ Ensure checkpoint contains both `diffuser.*` and `text_encoder.*` weights

3. **Wrong file path**
   - ✅ Use file browser to select file (avoid typos)
   - ✅ Check file actually exists at specified path
   - ✅ Verify file permissions (readable)

4. **Dimension mismatch**
   - ℹ️ UI should auto-detect dimensions
   - ✅ If fails, manually verify VAE/feature dims match checkpoint

**Verify fix:**
```
Status should show: "Model loaded (VAE=XX, Feature=YY)"
```

---

### Generated Images Are Noise/Random

**Symptom:** Output is pure noise or random pixels

**Causes & Solutions:**

1. **Model not trained**
   - ✅ Verify checkpoint is from completed training run
   - ✅ Check training loss was decreasing (not stuck)
   - ℹ️ Untrained/early checkpoints produce noise

2. **Too few sampling steps**
   - ✅ Increase steps to 50+ (try 50-100)
   - ℹ️ 10 steps may not be enough for convergence

3. **Incorrect model dimensions**
   - ✅ Verify VAE/feature dims match training config
   - ✅ Re-load model and check auto-detected dimensions

4. **Corrupted checkpoint**
   - ✅ Try different checkpoint from same training run
   - ✅ Re-train if all checkpoints corrupted

**Quick test:**
```
- Steps: 50
- Dimensions: 512×512
- Seed: 42 (fixed, for reproducibility)
- Prompt: "simple test image"
```

---

### CFG Not Working / No Effect

**Symptom:** Enabling CFG doesn't improve quality

**Causes & Solutions:**

1. **Model not trained with CFG**
   - ❌ Model must be trained with `cfg_dropout_prob > 0`
   - ✅ Retrain model with CFG enabled (dropout 0.10-0.15)
   - ℹ️ Cannot add CFG to already-trained model

2. **Guidance scale too low**
   - ✅ Try guidance scale 5.0-7.0 (not 1.0-2.0)
   - ℹ️ Scale 1.0 = no guidance (same as disabled)

3. **CFG not enabled**
   - ✅ Check CFG checkbox is actually checked ☑️
   - ✅ Verify guidance scale is set

**Verify CFG working:**
```
Generate same image twice:
1. CFG disabled → Image A
2. CFG enabled (scale 7.0) → Image B
Images should be noticeably different
```

---

### "Width and height must be multiples of 16"

**Symptom:** Error when clicking Generate

**Cause:** Image dimensions not divisible by 16

**Solution:**
✅ Use dimensions from this list:
- 256×256 (fast)
- 512×512 (recommended)
- 768×768
- 1024×1024 (high quality)
- 512×768 (portrait)
- 768×512 (landscape)

❌ Invalid dimensions:
- 500×500
- 600×800
- 720×480

**Why:** VAE encodes images at 1/16th resolution. Dimensions must be divisible by 16 for proper encoding/decoding.

---

## Training Issues

### "No images found" Error

**Symptom:** Training fails immediately with "No images found"

**Causes & Solutions:**

1. **Wrong path**
   - ✅ Verify path points to folder containing images
   - ✅ Use file browser to avoid typos
   - ℹ️ Path should be `/path/to/folder/`, not `/path/to/folder/image.jpg`

2. **No supported image formats**
   - ✅ Supported: `.jpg`, `.jpeg`, `.png`
   - ❌ Not supported: `.bmp`, `.gif`, `.webp`, `.tif`
   - ✅ Convert images to JPG/PNG if needed

3. **Permission denied**
   - ✅ Check folder is readable
   - ✅ On macOS: Grant Terminal/Python "Files and Folders" permission in System Preferences

4. **Symlinks or network drives**
   - ℹ️ Some network/remote paths may not work
   - ✅ Try copying dataset to local disk

---

### "Captions file not found"

**Symptom:** Training fails with captions error

**Causes & Solutions:**

1. **Wrong path**
   - ✅ Verify file exists at specified path
   - ✅ Use file browser to select file

2. **Wrong extension**
   - ✅ Must be `.txt` file
   - ❌ Not `.csv`, `.json`, or other formats
   - ✅ Convert to plain text if needed

3. **File format issues**
   - ✅ One caption per line
   - ✅ Same number of captions as images
   - ✅ UTF-8 encoding (not UTF-16 or other)

**Verify captions file:**
```bash
# Count lines in captions file
wc -l /path/to/captions.txt

# Count images in folder
ls /path/to/images/*.jpg | wc -l

# Numbers should match!
```

---

### Training Crashes / Out of Memory

**Symptom:** Training starts then crashes with CUDA/MPS error

**Causes & Solutions:**

1. **Batch size too large**
   - ✅ Reduce batch size to 1
   - ℹ️ Batch size limited by VRAM
   - Recommended: 1 (<8GB VRAM), 2-4 (8-16GB), 4-8 (16GB+)

2. **Model dimensions too large**
   - ✅ Reduce VAE dim to 32
   - ✅ Reduce feature dims to 32
   - ℹ️ Can increase later if stable

3. **Other GPU processes**
   - ✅ Close other GPU-using applications
   - ✅ Check `nvidia-smi` or Activity Monitor

4. **Insufficient system RAM**
   - ✅ Close browser tabs/other apps
   - ✅ Reduce data loader workers to 1

**Conservative settings for limited VRAM:**
```
Batch size: 1
VAE dim: 32
Feature dim: 32
Workers: 1
```

---

### Dimension Scanning Takes Forever

**Symptom:** Stuck at "Scanning image dimensions: 1% | ..."

**This is NORMAL for large datasets!**

**Expected times:**
- 1,000 images: ~1 minute
- 10,000 images: ~5-10 minutes
- 100,000 images: ~15-20 minutes
- 134,000 images: ~20-30 minutes

**Why slow:**
- Must open every image to read dimensions
- Builds multi-resolution training cache
- Disk I/O bound (limited by drive speed)

**Solutions:**
✅ **Be patient** - this only happens once per dataset
✅ **Cache is saved** - subsequent runs much faster (2-3 minutes)
✅ **Cannot skip** - required for multi-resolution training

**Speed up:**
- Use SSD instead of HDD (3-5× faster)
- Reduce dataset size for testing
- Reduce workers to 0 if I/O bottleneck

**Not an error if:**
- Progress bar moving (even slowly)
- Percentage increasing over time
- No error messages

---

### "Leaked semaphore objects" Warning

**Symptom:** Warning message in training console

**Full message:**
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

**Status: NOT AN ERROR** ℹ️

**Explanation:**
- Python multiprocessing cleanup warning
- Occurs when DataLoader workers don't shut down cleanly
- **Common on macOS/MPS** (Apple Silicon)
- **Training continues normally** - not a blocker

**Impact:**
- ✅ Training works fine
- ✅ Models train correctly
- ✅ Checkpoints save properly
- ℹ️ Purely cosmetic warning

**If it bothers you:**
```
Set workers: 0 (disables multiprocessing)
Training slower but no warning
```

**DO NOT:**
- ❌ Stop training due to this warning
- ❌ Restart training
- ❌ Report as bug

---

### Training Loss Not Decreasing

**Symptom:** Loss stays high (0.1+) or increases

**Causes & Solutions:**

1. **Too early** (common)
   - ℹ️ Loss may stay high for first 100-500 steps
   - ℹ️ May plateau then drop suddenly
   - ✅ Wait longer before judging

2. **Learning rate too high**
   - ✅ Default 1e-5 usually good
   - ✅ Try 5e-6 if unstable
   - ❌ Don't go below 1e-6 (too slow)

3. **Bad dataset**
   - ✅ Check images load correctly
   - ✅ Verify captions are meaningful
   - ✅ Ensure sufficient diversity

4. **Model capacity too small**
   - ℹ️ VAE dim 32 has limited capacity
   - ✅ Try VAE dim 64 or 128

**Normal loss curve:**
```
Steps 0-100:    0.5 → 0.3  (rapid drop)
Steps 100-500:  0.3 → 0.15 (steady decline)
Steps 500-1000: 0.15 → 0.08 (plateau)
Steps 1000-2000: 0.08 → 0.03 (slow improvement)
```

---

## Installation Issues

### "fluxflow-ui: command not found"

**Symptom:** `fluxflow-ui` doesn't run after install

**Causes & Solutions:**

1. **Not in PATH**
   - ✅ Try: `python -m fluxflow_ui.app_flask`
   - ✅ Restart terminal
   - ✅ Check: `pip show fluxflow-ui` (verify installed)

2. **Wrong Python environment**
   - ✅ Activate correct venv: `source venv/bin/activate`
   - ✅ Verify: `which python` points to venv

3. **Installation failed**
   - ✅ Reinstall: `pip install --force-reinstall fluxflow-ui`
   - ✅ Check for error messages during install

---

### Import Errors

**Symptom:** "ModuleNotFoundError" when starting UI

**Solutions:**

1. **Missing dependencies**
   - ✅ `pip install -r requirements.txt`
   - ✅ `pip install fluxflow-ui` (reinstall)

2. **Version conflicts**
   - ✅ Create fresh venv
   - ✅ `pip install fluxflow-ui` in clean environment

---

## Performance Issues

### Generation is Very Slow

**Expected times** (512×512, 50 steps):
- **GPU (NVIDIA RTX 3090):** 3-5 seconds
- **MPS (Apple M1 Max):** 8-12 seconds  
- **CPU:** 60-120 seconds

**If slower than expected:**

1. **Check device usage**
   - ✅ Verify GPU/MPS is being used (check logs)
   - ℹ️ CPU fallback is 10-20× slower

2. **CFG enabled**
   - ℹ️ CFG doubles generation time (expected)
   - ✅ Disable CFG for faster preview

3. **Too many steps**
   - ✅ Reduce to 20-30 steps for faster testing
   - ℹ️ 50-100 steps for production quality

4. **Large image size**
   - ℹ️ 1024×1024 is 4× slower than 512×512
   - ✅ Use 256×256 for quick tests

---

### Training is Very Slow

**Expected speed** (depends on hardware):
- **RTX 3090:** 2-5 steps/second
- **Apple M1 Max:** 0.5-2 steps/second
- **CPU:** 0.1-0.3 steps/second

**Speed up training:**

1. **Increase batch size**
   - ✅ Batch size 4 = ~2× faster than batch size 1
   - ⚠️ Limited by VRAM

2. **Use GPU instead of CPU**
   - ℹ️ 10-30× speedup
   - ✅ Install CUDA/MPS support

3. **Reduce image resolution**
   - ℹ️ Smaller images train faster
   - ✅ Resize dataset to 512×512 max

4. **Disable unnecessary features**
   - ✅ Disable SPADE if not needed
   - ✅ Reduce data loader workers if I/O bound

---

## UI Issues

### Buttons Don't Work / Can't Click

**Symptom:** Clicking buttons has no effect

**Solutions:**

1. **Refresh page**
   - ✅ Press Ctrl+R (Windows) or Cmd+R (Mac)
   - ✅ Hard refresh: Ctrl+Shift+R / Cmd+Shift+R

2. **Clear browser cache**
   - ✅ Ctrl+Shift+Delete → Clear cache
   - ✅ Restart browser

3. **JavaScript disabled**
   - ✅ Enable JavaScript in browser settings
   - ✅ Check for script blockers (uBlock, NoScript, etc.)

4. **Try different browser**
   - ✅ Chrome/Chromium recommended
   - ℹ️ Safari/Firefox may have compatibility issues

---

### File Browser Doesn't Open

**Symptom:** Clicking "Browse" button does nothing

**Solutions:**

1. **Modal already open**
   - ✅ Press ESC key to close hidden modal
   - ✅ Refresh page

2. **JavaScript error**
   - ✅ Open browser console (F12)
   - ✅ Check for errors
   - ✅ Screenshot and report if persistent

3. **Path field disabled**
   - ℹ️ Some fields auto-populate
   - ✅ Try different browse button

---

### Training Console Not Updating

**Symptom:** Console shows "Waiting to start training..." forever

**Causes & Solutions:**

1. **Training actually crashed**
   - ✅ Check terminal where Flask is running
   - ✅ Look for Python errors/stack traces
   - ✅ Restart Flask server

2. **WebSocket disconnected**
   - ✅ Refresh page to reconnect
   - ✅ Check Flask server still running

3. **Training hasn't started yet**
   - ℹ️ Dimension scanning can take 10-20 minutes
   - ✅ Be patient
   - ✅ Check terminal for progress

---

## Getting Help

### Before Reporting Issues

**Gather this information:**

1. **System info:**
   - OS and version
   - Python version (`python --version`)
   - GPU/MPS or CPU
   - VRAM/RAM amount

2. **Package versions:**
   ```bash
   pip show fluxflow-ui
   pip show fluxflow-training
   pip show fluxflow
   ```

3. **Full error message:**
   - Copy entire error/stack trace
   - Include console output
   - Screenshot if UI issue

4. **Reproduction steps:**
   - Exact steps to trigger issue
   - Configuration used
   - Dataset size/type

### Where to Get Help

1. **Check documentation:**
   - [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive guide
   - [README.md](../README.md) - Quick start
   - This troubleshooting guide

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/danny-mio/fluxflow-ui/issues)
   - May already be solved

3. **Report new issue:**
   - [Create GitHub Issue](https://github.com/danny-mio/fluxflow-ui/issues/new)
   - Include all info from "Before Reporting" section

---

## FAQ

**Q: Can I use FluxFlow UI remotely?**  
A: Yes, but **NOT recommended without security hardening**. See [SECURITY.md](../SECURITY.md). No authentication by default.

**Q: Can I train on multiple GPUs?**  
A: Not currently supported in UI. Use CLI for distributed training.

**Q: How long does training take?**  
A: Depends on dataset size and hardware. Example: 10K images, 5 epochs, RTX 3090 = ~8 hours.

**Q: Can I pause and resume training?**  
A: No automatic pause/resume. Stop training and resume from last checkpoint manually.

**Q: Do I need a GPU?**  
A: No, but strongly recommended. CPU training is 10-30× slower.

**Q: What's the minimum dataset size?**  
A: 1,000 images minimum for testing. 10,000+ recommended for good results.

**Q: Can I use my own captions?**  
A: Yes! Plain text file, one caption per line, matching image order.

---

**Still stuck? [Open an issue](https://github.com/danny-mio/fluxflow-ui/issues) with full details.**

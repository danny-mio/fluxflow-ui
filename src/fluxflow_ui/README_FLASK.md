# FluxFlow UI (Flask) - Python 3.14 Compatible

A lightweight, Python 3.14 compatible web interface for FluxFlow using Flask instead of Gradio.

## Why Flask?

Gradio requires `pydantic` which uses Rust compilation (via pydantic-core) that doesn't support Python 3.14 yet. Flask provides a simpler, pure-Python alternative that works perfectly with Python 3.14.

## Installation

```bash
# Create venv with Python 3.14
python3.14 -m venv venv
source venv/bin/activate

# Install UI dependencies
pip install -r ui/requirements_ui.txt

# Install PyTorch (CPU version for compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other FluxFlow dependencies
pip install transformers safetensors diffusers accelerate
```

## Quick Start

```bash
./launch_ui.sh
```

Then open: **http://localhost:7860**

## Features

### Training Tab
- Full training parameter configuration
- Real-time console output (updates every second)
- Start/Stop training controls
- Configuration auto-save

### Generation Tab
- Model loading with dimension configuration
- Text-to-image generation
- Parameter controls (size, steps, seed)
- Live image preview

## Tech Stack

- **Backend**: Flask 3.1+ (pure Python, no Rust dependencies)
- **Frontend**: Vanilla JavaScript + HTML/CSS
- **ML**: PyTorch 2.9+ (Python 3.14 compatible)
- **API**: RESTful JSON endpoints

## API Endpoints

### Training
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training  
- `GET /api/training/status` - Get status + console output

### Generation
- `POST /api/generation/load` - Load model
- `POST /api/generation/generate` - Generate image
- `GET /api/generation/status` - Get model status

### Configuration
- `GET /api/config/training` - Get training config
- `POST /api/config/training` - Save training config
- `GET /api/config/generation` - Get generation config
- `POST /api/config/generation` - Save generation config

## Python 3.14 Compatibility

✅ **Working**:
- Flask 3.1.2
- PyTorch 2.9.1 (CPU)
- Transformers 4.57.1
- Safetensors 0.6.2
- NumPy 2.3.4
- Pillow 12.0.0

❌ **Not Working** (Rust compilation issues):
- Gradio (depends on pydantic)
- Pydantic 2.x (pydantic-core doesn't support Python 3.14)

## Differences from Gradio Version

| Feature | Gradio UI | Flask UI |
|---------|-----------|----------|
| Framework | Gradio 5.x | Flask 3.x |
| Python 3.14 | ❌ | ✅ |
| Auto-refresh UI | Built-in | JavaScript polling |
| File uploads | Drag & drop | Text input |
| Components | Pre-built | Custom HTML |
| API | Auto-generated | Manual endpoints |

## Development

To add new features:

1. **Backend**: Add route to `ui/app_flask.py`
2. **Frontend**: Update `ui/templates/index.html`
3. **API**: Use JSON for request/response

Example adding a new endpoint:

```python
@app.route('/api/custom/endpoint', methods=['POST'])
def custom_endpoint():
    data = request.json
    # Your logic here
    return jsonify({"status": "success", "data": result})
```

## Troubleshooting

### Port already in use
```bash
# Find process using port 7860
lsof -ti:7860 | xargs kill -9
```

### Import errors
```bash
# Reinstall dependencies
source venv/bin/activate
pip install --force-reinstall -r ui/requirements_ui.txt
```

### Torch not found
```bash
# Install PyTorch from official wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] File upload widgets (instead of text paths)
- [ ] Training metrics visualization
- [ ] Image gallery for generation history
- [ ] Batch generation queue

## License

Same as FluxFlow project (MIT)

---

**Built with ❤️ for Python 3.14 compatibility**

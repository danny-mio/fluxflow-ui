# FluxFlow UI - User Guide

## Installation

### Prerequisites

- Python >= 3.10
- pip
- Git (for cloning repository)

### Development Installation

FluxFlow UI is **not published to PyPI**. Install from source:

```bash
# Clone the repository
git clone https://github.com/danny-mio/fluxflow-ui.git
cd fluxflow-ui

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

This will install:
- `fluxflow-ui` (this package)
- `fluxflow-training` (training tools dependency)
- `fluxflow` (core library dependency)
- All development tools (pytest, black, flake8, etc.)

## Launching the UI

### Flask Interface (Primary)

The Flask interface is the **primary and recommended** way to use FluxFlow UI:

```bash
fluxflow-ui
```

Features:
- Modern REST API
- Structured logging
- Better error handling
- Security features (CORS, input validation)
- Production-ready architecture

Access at: `http://localhost:7860`

### Gradio Interface (Alternative)

Gradio is provided as an alternative interface:

```bash
python -m fluxflow_ui.app
```

**Note:** Gradio may have limited functionality compared to Flask. Use Flask for full features.

## Security Configuration

### Local Development (Default)

FluxFlow UI is configured for **local development only**:

- CORS restricted to `localhost:7860` and `127.0.0.1:7860`
- No authentication (suitable for single-user local use)
- File browser allows filesystem access (within OS permissions)

### Production Deployment

⚠️ **DO NOT deploy to production without additional security measures:**

1. **Authentication Required**
   - Add OAuth2, API keys, or basic authentication
   - FluxFlow UI has NO built-in authentication

2. **Network Security**
   - Use HTTPS with valid certificates
   - Deploy behind reverse proxy (nginx, Caddy)
   - Restrict network access (VPN, firewall)

3. **Input Validation**
   - Review and enhance bounds checking
   - Add rate limiting
   - Implement request size limits

4. **File System Protection**
   - Restrict file browser to specific directories
   - Currently allows access to entire filesystem
   - Use `safe_path()` in all file operations

5. **Process Isolation**
   - Run in containers (Docker, Podman)
   - Use separate user accounts with limited permissions

See [SECURITY.md](SECURITY.md) for complete security documentation.

## Using the Interface

### Training Tab

1. **Configure Training Parameters**
   - Dataset path
   - Model dimensions
   - Learning rate, batch size
   - Training steps and checkpointing

2. **Start Training**
   - Click "Start Training"
   - Monitor progress in real-time
   - View logs and metrics

3. **Stop Training**
   - Click "Stop Training" to interrupt
   - Checkpoints saved automatically

### Generation Tab

1. **Load Model**
   - Browse for `.safetensors` checkpoint
   - Inspect model to detect dimensions
   - Click "Load Model"

2. **Generate Images**
   - Enter text prompt
   - Configure image size, steps, seed
   - Click "Generate"
   - View generated image

3. **Adjust Parameters**
   - Experiment with different settings
   - Save favorite configurations

## Configuration Files

FluxFlow UI stores configuration in:

```
~/.fluxflow/
├── training_config.json    # Training parameters
└── generation_config.json  # Generation parameters
```

These files are created automatically and can be edited manually if needed.

## Known Limitations

### Test Coverage

Current test coverage: **~2%**

- Only basic import and configuration tests exist
- Most functionality lacks automated testing
- Manual testing required for validation

**Impact:**
- Regressions may not be caught automatically
- Refactoring is risky without comprehensive tests
- Production deployment requires thorough manual QA

### File Browser Security

The file browser (`/api/files/browse`) allows access to the **entire filesystem** (within OS permissions):

- No base directory restriction enforced
- `safe_path()` function exists but not used
- Suitable for local development only
- **CRITICAL** security concern for production

### Thread Safety

`TrainingRunner` may have race conditions in multi-threaded scenarios:

- Shared state accessed without locks
- Generally safe for single-user local use
- Needs review for production deployment

### Input Validation

Limited input validation implemented:

- JSON structure validated via `@require_json`
- Numeric bounds checking not implemented
- String length limits not enforced
- File path sanitization incomplete

## Troubleshooting

### Port Already in Use

If port 7860 is already in use:

```bash
# Find process using port
lsof -i :7860

# Kill process or change port in source
# Edit src/fluxflow_ui/app_flask.py, line 382
```

### Model Won't Load

1. Verify checkpoint path is correct
2. Check model dimensions match checkpoint
3. Ensure sufficient GPU/CPU memory
4. Review logs for error messages

### Training Fails to Start

1. Verify dataset path exists
2. Check configuration validity
3. Ensure sufficient disk space
4. Review training logs for errors

### Permission Errors

File browser may encounter permission errors:

- Run with appropriate user permissions
- Check file/directory ownership
- Review OS-level access controls

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_config_manager.py

# With coverage
pytest --cov=fluxflow_ui
```

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Type checking
make type-check

# All checks
make check
```

### Project Structure

```
fluxflow-ui/
├── src/fluxflow_ui/
│   ├── app_flask.py       # Flask application (primary)
│   ├── app.py             # Gradio application (alternative)
│   ├── tabs/              # Gradio UI components
│   ├── utils/             # Core utilities
│   ├── templates/         # HTML templates
│   └── static/            # CSS/JS assets
├── tests/                 # Test suite (~2% coverage)
├── SECURITY.md            # Security documentation
├── USER_GUIDE.md          # This file - usage guide
└── pyproject.toml         # Package configuration
```

## Support

- **Issues:** [GitHub Issues](https://github.com/danny-mio/fluxflow-ui/issues)
- **Security:** Report to danielecamisani@inspiredthinking.group (do not open public issues)
- **Documentation:** See project `.md` files

## License

MIT License - see LICENSE file for details.

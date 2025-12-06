# CLAUDE.md

This file provides guidance for Claude Code when working with the FluxFlow UI codebase.

## Project Overview

FluxFlow UI is a web interface for FluxFlow text-to-image generation and training. It provides both Flask (primary) and Gradio-based interfaces for training and generating images with FluxFlow models.

- **Python**: 3.10+
- **Primary Framework**: Flask (REST API on port 7860)
- **Alternative UI**: Gradio

## Quick Reference

### Common Commands

```bash
# Install
make install              # Install in editable mode
make install-dev          # Install with dev dependencies

# Test
make test                 # Run all tests
pytest tests/ -v          # Verbose test output

# Lint & Format
make lint                 # Run flake8, black --check, isort --check
make format               # Format with black and isort

# Run
make run                  # Launch web UI
fluxflow-ui               # Direct command
```

### Code Quality Tools

| Tool | Command | Purpose |
|------|---------|---------|
| Black | `black src/` | Code formatting |
| isort | `isort src/` | Import sorting |
| flake8 | `flake8 src/` | Linting |
| mypy | `mypy src/fluxflow_ui` | Type checking |
| pytest | `pytest tests/` | Testing |

## Code Style

- **Line length**: 100 characters
- **Formatter**: Black (enforced)
- **Imports**: isort with Black profile
- **Docstrings**: Google-style
- **Type hints**: Required on all public APIs

### Naming Conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### Import Order

```python
# Standard library
import os
import sys

# Third-party
import flask
import gradio

# Local
from fluxflow_ui.utils import config_manager
```

## Project Structure

```
src/fluxflow_ui/
├── app.py                 # Gradio application
├── app_flask.py           # Flask REST API (primary entry point)
├── utils/
│   ├── config_manager.py  # JSON config persistence
│   ├── training_runner.py # Background training process
│   └── generation_worker.py # Image generation
├── tabs/
│   ├── training.py        # Training tab UI
│   └── generation.py      # Generation tab UI
├── templates/             # Flask HTML templates
└── static/                # CSS/JS assets

tests/
└── test_ui.py             # Test suite
```

## Key Files

- **Entry point**: `src/fluxflow_ui/app_flask.py:main()`
- **Config storage**: `.ui_configs/` (gitignored)
- **Package config**: `pyproject.toml`
- **Lint config**: `.flake8`, `pyproject.toml`

## Testing

- Framework: pytest
- Location: `tests/`
- Timeout: 300 seconds per test
- Coverage config: `.coveragerc`

```bash
# Run specific test
pytest tests/test_ui.py::TestUIImports -v

# Run with coverage
pytest tests/ --cov=src/fluxflow_ui
```

## Architecture Notes

### Flask API Endpoints

The Flask app (`app_flask.py`) provides REST endpoints for:
- Configuration management (GET/POST)
- Training control (start, stop, status)
- Generation (load model, generate images)
- File browsing

### Background Processes

- **Training**: Uses `subprocess.Popen` with real-time output streaming
- **Generation**: Lazy model loading with automatic device detection (CUDA/MPS/CPU)

### Configuration

- Stored as JSON in `.ui_configs/`
- Separate configs for training and generation
- Managed by `config_manager.py`

## Dependencies

### Core Dependency

The `fluxflow-training` package is installed from GitHub:
```
fluxflow-training @ git+https://github.com/danny-mio/fluxflow-training.git
```

This is a sibling project that provides the core training functionality.

## Pre-commit Hooks

The following run automatically:
- trailing-whitespace fix
- Black formatting
- isort sorting
- flake8 linting
- pytest (on push)

Run manually: `pre-commit run --all-files`

## CI/CD

GitHub Actions runs on push/PR:
1. Lint (flake8)
2. Format check (black)
3. Type check (mypy)
4. Tests (pytest with coverage)

Tests run on Python 3.10 and 3.11.

## Common Patterns

### Error Handling

```python
try:
    result = process_data(data)
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    raise CustomError(f"Processing failed: {e}") from e
```

### Type Hints

```python
def load_config(path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    ...
```

### Google-style Docstrings

```python
def train_model(config: dict, output_dir: str) -> None:
    """
    Start model training with given configuration.

    Args:
        config: Training configuration dictionary
        output_dir: Directory for output files

    Raises:
        ValueError: If config is invalid
    """
```

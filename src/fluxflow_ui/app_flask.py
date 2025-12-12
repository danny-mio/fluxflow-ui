"""FluxFlow UI - Flask-based application (Python 3.14 compatible)."""

import base64
import io
import logging
import os
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from fluxflow_ui.utils.config_manager import ConfigManager
from fluxflow_ui.utils.generation_worker import GenerationWorker
from fluxflow_ui.utils.training_runner import TrainingRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Restrict CORS to localhost only for security
CORS(app, origins=["http://localhost:7860", "http://127.0.0.1:7860"])

# Initialize utilities
training_runner = TrainingRunner()
generation_worker = GenerationWorker()
config_manager = ConfigManager()

# Base directory for file browser (security: prevent path traversal)
FILE_BROWSER_BASE_DIR = os.path.abspath(os.getcwd())


def require_json(f):
    """Decorator to require JSON body in POST requests."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.json:
            return jsonify({"status": "error", "message": "JSON body required"}), 400
        return f(*args, **kwargs)

    return decorated


def safe_path(user_path: str, base_dir: str = FILE_BROWSER_BASE_DIR) -> str:
    """Validate and sanitize path to prevent directory traversal.

    Args:
        user_path: User-provided path
        base_dir: Base directory to restrict access to

    Returns:
        Safe absolute path

    Raises:
        ValueError: If path traversal attempt detected
    """
    # Resolve the full path
    full_path = os.path.realpath(os.path.join(base_dir, user_path))

    # Ensure the path starts with the base directory
    if not full_path.startswith(os.path.realpath(base_dir)):
        raise ValueError("Path traversal attempt detected")

    return full_path


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/config/training", methods=["GET"])
def get_training_config():
    """Get training configuration."""
    config = config_manager.load_training_config()
    if not config:
        config = config_manager.get_default_training_config()
    return jsonify(config)


@app.route("/api/config/training", methods=["POST"])
@require_json
def save_training_config():
    """Save training configuration."""
    config = request.json
    config_manager.save_training_config(config)
    return jsonify({"status": "success"})


@app.route("/api/config/generation", methods=["GET"])
def get_generation_config():
    """Get generation configuration."""
    config = config_manager.load_generation_config()
    if not config:
        config = config_manager.get_default_generation_config()
    return jsonify(config)


@app.route("/api/config/generation", methods=["POST"])
@require_json
def save_generation_config():
    """Save generation configuration."""
    config = request.json
    config_manager.save_generation_config(config)
    return jsonify({"status": "success"})


@app.route("/api/training/start", methods=["POST"])
@require_json
def start_training():
    """Start training."""
    config = request.json

    if training_runner.is_running:
        return jsonify({"status": "error", "message": "Training already running"}), 400

    success = training_runner.start_training(config)

    if success:
        logger.info("Training started successfully")
        return jsonify({"status": "success", "message": "Training started"})
    else:
        logger.error("Failed to start training")
        return jsonify({"status": "error", "message": "Failed to start training"}), 500


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    """Stop training."""
    if not training_runner.is_running:
        return jsonify({"status": "error", "message": "No training is running"}), 400

    success = training_runner.stop_training()

    if success:
        logger.info("Training stopped successfully")
        return jsonify({"status": "success", "message": "Training stopped"})
    else:
        logger.error("Failed to stop training")
        return jsonify({"status": "error", "message": "Failed to stop training"}), 500


@app.route("/api/training/status", methods=["GET"])
def training_status():
    """Get training status."""
    return jsonify({"running": training_runner.is_running, "output": training_runner.get_output()})


@app.route("/api/generation/inspect", methods=["POST"])
@require_json
def inspect_model():
    """Inspect model checkpoint and detect dimensions."""
    data = request.json
    checkpoint_path = data.get("checkpoint_path")

    if not checkpoint_path:
        return jsonify({"status": "error", "message": "checkpoint_path required"}), 400

    if not Path(checkpoint_path).exists():
        return jsonify({"status": "error", "message": "Invalid checkpoint path"}), 400

    try:
        import safetensors.torch

        # Load only metadata to detect dimensions
        state_dict = safetensors.torch.load_file(checkpoint_path)

        # Detect VAE dimension from vae_to_dmodel layer (this is the VAE latent dim)
        vae_dim = None
        for key in state_dict.keys():
            if "flow_processor.vae_to_dmodel.weight" in key:
                shape = state_dict[key].shape
                # Shape is [d_model, vae_dim], so vae_dim is shape[1]
                vae_dim = int(shape[1])
                break

        # Detect d_model (feature dimension) from vae_to_dmodel output dimension
        feature_dim = None
        for key in state_dict.keys():
            if "flow_processor.vae_to_dmodel.weight" in key:
                shape = state_dict[key].shape
                # Shape is [d_model, vae_dim], so d_model is shape[0]
                feature_dim = int(shape[0])
                break

        # Fallback to reasonable defaults if detection failed
        if not vae_dim:
            vae_dim = 64
        if not feature_dim:
            feature_dim = 64

        return jsonify(
            {
                "status": "success",
                "vae_dim": int(vae_dim),
                "feature_maps_dim": int(feature_dim),
                "text_embedding_dim": 1024,
                "message": f"Detected: VAE={vae_dim}, Feature={feature_dim}",
            }
        )

    except Exception as e:
        logger.error(f"Failed to inspect model: {e}")
        return jsonify({"status": "error", "message": f"Failed to inspect model: {str(e)}"}), 500


@app.route("/api/generation/load", methods=["POST"])
@require_json
def load_model():
    """Load generation model."""
    data = request.json

    checkpoint_path = data.get("checkpoint_path")
    if not checkpoint_path:
        return jsonify({"status": "error", "message": "checkpoint_path required"}), 400

    success, message = generation_worker.load_model(
        checkpoint_path=checkpoint_path,
        vae_dim=data.get("vae_dim", 64),
        feature_maps_dim=data.get("feature_maps_dim", 64),
        text_embedding_dim=data.get("text_embedding_dim", 1024),
    )

    if success:
        logger.info(f"Model loaded: {checkpoint_path}")
        return jsonify({"status": "success", "message": message})
    else:
        logger.error(f"Failed to load model: {message}")
        return jsonify({"status": "error", "message": message}), 500


@app.route("/api/generation/generate", methods=["POST"])
@require_json
def generate_image():
    """Generate image from prompt."""
    data = request.json

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"status": "error", "message": "prompt required"}), 400

    image, message = generation_worker.generate_image(
        prompt=prompt,
        img_width=data.get("img_width", 512),
        img_height=data.get("img_height", 512),
        ddim_steps=data.get("ddim_steps", 50),
        seed=data.get("seed") if data.get("use_seed") else None,
        use_cfg=data.get("use_cfg", False),
        guidance_scale=data.get("guidance_scale", 5.0),
        negative_prompt=data.get("negative_prompt"),
    )

    if image is not None:
        # Convert numpy array to base64
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        logger.info(f"Image generated for prompt: {prompt[:50]}...")
        return jsonify(
            {
                "status": "success",
                "message": message,
                "image": f"data:image/png;base64,{img_base64}",
            }
        )
    else:
        logger.error(f"Generation failed: {message}")
        return jsonify({"status": "error", "message": message}), 500


@app.route("/api/generation/status", methods=["GET"])
def generation_status():
    """Get generation status."""
    return jsonify({"loaded": generation_worker.is_loaded(), "config": generation_worker.config})


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def _should_include_file(item: Path, file_type: str) -> bool:
    """Check if file should be included based on filter type."""
    if item.is_dir():
        return True  # Always include directories for navigation
    if file_type == "dir":
        return False  # Skip files when only looking for directories
    if file_type == "safetensors":
        return item.name.endswith(".safetensors")
    return True  # 'all' or 'file' - include everything


def _resolve_browse_path(current_path: str) -> Path:
    """Resolve and validate browse path."""
    if not os.path.isabs(current_path):
        path_obj = Path(current_path).expanduser().resolve()
    else:
        path_obj = Path(current_path).resolve()

    if not path_obj.exists():
        path_obj = Path.cwd()

    if path_obj.is_file():
        path_obj = path_obj.parent

    return path_obj


@app.route("/api/files/browse", methods=["POST"])
@require_json
def browse_files():
    """Browse files and directories with path traversal protection."""
    data = request.json
    current_path = data.get("path", ".")
    file_type = data.get("type", "all")  # 'all', 'dir', 'file', 'safetensors'

    try:
        path_obj = _resolve_browse_path(current_path)
        items = []

        # Add parent directory option
        if path_obj.parent != path_obj:
            items.append({"name": "..", "path": str(path_obj.parent), "type": "dir", "size": ""})

        # List directory contents
        for item in sorted(path_obj.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip hidden files
            if item.name.startswith("."):
                continue

            if not _should_include_file(item, file_type):
                continue

            item_type = "dir" if item.is_dir() else "file"
            size = _format_file_size(item.stat().st_size) if item.is_file() else ""

            items.append({"name": item.name, "path": str(item), "type": item_type, "size": size})

        return jsonify({"status": "success", "current_path": str(path_obj), "items": items})

    except PermissionError:
        logger.warning(f"Permission denied accessing path: {current_path}")
        return jsonify({"status": "error", "message": "Permission denied"}), 403
    except Exception as e:
        logger.error(f"Error browsing files: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


def main():
    """Main entry point for fluxflow-ui command."""
    logger.info("=" * 60)
    logger.info("FluxFlow UI (Flask)")
    logger.info("=" * 60)
    logger.info("Starting server at http://localhost:7860")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()

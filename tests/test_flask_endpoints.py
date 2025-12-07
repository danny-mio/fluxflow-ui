"""Tests for Flask API endpoints."""

import json
from unittest.mock import patch

import pytest

from fluxflow_ui.app_flask import app


@pytest.fixture
def client():
    """Create test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestIndexEndpoint:
    """Tests for main page endpoint."""

    def test_index_returns_html(self, client):
        """Should return HTML page."""
        response = client.get("/")

        assert response.status_code == 200


class TestTrainingConfigEndpoints:
    """Tests for training config endpoints."""

    def test_get_training_config_returns_json(self, client):
        """GET should return JSON config."""
        response = client.get("/api/config/training")

        assert response.status_code == 200
        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_post_training_config_requires_json(self, client):
        """POST should require JSON body."""
        response = client.post("/api/config/training")

        # 415 = UNSUPPORTED MEDIA TYPE (no Content-Type header)
        assert response.status_code in [400, 415]

    def test_post_training_config_saves(self, client):
        """POST should save config."""
        config = {"batch_size": 4}
        response = client.post(
            "/api/config/training", data=json.dumps(config), content_type="application/json"
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"


class TestGenerationConfigEndpoints:
    """Tests for generation config endpoints."""

    def test_get_generation_config_returns_json(self, client):
        """GET should return JSON config."""
        response = client.get("/api/config/generation")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_post_generation_config_requires_json(self, client):
        """POST should require JSON body."""
        response = client.post("/api/config/generation")

        assert response.status_code in [400, 415]

    def test_post_generation_config_saves(self, client):
        """POST should save config."""
        config = {"img_size": 512}
        response = client.post(
            "/api/config/generation", data=json.dumps(config), content_type="application/json"
        )

        assert response.status_code == 200


class TestTrainingEndpoints:
    """Tests for training control endpoints."""

    def test_start_training_requires_json(self, client):
        """POST start should require JSON body."""
        response = client.post("/api/training/start")

        assert response.status_code in [400, 415]

    @patch("fluxflow_ui.app_flask.training_runner")
    def test_start_training_returns_error_if_running(self, mock_runner, client):
        """Should return error if already running."""
        mock_runner.is_running = True

        response = client.post(
            "/api/training/start", data=json.dumps({}), content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        # Check for either message format
        assert "already running" in data.get("message", "") or data.get("status") == "error"

    @patch("fluxflow_ui.app_flask.training_runner")
    def test_start_training_success(self, mock_runner, client):
        """Should start training successfully."""
        mock_runner.is_running = False
        mock_runner.start_training.return_value = True

        response = client.post(
            "/api/training/start",
            data=json.dumps({"batch_size": 4}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"

    @patch("fluxflow_ui.app_flask.training_runner")
    def test_stop_training_returns_error_if_not_running(self, mock_runner, client):
        """Should return error if not running."""
        mock_runner.is_running = False

        response = client.post("/api/training/stop")

        assert response.status_code == 400

    @patch("fluxflow_ui.app_flask.training_runner")
    def test_stop_training_success(self, mock_runner, client):
        """Should stop training successfully."""
        mock_runner.is_running = True
        mock_runner.stop_training.return_value = True

        response = client.post("/api/training/stop")

        assert response.status_code == 200

    @patch("fluxflow_ui.app_flask.training_runner")
    def test_training_status_returns_state(self, mock_runner, client):
        """Should return current training state."""
        mock_runner.is_running = True
        mock_runner.get_output.return_value = ["line 1", "line 2"]

        response = client.get("/api/training/status")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["running"] is True
        assert data["output"] == ["line 1", "line 2"]


class TestGenerationEndpoints:
    """Tests for generation endpoints."""

    def test_inspect_model_requires_json(self, client):
        """Should require JSON body."""
        response = client.post("/api/generation/inspect")

        assert response.status_code in [400, 415]

    def test_inspect_model_requires_checkpoint_path(self, client):
        """Should require checkpoint_path."""
        response = client.post(
            "/api/generation/inspect", data=json.dumps({}), content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        # Check message contains expected text or is an error status
        assert "checkpoint_path" in data.get("message", "") or data.get("status") == "error"

    def test_inspect_model_returns_error_for_invalid_path(self, client):
        """Should return error for non-existent path."""
        response = client.post(
            "/api/generation/inspect",
            data=json.dumps({"checkpoint_path": "/nonexistent/model.safetensors"}),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_load_model_requires_json(self, client):
        """Should require JSON body."""
        response = client.post("/api/generation/load")

        assert response.status_code in [400, 415]

    def test_load_model_requires_checkpoint_path(self, client):
        """Should require checkpoint_path."""
        response = client.post(
            "/api/generation/load", data=json.dumps({}), content_type="application/json"
        )

        assert response.status_code == 400

    def test_generate_image_requires_json(self, client):
        """Should require JSON body."""
        response = client.post("/api/generation/generate")

        assert response.status_code in [400, 415]

    def test_generate_image_requires_prompt(self, client):
        """Should require prompt."""
        response = client.post(
            "/api/generation/generate", data=json.dumps({}), content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "prompt" in data.get("message", "") or data.get("status") == "error"

    @patch("fluxflow_ui.app_flask.generation_worker")
    def test_generation_status_returns_state(self, mock_worker, client):
        """Should return generation worker state."""
        mock_worker.is_loaded.return_value = True
        mock_worker.config = {"vae_dim": 64}

        response = client.get("/api/generation/status")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["loaded"] is True


class TestFileBrowserEndpoint:
    """Tests for file browser endpoint."""

    def test_browse_files_requires_json(self, client):
        """Should require JSON body."""
        response = client.post("/api/files/browse")

        assert response.status_code in [400, 415]

    def test_browse_files_returns_items(self, client, tmp_path):
        """Should return directory items."""
        # Create test files
        (tmp_path / "test.txt").touch()
        (tmp_path / "subdir").mkdir()

        response = client.post(
            "/api/files/browse",
            data=json.dumps({"path": str(tmp_path)}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "items" in data

        names = [item["name"] for item in data["items"]]
        assert "test.txt" in names
        assert "subdir" in names

    def test_browse_files_defaults_to_cwd(self, client):
        """Should default to current directory."""
        response = client.post(
            "/api/files/browse", data=json.dumps({"path": "."}), content_type="application/json"
        )

        assert response.status_code == 200

    def test_browse_files_filters_by_type(self, client, tmp_path):
        """Should filter by file type."""
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "other.txt").touch()

        response = client.post(
            "/api/files/browse",
            data=json.dumps({"path": str(tmp_path), "type": "safetensors"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        file_names = [item["name"] for item in data["items"] if item["type"] == "file"]
        assert "model.safetensors" in file_names
        assert "other.txt" not in file_names

    def test_browse_files_hides_dotfiles(self, client, tmp_path):
        """Should hide hidden files."""
        (tmp_path / ".hidden").touch()
        (tmp_path / "visible").touch()

        response = client.post(
            "/api/files/browse",
            data=json.dumps({"path": str(tmp_path)}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        names = [item["name"] for item in data["items"]]
        assert ".hidden" not in names
        assert "visible" in names

    def test_browse_files_includes_parent_dir(self, client, tmp_path):
        """Should include parent directory option."""
        response = client.post(
            "/api/files/browse",
            data=json.dumps({"path": str(tmp_path)}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        names = [item["name"] for item in data["items"]]
        assert ".." in names


class TestJSONValidation:
    """Tests for JSON validation decorator."""

    def test_empty_body_rejected(self, client):
        """Should reject empty request body."""
        response = client.post("/api/config/training", data="", content_type="application/json")

        assert response.status_code == 400

    def test_invalid_json_rejected(self, client):
        """Should reject invalid JSON."""
        response = client.post(
            "/api/config/training", data="not json", content_type="application/json"
        )

        # Flask returns 400 for invalid JSON
        assert response.status_code in [400, 415]

    def test_valid_json_accepted(self, client):
        """Should accept valid JSON."""
        response = client.post(
            "/api/config/training", data=json.dumps({"test": True}), content_type="application/json"
        )

        assert response.status_code == 200

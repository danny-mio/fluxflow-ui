"""Tests for ConfigManager."""

import json
import os

from fluxflow_ui.utils.config_manager import ConfigManager


class TestConfigManagerInit:
    """Tests for ConfigManager initialization."""

    def test_creates_config_directory(self, tmp_path):
        """Should create config directory if it doesn't exist."""
        config_dir = tmp_path / "test_configs"
        assert not config_dir.exists()

        ConfigManager(str(config_dir))

        assert config_dir.exists()
        assert config_dir.is_dir()

    def test_sets_config_paths(self, tmp_path):
        """Should set correct paths for config files."""
        config_dir = tmp_path / "configs"
        manager = ConfigManager(str(config_dir))

        assert manager.training_config_path == config_dir / "training_config.json"
        assert manager.generation_config_path == config_dir / "generation_config.json"


class TestSaveTrainingConfig:
    """Tests for saving training configuration."""

    def test_save_valid_config(self, tmp_path):
        """Should save valid config to file."""
        manager = ConfigManager(str(tmp_path / "configs"))
        config = {"batch_size": 4, "lr": 1e-4}

        result = manager.save_training_config(config)

        assert result is True
        assert manager.training_config_path.exists()

        with open(manager.training_config_path) as f:
            saved = json.load(f)
        assert saved == config

    def test_save_complex_config(self, tmp_path):
        """Should save complex nested config."""
        manager = ConfigManager(str(tmp_path / "configs"))
        config = {
            "optimizers": {
                "flow": {"type": "AdamW", "lr": 5e-7},
                "vae": {"type": "AdamW", "lr": 5e-7},
            },
            "schedulers": {
                "flow": {"type": "CosineAnnealingLR"},
            },
        }

        result = manager.save_training_config(config)

        assert result is True
        with open(manager.training_config_path) as f:
            saved = json.load(f)
        assert saved == config

    def test_save_overwrites_existing(self, tmp_path):
        """Should overwrite existing config file."""
        manager = ConfigManager(str(tmp_path / "configs"))

        manager.save_training_config({"version": 1})
        manager.save_training_config({"version": 2})

        with open(manager.training_config_path) as f:
            saved = json.load(f)
        assert saved["version"] == 2

    def test_save_returns_false_on_permission_error(self, tmp_path):
        """Should return False when cannot write file."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager = ConfigManager(str(config_dir))

        # Make directory read-only
        os.chmod(config_dir, 0o444)

        try:
            result = manager.save_training_config({"test": True})
            assert result is False
        finally:
            # Restore permissions for cleanup
            os.chmod(config_dir, 0o755)


class TestLoadTrainingConfig:
    """Tests for loading training configuration."""

    def test_load_existing_config(self, tmp_path):
        """Should load existing config file."""
        manager = ConfigManager(str(tmp_path / "configs"))
        config = {"batch_size": 8, "epochs": 10}

        with open(manager.training_config_path, "w") as f:
            json.dump(config, f)

        loaded = manager.load_training_config()

        assert loaded == config

    def test_load_returns_none_when_missing(self, tmp_path):
        """Should return None when config file doesn't exist."""
        manager = ConfigManager(str(tmp_path / "configs"))

        loaded = manager.load_training_config()

        assert loaded is None

    def test_load_returns_none_on_corrupted_json(self, tmp_path):
        """Should return None when config file is corrupted."""
        manager = ConfigManager(str(tmp_path / "configs"))

        with open(manager.training_config_path, "w") as f:
            f.write("not valid json {{{")

        loaded = manager.load_training_config()

        assert loaded is None

    def test_load_returns_none_on_empty_file(self, tmp_path):
        """Should return None when config file is empty."""
        manager = ConfigManager(str(tmp_path / "configs"))

        manager.training_config_path.touch()

        loaded = manager.load_training_config()

        assert loaded is None


class TestSaveGenerationConfig:
    """Tests for saving generation configuration."""

    def test_save_valid_config(self, tmp_path):
        """Should save valid generation config."""
        manager = ConfigManager(str(tmp_path / "configs"))
        config = {"img_size": 512, "ddim_steps": 50}

        result = manager.save_generation_config(config)

        assert result is True
        assert manager.generation_config_path.exists()

    def test_save_returns_false_on_invalid_data(self, tmp_path):
        """Should return False when data cannot be serialized."""
        manager = ConfigManager(str(tmp_path / "configs"))

        # Circular reference cannot be serialized
        config = {}
        config["self"] = config

        result = manager.save_generation_config(config)

        assert result is False


class TestLoadGenerationConfig:
    """Tests for loading generation configuration."""

    def test_load_existing_config(self, tmp_path):
        """Should load existing generation config."""
        manager = ConfigManager(str(tmp_path / "configs"))
        config = {"model_checkpoint": "/path/to/model.safetensors"}

        with open(manager.generation_config_path, "w") as f:
            json.dump(config, f)

        loaded = manager.load_generation_config()

        assert loaded == config

    def test_load_returns_none_when_missing(self, tmp_path):
        """Should return None when config doesn't exist."""
        manager = ConfigManager(str(tmp_path / "configs"))

        loaded = manager.load_generation_config()

        assert loaded is None


class TestDefaultConfigs:
    """Tests for default configuration methods."""

    def test_default_training_config_has_required_keys(self, tmp_path):
        """Default training config should have all required keys."""
        manager = ConfigManager(str(tmp_path / "configs"))

        config = manager.get_default_training_config()

        # Check essential keys
        assert "batch_size" in config
        assert "lr" in config
        assert "n_epochs" in config
        assert "vae_dim" in config
        assert "optimizers" in config
        assert "schedulers" in config

    def test_default_training_config_has_valid_values(self, tmp_path):
        """Default training config should have sensible values."""
        manager = ConfigManager(str(tmp_path / "configs"))

        config = manager.get_default_training_config()

        assert config["batch_size"] > 0
        assert config["lr"] > 0
        assert config["n_epochs"] > 0

    def test_default_generation_config_has_required_keys(self, tmp_path):
        """Default generation config should have all required keys."""
        manager = ConfigManager(str(tmp_path / "configs"))

        config = manager.get_default_generation_config()

        assert "img_size" in config
        assert "ddim_steps" in config
        assert "vae_dim" in config
        assert "model_checkpoint" in config

    def test_default_generation_config_has_valid_values(self, tmp_path):
        """Default generation config should have sensible values."""
        manager = ConfigManager(str(tmp_path / "configs"))

        config = manager.get_default_generation_config()

        assert config["img_size"] > 0
        assert config["ddim_steps"] > 0


class TestRoundTrip:
    """Tests for save/load round-trip consistency."""

    def test_training_config_round_trip(self, tmp_path):
        """Saving and loading training config should preserve data."""
        manager = ConfigManager(str(tmp_path / "configs"))
        original = manager.get_default_training_config()

        manager.save_training_config(original)
        loaded = manager.load_training_config()

        assert loaded == original

    def test_generation_config_round_trip(self, tmp_path):
        """Saving and loading generation config should preserve data."""
        manager = ConfigManager(str(tmp_path / "configs"))
        original = manager.get_default_generation_config()

        manager.save_generation_config(original)
        loaded = manager.load_generation_config()

        assert loaded == original

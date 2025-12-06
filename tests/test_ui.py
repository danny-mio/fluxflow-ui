"""Basic tests for FluxFlow UI package."""


class TestUIImports:
    """Test that UI package imports work correctly."""

    def test_import_fluxflow_ui(self):
        """Test importing the main package."""
        import fluxflow_ui

        assert hasattr(fluxflow_ui, "__version__")

    def test_import_app_flask(self):
        """Test importing Flask app module."""
        from fluxflow_ui import app_flask

        assert app_flask is not None

    def test_import_config_manager(self):
        """Test importing config manager."""
        from fluxflow_ui.utils import config_manager

        assert config_manager is not None


class TestUIConfiguration:
    """Test UI configuration functionality."""

    def test_config_manager_exists(self):
        """Test that ConfigManager class exists."""
        from fluxflow_ui.utils.config_manager import ConfigManager

        assert ConfigManager is not None

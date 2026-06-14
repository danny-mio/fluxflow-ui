"""FluxFlow UI - Main application."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr  # noqa: E402

from fluxflow_ui.tabs.generation import create_generation_tab  # noqa: E402
from fluxflow_ui.tabs.training import create_training_tab  # noqa: E402
from fluxflow_ui.utils.config_manager import ConfigManager  # noqa: E402
from fluxflow_ui.utils.generation_worker import GenerationWorker  # noqa: E402
from fluxflow_ui.utils.training_runner import TrainingRunner  # noqa: E402


def create_app() -> gr.Blocks:
    """Create FluxFlow UI application.

    Returns:
        Gradio Blocks app
    """
    # Initialize utilities
    training_runner = TrainingRunner()
    generation_worker = GenerationWorker()
    config_manager = ConfigManager()

    # Create app
    app: gr.Blocks
    with gr.Blocks(
        title="FluxFlow UI",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🎨 FluxFlow UI

            Train and generate images with FluxFlow text-to-image models
            """
        )

        with gr.Tabs():
            create_training_tab(training_runner, config_manager)
            create_generation_tab(generation_worker, config_manager)

        gr.Markdown(
            """
            ---
            **FluxFlow** - Flow-based Text-to-Image Generation
            Made with ❤️ by Daniele Camisani
            """
        )

    return app


def main():
    """Main entry point."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

"""Generation tab for FluxFlow UI."""

from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from fluxflow_ui.utils.config_manager import ConfigManager
from fluxflow_ui.utils.generation_worker import GenerationWorker


def create_generation_tab(worker: GenerationWorker, config_mgr: ConfigManager) -> gr.Tab:
    """Create generation interface tab.

    Args:
        worker: Generation worker instance
        config_mgr: Configuration manager

    Returns:
        Gradio Tab component
    """

    def load_model_handler(
        checkpoint: str,
        vae_dim: int,
        feature_dim: int,
        text_dim: int,
    ) -> Tuple[str, str]:
        """Handle model loading.

        Returns:
            Tuple of (status message, button label)
        """
        if not checkpoint or not Path(checkpoint).exists():
            return "❌ Please provide a valid checkpoint path", "🔄 Load Model"

        success, message = worker.load_model(
            checkpoint_path=checkpoint,
            vae_dim=vae_dim,
            feature_maps_dim=feature_dim,
            text_embedding_dim=text_dim,
        )

        if success:
            config_mgr.save_generation_config(
                {
                    "model_checkpoint": checkpoint,
                    "vae_dim": vae_dim,
                    "feature_maps_dim": feature_dim,
                    "text_embedding_dim": text_dim,
                }
            )
            return f"✅ {message}", "✅ Model Loaded"
        else:
            return f"❌ {message}", "🔄 Load Model"

    def generate_handler(
        prompt: str,
        img_size: int,
        steps: int,
        seed: Optional[int],
        use_seed: bool,
        use_cfg: bool,
        guidance_scale: float,
        negative_prompt: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Handle image generation.

        Returns:
            Tuple of (image, status message)
        """
        if not worker.is_loaded():
            return None, "❌ Please load a model first"

        if not prompt.strip():
            return None, "❌ Please enter a prompt"

        actual_seed = seed if use_seed else None
        image, message = worker.generate_image(
            prompt=prompt,
            img_width=img_size,
            img_height=img_size,
            ddim_steps=steps,
            seed=actual_seed,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale if use_cfg else 1.0,
            negative_prompt=negative_prompt if use_cfg and negative_prompt.strip() else None,
        )

        return image, message

    # Load previous config
    prev_config = config_mgr.load_generation_config()
    if not prev_config:
        prev_config = config_mgr.get_default_generation_config()

    tab: gr.Tab
    with gr.Tab("🎨 Generate") as tab:
        gr.Markdown("# FluxFlow Image Generation")
        gr.Markdown("Generate images from text prompts using trained models")

        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")

                checkpoint_input = gr.Textbox(
                    label="Model Checkpoint",
                    placeholder="outputs/flux/flxflow_final.safetensors",
                    value=prev_config.get("model_checkpoint", ""),
                )

                with gr.Row():
                    vae_dim_input = gr.Number(
                        label="VAE Dimension",
                        value=prev_config.get("vae_dim", 64),
                        precision=0,
                    )
                    feature_dim_input = gr.Number(
                        label="Feature Dimension",
                        value=prev_config.get("feature_maps_dim", 64),
                        precision=0,
                    )

                text_dim_input = gr.Number(
                    label="Text Embedding Dimension",
                    value=prev_config.get("text_embedding_dim", 1024),
                    precision=0,
                )

                load_btn = gr.Button("🔄 Load Model", variant="primary")
                load_status = gr.Textbox(
                    label="Status",
                    value="Model not loaded",
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### Generation Parameters")

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful sunset over mountains",
                    lines=3,
                )

                with gr.Row():
                    img_size_input = gr.Slider(
                        label="Image Size",
                        minimum=256,
                        maximum=1024,
                        step=64,
                        value=512,
                    )
                    steps_input = gr.Slider(
                        label="Sampling Steps",
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=50,
                    )

                with gr.Row():
                    use_seed_checkbox = gr.Checkbox(
                        label="Use Fixed Seed",
                        value=False,
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                    )

                with gr.Accordion("Classifier-Free Guidance", open=False):
                    gr.Markdown(
                        """
                        **CFG** increases prompt adherence. Requires model trained with CFG dropout.
                        """
                    )
                    use_cfg_checkbox = gr.Checkbox(
                        label="Enable CFG",
                        value=False,
                        info="Model must be trained with cfg_dropout_prob > 0",
                    )
                    guidance_scale_input = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        step=0.5,
                        value=5.0,
                        info="Higher = stronger prompt adherence (3-7 recommended)",
                        visible=False,
                    )
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="blurry, low quality, distorted",
                        lines=2,
                        info="What to avoid in the image",
                        visible=False,
                    )

                    def toggle_cfg_gen(enabled):
                        return gr.update(visible=enabled), gr.update(visible=enabled)

                    use_cfg_checkbox.change(
                        fn=toggle_cfg_gen,
                        inputs=[use_cfg_checkbox],
                        outputs=[guidance_scale_input, negative_prompt_input],
                    )

                generate_btn = gr.Button("✨ Generate Image", variant="primary", size="lg")
                gen_status = gr.Textbox(
                    label="Generation Status",
                    value="Ready to generate",
                    interactive=False,
                )

            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### Generated Image")
                output_image = gr.Image(
                    label="Result",
                    type="numpy",
                    height=600,
                )

                gr.Markdown("### Tips")
                gr.Markdown(
                    """
                - **Model Loading**: Select a checkpoint and click Load Model first
                - **Prompt**: Describe the image you want to generate
                - **Image Size**: Larger sizes take longer but produce more detail
                - **Steps**: More steps = higher quality but slower (50 is usually good)
                - **Seed**: Enable for reproducible results
                """
                )

        # Event handlers
        load_btn.click(
            fn=load_model_handler,
            inputs=[checkpoint_input, vae_dim_input, feature_dim_input, text_dim_input],
            outputs=[load_status, load_btn],
        )

        generate_btn.click(
            fn=generate_handler,
            inputs=[
                prompt_input,
                img_size_input,
                steps_input,
                seed_input,
                use_seed_checkbox,
                use_cfg_checkbox,
                guidance_scale_input,
                negative_prompt_input,
            ],
            outputs=[output_image, gen_status],
        )

    return tab

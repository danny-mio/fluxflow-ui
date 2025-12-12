"""Training tab for FluxFlow UI."""

from pathlib import Path
from typing import Tuple

import gradio as gr

from fluxflow_ui.utils.config_manager import ConfigManager
from fluxflow_ui.utils.training_runner import TrainingRunner


def create_training_tab(runner: TrainingRunner, config_mgr: ConfigManager) -> gr.Tab:
    """Create training interface tab.

    Args:
        runner: Training runner instance
        config_mgr: Configuration manager

    Returns:
        Gradio Tab component
    """

    output_lines: list[str] = []

    def start_training_handler(
        # Data
        data_path: str,
        captions_file: str,
        use_webdataset: bool,
        webdataset_token: str,
        webdataset_url: str,
        webdataset_image_key: str,
        webdataset_caption_key: str,
        # Model
        output_path: str,
        model_checkpoint: str,
        vae_dim: int,
        text_embedding_dim: int,
        feature_dim: int,
        feature_dim_disc: int,
        pretrained_bert_model: str,
        # Training
        n_epochs: int,
        batch_size: int,
        lr: float,
        lr_min: float,
        workers: int,
        preserve_lr: bool,
        optim_sched_config: str,
        training_steps: int,
        use_fp16: bool,
        initial_clipping_norm: float,
        # Advanced training
        use_gradient_checkpointing: bool,
        use_lpips: bool,
        lambda_lpips: float,
        use_cfg: bool,
        cfg_dropout_prob: float,
        use_multires: bool,
        reduced_min_sizes: str,
        # Training modes
        train_vae: bool,
        train_no_gan: bool,
        train_spade: bool,
        train_diff: bool,
        train_diff_full: bool,
        # KL divergence
        kl_beta: float,
        kl_warmup: int,
        kl_free_bits: float,
        # Output
        lambda_adv: float,
        sample_interval: int,
        log_interval: int,
        no_samples: bool,
        test_image_address: str,
        sample_captions: str,
        # Misc
        tokenizer_name: str,
        img_size: int,
        channels: int,
    ) -> Tuple[str, str, str]:
        """Handle training start.

        Returns:
            Tuple of (status, start button label, stop button label)
        """
        if runner.is_running:
            return "⚠️ Training is already running", "⏸️ Training...", "⏹️ Stop Training"

        # Validate inputs
        if not use_webdataset:
            if not data_path or not Path(data_path).exists():
                return "❌ Invalid data path", "▶️ Start Training", "⏹️ Stop Training"
            if not captions_file or not Path(captions_file).exists():
                return "❌ Invalid captions file", "▶️ Start Training", "⏹️ Stop Training"
        else:
            if not webdataset_token or webdataset_token == "your_token_here":
                return "❌ Invalid HuggingFace token", "▶️ Start Training", "⏹️ Stop Training"
            if not webdataset_url:
                return "❌ Invalid WebDataset URL", "▶️ Start Training", "⏹️ Stop Training"

        # Build config
        config = {
            # Data
            "data_path": data_path,
            "captions_file": captions_file,
            "use_webdataset": use_webdataset,
            "webdataset_token": webdataset_token if webdataset_token else None,
            "webdataset_url": webdataset_url if webdataset_url else None,
            "webdataset_image_key": webdataset_image_key if webdataset_image_key else "jpg",
            "webdataset_caption_key": (
                webdataset_caption_key if webdataset_caption_key else "prompt"
            ),
            # Model
            "output_path": output_path,
            "model_checkpoint": model_checkpoint if model_checkpoint else None,
            "vae_dim": int(vae_dim),
            "text_embedding_dim": int(text_embedding_dim),
            "feature_maps_dim": int(feature_dim),
            "feature_maps_dim_disc": int(feature_dim_disc),
            "pretrained_bert_model": pretrained_bert_model if pretrained_bert_model else None,
            # Training
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "lr_min": float(lr_min),
            "workers": int(workers),
            "preserve_lr": preserve_lr,
            "optim_sched_config": optim_sched_config if optim_sched_config else None,
            "training_steps": int(training_steps),
            "use_fp16": use_fp16,
            "initial_clipping_norm": float(initial_clipping_norm),
            # Advanced training
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_lpips": use_lpips,
            "lambda_lpips": float(lambda_lpips),
            "cfg_dropout_prob": float(cfg_dropout_prob) if use_cfg else 0.0,
            "reduced_min_sizes": reduced_min_sizes if use_multires else "",
            # Training modes
            "train_vae": train_vae,
            "train_no_gan": train_no_gan,
            "train_spade": train_spade,
            "train_diff": train_diff,
            "train_diff_full": train_diff_full,
            # KL divergence
            "kl_beta": float(kl_beta),
            "kl_warmup_steps": int(kl_warmup),
            "kl_free_bits": float(kl_free_bits),
            # Output
            "lambda_adv": float(lambda_adv),
            "sample_interval": int(sample_interval),
            "log_interval": int(log_interval),
            "no_samples": no_samples,
            "test_image_address": test_image_address if test_image_address else None,
            "sample_captions": sample_captions if sample_captions else None,
            # Misc
            "tokenizer_name": tokenizer_name,
            "img_size": int(img_size),
            "channels": int(channels),
        }

        # Save config
        config_mgr.save_training_config(config)

        # Start training
        success = runner.start_training(config)

        if success:
            output_lines.clear()
            return "✅ Training started!", "⏸️ Training...", "⏹️ Stop Training"
        else:
            return "❌ Failed to start training", "▶️ Start Training", "⏹️ Stop Training"

    def stop_training_handler() -> Tuple[str, str, str]:
        """Handle training stop.

        Returns:
            Tuple of (status, start button label, stop button label)
        """
        if not runner.is_running:
            return "⚠️ No training is running", "▶️ Start Training", "⏹️ Stop Training"

        success = runner.stop_training()

        if success:
            return "✅ Training stopped", "▶️ Start Training", "⏹️ Stop Training"
        else:
            return "❌ Failed to stop training", "⏸️ Training...", "⏹️ Stop Training"

    def update_output() -> str:
        """Update console output.

        Returns:
            Console output text
        """
        new_lines = runner.get_output()
        output_lines.extend(new_lines)

        # Keep last 500 lines
        if len(output_lines) > 500:
            output_lines[:] = output_lines[-500:]

        return "\n".join(output_lines)

    # Load previous config
    prev_config = config_mgr.load_training_config()
    if not prev_config:
        prev_config = config_mgr.get_default_training_config()

    tab: gr.Tab
    with gr.Tab("🚀 Train") as tab:
        gr.Markdown("# FluxFlow Model Training")
        gr.Markdown(
            "Configure and train FluxFlow text-to-image models. "
            "See TRAINING_GUIDE.md for detailed documentation."
        )

        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### Dataset Configuration")

                use_webdataset_checkbox = gr.Checkbox(
                    label="Use WebDataset Streaming",
                    value=prev_config.get("use_webdataset", False),
                    info="Stream from HuggingFace datasets (no download needed)",
                )

                with gr.Column(
                    visible=not prev_config.get("use_webdataset", False)
                ) as local_data_group:
                    data_path_input = gr.Textbox(
                        label="Data Path",
                        placeholder="/path/to/images",
                        value=prev_config.get("data_path", ""),
                    )
                    captions_file_input = gr.Textbox(
                        label="Captions File",
                        placeholder="/path/to/captions.txt",
                        value=prev_config.get("captions_file", ""),
                    )

                with gr.Column(
                    visible=prev_config.get("use_webdataset", False)
                ) as webdataset_data_group:
                    webdataset_token_input = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="hf_your_token_here",
                        value=prev_config.get("webdataset_token", ""),
                        type="password",
                    )
                    webdataset_url_input = gr.Textbox(
                        label="WebDataset URL Pattern",
                        placeholder="hf://datasets/username/dataset/*.tar",
                        value=prev_config.get("webdataset_url", ""),
                        info="HuggingFace dataset URL with tar pattern",
                    )
                    with gr.Row():
                        webdataset_image_key_input = gr.Textbox(
                            label="Image Key",
                            value=prev_config.get("webdataset_image_key", "jpg"),
                            info="File extension in tar (e.g., jpg, png)",
                        )
                        webdataset_caption_key_input = gr.Textbox(
                            label="Caption Key",
                            value=prev_config.get("webdataset_caption_key", "prompt"),
                            info="JSON field for captions",
                        )

                output_path_input = gr.Textbox(
                    label="Output Path",
                    value=prev_config.get("output_path", "outputs/flux"),
                )
                model_checkpoint_input = gr.Textbox(
                    label="Resume from Checkpoint (optional)",
                    placeholder="outputs/flux/flxflow_final.safetensors",
                    value=prev_config.get("model_checkpoint", ""),
                )

                gr.Markdown("### Model Architecture")

                with gr.Row():
                    vae_dim_input = gr.Number(
                        label="VAE Dimension",
                        value=prev_config.get("vae_dim", 64),
                        precision=0,
                        info="32=Low VRAM, 64=Balanced, 128=High Quality, 256=Max",
                    )
                    text_embedding_dim_input = gr.Number(
                        label="Text Embedding Dim",
                        value=prev_config.get("text_embedding_dim", 1024),
                        precision=0,
                    )

                with gr.Row():
                    feature_dim_input = gr.Number(
                        label="Feature Dimension",
                        value=prev_config.get("feature_maps_dim", 64),
                        precision=0,
                    )
                    feature_dim_disc_input = gr.Number(
                        label="Discriminator Dim",
                        value=prev_config.get("feature_maps_dim_disc", 128),
                        precision=0,
                    )

                pretrained_bert_input = gr.Textbox(
                    label="Pretrained BERT (optional)",
                    placeholder="path/to/bert_checkpoint.safetensors",
                    value=prev_config.get("pretrained_bert_model", ""),
                )

                gr.Markdown("### Training Parameters")

                with gr.Row():
                    n_epochs_input = gr.Number(
                        label="Epochs",
                        value=prev_config.get("n_epochs", 10),
                        precision=0,
                    )
                    batch_size_input = gr.Number(
                        label="Batch Size",
                        value=prev_config.get("batch_size", 1),
                        precision=0,
                    )

                with gr.Row():
                    lr_input = gr.Number(
                        label="Learning Rate",
                        value=prev_config.get("lr", 1e-5),
                        step=1e-6,
                        info="VAE: 1e-5, Flow: 5e-7",
                    )
                    lr_min_input = gr.Number(
                        label="Min LR Multiplier",
                        value=prev_config.get("lr_min", 0.1),
                        step=0.01,
                    )

                with gr.Row():
                    workers_input = gr.Number(
                        label="Workers",
                        value=prev_config.get("workers", 4),
                        precision=0,
                    )
                    training_steps_input = gr.Number(
                        label="Training Steps (Grad Accum)",
                        value=prev_config.get("training_steps", 1),
                        precision=0,
                        info="Effective batch = batch_size * training_steps",
                    )

                with gr.Row():
                    initial_clipping_norm_input = gr.Number(
                        label="Gradient Clipping Norm",
                        value=prev_config.get("initial_clipping_norm", 1.0),
                        step=0.1,
                    )
                    use_fp16_checkbox = gr.Checkbox(
                        label="Use FP16 (Mixed Precision)",
                        value=prev_config.get("use_fp16", False),
                        info="~40% faster on RTX GPUs",
                    )
                    preserve_lr_checkbox = gr.Checkbox(
                        label="Preserve LR",
                        value=prev_config.get("preserve_lr", False),
                    )

                optim_sched_config_input = gr.Textbox(
                    label="Optimizer/Scheduler Config (optional)",
                    placeholder="path/to/optim_config.json",
                    value=prev_config.get("optim_sched_config", ""),
                    info="Advanced: JSON file for per-model optimizer/scheduler config",
                )

                gr.Markdown("### Training Modes")

                with gr.Row():
                    train_vae_checkbox = gr.Checkbox(
                        label="Train VAE",
                        value=prev_config.get("train_vae", True),
                    )
                    train_no_gan_checkbox = gr.Checkbox(
                        label="Disable GAN",
                        value=prev_config.get("train_no_gan", False),
                    )
                    train_spade_checkbox = gr.Checkbox(
                        label="Use SPADE",
                        value=prev_config.get("train_spade", True),
                    )

                with gr.Row():
                    train_diff_checkbox = gr.Checkbox(
                        label="Train Flow",
                        value=prev_config.get("train_diff", False),
                    )
                    train_diff_full_checkbox = gr.Checkbox(
                        label="Train Flow (Full)",
                        value=prev_config.get("train_diff_full", False),
                    )

                with gr.Accordion("KL Divergence Settings", open=False):
                    kl_beta_input = gr.Slider(
                        label="KL Beta",
                        minimum=0.0,
                        maximum=5.0,
                        step=0.0001,
                        value=prev_config.get("kl_beta", 1.0),
                        info="Regularization strength",
                    )
                    kl_warmup_input = gr.Number(
                        label="KL Warmup Steps",
                        value=prev_config.get("kl_warmup_steps", 100000),
                        precision=0,
                    )
                    kl_free_bits_input = gr.Number(
                        label="KL Free Bits (nats)",
                        value=prev_config.get("kl_free_bits", 0.0),
                        step=0.1,
                        info="Minimum KL before penalty",
                    )

                with gr.Accordion("Output & Sampling Settings", open=False):
                    lambda_adv_input = gr.Slider(
                        label="Adversarial Weight",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=prev_config.get("lambda_adv", 0.9),
                    )
                    sample_interval_input = gr.Number(
                        label="Sample Interval (batches)",
                        value=prev_config.get("sample_interval", 50),
                        precision=0,
                    )
                    log_interval_input = gr.Number(
                        label="Log Interval (batches)",
                        value=prev_config.get("log_interval", 10),
                        precision=0,
                    )
                    no_samples_checkbox = gr.Checkbox(
                        label="Disable Sampling (faster training)",
                        value=prev_config.get("no_samples", False),
                    )
                    test_image_address_input = gr.Textbox(
                        label="Test Images for VAE (space-separated paths)",
                        placeholder="test1.jpg test2.png",
                        value=prev_config.get("test_image_address", ""),
                    )
                    sample_captions_input = gr.Textbox(
                        label="Sample Captions for Flow (one per line)",
                        placeholder="a photo of a cat\nan illustration of mountains",
                        value=prev_config.get("sample_captions", "a photo of a cat"),
                        lines=3,
                    )

                with gr.Accordion("Advanced Training Parameters", open=False):
                    with gr.Row():
                        use_gradient_checkpointing_checkbox = gr.Checkbox(
                            label="Gradient Checkpointing",
                            value=prev_config.get("use_gradient_checkpointing", False),
                            info="Save VRAM at cost of speed",
                        )
                        use_lpips_checkbox = gr.Checkbox(
                            label="Use LPIPS Perceptual Loss",
                            value=prev_config.get("use_lpips", True),
                            info="Improves perceptual quality (adds ~2GB VRAM)",
                        )

                    lambda_lpips_input = gr.Slider(
                        label="LPIPS Weight",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=prev_config.get("lambda_lpips", 0.1),
                        info="Perceptual loss strength (0.1 recommended)",
                    )

                with gr.Accordion("Classifier-Free Guidance (CFG)", open=False):
                    gr.Markdown(
                        """
                        **CFG Training** enables guided generation at inference time.
                        Train with CFG dropout to support guidance_scale > 1.0.
                        """
                    )
                    use_cfg_checkbox = gr.Checkbox(
                        label="Enable CFG Training",
                        value=prev_config.get("cfg_dropout_prob", 0.0) > 0.0,
                        info="Required for guided generation",
                    )
                    cfg_dropout_prob_input = gr.Slider(
                        label="CFG Dropout Probability",
                        minimum=0.0,
                        maximum=0.20,
                        step=0.01,
                        value=prev_config.get("cfg_dropout_prob", 0.10),
                        info="Recommended: 0.10 (10% null conditioning)",
                        visible=prev_config.get("cfg_dropout_prob", 0.0) > 0.0,
                    )

                    def toggle_cfg(enabled):
                        return gr.update(visible=enabled)

                    use_cfg_checkbox.change(
                        fn=toggle_cfg,
                        inputs=[use_cfg_checkbox],
                        outputs=[cfg_dropout_prob_input],
                    )

                with gr.Accordion("Multi-Resolution Training", open=False):
                    gr.Markdown(
                        """
                        **Progressive resolution** trains on smaller images first,
                        then gradually increases. Improves convergence and reduces
                        initial VRAM usage.
                        """
                    )
                    use_multires_checkbox = gr.Checkbox(
                        label="Enable Progressive Resolution",
                        value=bool(prev_config.get("reduced_min_sizes", "")),
                    )
                    reduced_min_sizes_input = gr.Textbox(
                        label="Resolution Stages (comma-separated)",
                        placeholder="256, 384, 512, 768, 1024",
                        value=prev_config.get("reduced_min_sizes", ""),
                        info="Trains progressively from smallest to img_size",
                        visible=bool(prev_config.get("reduced_min_sizes", "")),
                    )

                    def toggle_multires(enabled):
                        return gr.update(visible=enabled)

                    use_multires_checkbox.change(
                        fn=toggle_multires,
                        inputs=[use_multires_checkbox],
                        outputs=[reduced_min_sizes_input],
                    )

                with gr.Accordion("Misc Settings", open=False):
                    tokenizer_name_input = gr.Textbox(
                        label="Tokenizer Name",
                        value=prev_config.get("tokenizer_name", "distilbert-base-uncased"),
                    )
                    img_size_input = gr.Number(
                        label="Image Size (px)",
                        value=prev_config.get("img_size", 1024),
                        precision=0,
                        info="Images resized to this size",
                    )
                    channels_input = gr.Number(
                        label="Image Channels",
                        value=prev_config.get("channels", 3),
                        precision=0,
                        info="3=RGB, 1=Grayscale",
                    )

                with gr.Row():
                    start_btn = gr.Button("▶️ Start Training", variant="primary", scale=3)
                    stop_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)

                status_output = gr.Textbox(
                    label="Status",
                    value="Ready to train",
                    interactive=False,
                )

            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### Training Console")
                console_output = gr.Textbox(
                    label="Output",
                    lines=30,
                    max_lines=30,
                    interactive=False,
                    autoscroll=True,
                )

                gr.Markdown("### Quick Guide")
                gr.Markdown(
                    """
                **Training Stages:**
                1. **VAE Pretraining** (50-100 epochs)
                   - Enable: Train VAE, Use SPADE
                   - LR: 1e-5, Disable: Train Flow
                2. **Flow Training** (100-200 epochs)
                   - Enable: Train Flow (Full)
                   - LR: 5e-7, Load VAE checkpoint
                3. **Joint Fine-tuning** (20-50 epochs, optional)
                   - Enable: Train VAE, Train Flow (Full), Use SPADE
                   - LR: 1e-6

                **VRAM Settings:**
                - **8GB**: dim=32, batch=1, fp16=on, img_size=512
                - **12-16GB**: dim=64, batch=2, fp16=on
                - **24GB+**: dim=128, batch=4, fp16=on

                **Advanced Features:**
                - **Optimizer/Scheduler Config**: Use custom optimizers (Lion, AdamW, etc.)
                  and schedulers per model via JSON file
                - See TRAINING_GUIDE.md for complete documentation and examples

                **Tips:**
                - Monitor console for loss values
                - Use FP16 for 2-4x speedup on RTX GPUs
                - TTI-2M streams 2M+ images (no download needed)
                - Default optimizers: Lion (flow), AdamW (vae/discriminator)
                """
                )

        # Toggle dataset inputs based on use_webdataset
        def toggle_dataset_inputs(use_webdataset):
            return gr.update(visible=not use_webdataset), gr.update(visible=use_webdataset)

        use_webdataset_checkbox.change(
            fn=toggle_dataset_inputs,
            inputs=[use_webdataset_checkbox],
            outputs=[local_data_group, webdataset_data_group],
        )

        # Event handlers
        start_btn.click(
            fn=start_training_handler,
            inputs=[
                # Data
                data_path_input,
                captions_file_input,
                use_webdataset_checkbox,
                webdataset_token_input,
                webdataset_url_input,
                webdataset_image_key_input,
                webdataset_caption_key_input,
                # Model
                output_path_input,
                model_checkpoint_input,
                vae_dim_input,
                text_embedding_dim_input,
                feature_dim_input,
                feature_dim_disc_input,
                pretrained_bert_input,
                # Training
                n_epochs_input,
                batch_size_input,
                lr_input,
                lr_min_input,
                workers_input,
                preserve_lr_checkbox,
                optim_sched_config_input,
                training_steps_input,
                use_fp16_checkbox,
                initial_clipping_norm_input,
                # Advanced training
                use_gradient_checkpointing_checkbox,
                use_lpips_checkbox,
                lambda_lpips_input,
                use_cfg_checkbox,
                cfg_dropout_prob_input,
                use_multires_checkbox,
                reduced_min_sizes_input,
                # Training modes
                train_vae_checkbox,
                train_no_gan_checkbox,
                train_spade_checkbox,
                train_diff_checkbox,
                train_diff_full_checkbox,
                # KL divergence
                kl_beta_input,
                kl_warmup_input,
                kl_free_bits_input,
                # Output
                lambda_adv_input,
                sample_interval_input,
                log_interval_input,
                no_samples_checkbox,
                test_image_address_input,
                sample_captions_input,
                # Misc
                tokenizer_name_input,
                img_size_input,
                channels_input,
            ],
            outputs=[status_output, start_btn, stop_btn],
        )

        stop_btn.click(
            fn=stop_training_handler,
            outputs=[status_output, start_btn, stop_btn],
        )

        # Auto-update console every 1 second
        console_output.change(
            fn=lambda: None,
            outputs=[],
            every=1,
        )

        demo_timer = gr.Timer(value=1, active=True)
        demo_timer.tick(
            fn=update_output,
            outputs=[console_output],
        )

    return tab

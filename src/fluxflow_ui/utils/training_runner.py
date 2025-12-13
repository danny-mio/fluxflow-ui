"""Background training runner for UI."""

import atexit
import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingRunner:
    """Run training in background thread with live output."""

    def __init__(self):
        """Initialize training runner."""
        self._lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self._is_running = False
        self.thread: Optional[threading.Thread] = None
        self.temp_config_file: Optional[str] = None

        # Register cleanup on exit
        atexit.register(self._cleanup)

    @property
    def is_running(self) -> bool:
        """Thread-safe check if training is running."""
        with self._lock:
            return self._is_running

    def _cleanup(self) -> None:
        """Cleanup temp files on exit."""
        if self.temp_config_file and os.path.exists(self.temp_config_file):
            try:
                os.remove(self.temp_config_file)
                logger.debug(f"Cleaned up temp config: {self.temp_config_file}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temp config: {e}")

    def _build_data_args(self, config: Dict[str, Any]) -> List[str]:
        """Build data-related command arguments."""
        cmd = []
        if config.get("data_path"):
            cmd.extend(["--data_path", config["data_path"]])
        if config.get("captions_file"):
            cmd.extend(["--captions_file", config["captions_file"]])

        # WebDataset support (new)
        if config.get("use_webdataset"):
            cmd.append("--use_webdataset")
            if config.get("webdataset_token"):
                cmd.extend(["--webdataset_token", config["webdataset_token"]])
            if config.get("webdataset_url"):
                cmd.extend(["--webdataset_url", config["webdataset_url"]])
            if config.get("webdataset_image_key"):
                cmd.extend(["--webdataset_image_key", config["webdataset_image_key"]])
            if config.get("webdataset_caption_key"):
                cmd.extend(["--webdataset_caption_key", config["webdataset_caption_key"]])

        # Legacy TTI-2M support (deprecated, maps to webdataset)
        elif config.get("use_tt2m"):
            cmd.append("--use_tt2m")
            if config.get("tt2m_token"):
                cmd.extend(["--tt2m_token", config["tt2m_token"]])

        return cmd

    def _build_model_args(self, config: Dict[str, Any]) -> List[str]:
        """Build model-related command arguments."""
        cmd = []
        if config.get("output_path"):
            cmd.extend(["--output_path", config["output_path"]])
        if config.get("model_checkpoint"):
            cmd.extend(["--model_checkpoint", config["model_checkpoint"]])
        if config.get("pretrained_bert_model"):
            cmd.extend(["--pretrained_bert_model", config["pretrained_bert_model"]])
        cmd.extend(
            [
                "--vae_dim",
                str(config.get("vae_dim", 64)),
                "--text_embedding_dim",
                str(config.get("text_embedding_dim", 1024)),
                "--feature_maps_dim",
                str(config.get("feature_maps_dim", 64)),
                "--feature_maps_dim_disc",
                str(config.get("feature_maps_dim_disc", 128)),
            ]
        )
        return cmd

    def _build_training_args(self, config: Dict[str, Any]) -> List[str]:
        """Build training-related command arguments."""
        cmd = [
            "--n_epochs",
            str(config.get("n_epochs", 10)),
            "--batch_size",
            str(config.get("batch_size", 1)),
            "--lr",
            str(config.get("lr", 1e-5)),
            "--lr_min",
            str(config.get("lr_min", 0.1)),
            "--workers",
            str(config.get("workers", 4)),
            "--training_steps",
            str(config.get("training_steps", 1)),
            "--initial_clipping_norm",
            str(config.get("initial_clipping_norm", 1.0)),
        ]
        # Training flags
        if config.get("preserve_lr"):
            cmd.append("--preserve_lr")
        if config.get("use_fp16"):
            cmd.append("--use_fp16")
        if config.get("use_gradient_checkpointing"):
            cmd.append("--use_gradient_checkpointing")

        # Training modes
        for flag in ["train_vae", "train_no_gan", "train_spade", "train_diff", "train_diff_full"]:
            if config.get(flag):
                cmd.append(f"--{flag}")

        # Advanced loss options
        if config.get("use_lpips"):
            cmd.append("--use_lpips")
            cmd.extend(["--lambda_lpips", str(config.get("lambda_lpips", 0.1))])

        # CFG training
        if config.get("cfg_dropout_prob", 0.0) > 0.0:
            cmd.extend(["--cfg_dropout_prob", str(config["cfg_dropout_prob"])])

        # Multi-resolution training
        if config.get("reduced_min_sizes"):
            sizes_str = config["reduced_min_sizes"].strip()
            if sizes_str:
                # Parse comma-separated list
                sizes = [s.strip() for s in sizes_str.split(",")]
                for size in sizes:
                    cmd.extend(["--reduced_min_sizes", size])

        # KL divergence
        cmd.extend(
            [
                "--kl_beta",
                str(config.get("kl_beta", 1.0)),
                "--kl_warmup_steps",
                str(config.get("kl_warmup_steps", 100000)),
                "--kl_free_bits",
                str(config.get("kl_free_bits", 0.0)),
            ]
        )
        return cmd

    def _build_output_args(self, config: Dict[str, Any]) -> List[str]:
        """Build output and logging command arguments."""
        cmd = [
            "--lambda_adv",
            str(config.get("lambda_adv", 0.9)),
            "--log_interval",
            str(config.get("log_interval", 10)),
        ]
        if config.get("no_samples"):
            cmd.append("--no_samples")
        if config.get("test_image_address"):
            for img in config["test_image_address"].strip().split():
                cmd.extend(["--test_image_address", img])
        if config.get("sample_captions"):
            for caption in config["sample_captions"].strip().split("\n"):
                if caption.strip():
                    cmd.extend(["--sample_captions", caption.strip()])
        # Misc
        cmd.extend(
            [
                "--tokenizer_name",
                config.get("tokenizer_name", "distilbert-base-uncased"),
                "--img_size",
                str(config.get("img_size", 1024)),
                "--channels",
                str(config.get("channels", 3)),
            ]
        )
        return cmd

    def _build_optim_sched_args(self, config: Dict[str, Any]) -> List[str]:
        """Build optimizer/scheduler config arguments."""
        cmd = []
        # Option 1: User provides a path to an existing config file
        if config.get("optim_sched_config") and Path(config["optim_sched_config"]).exists():
            cmd.extend(["--optim_sched_config", config["optim_sched_config"]])
        # Option 2: UI provides optimizers/schedulers dicts
        elif "optimizers" in config and "schedulers" in config:
            optim_sched_config = {
                "optimizers": config["optimizers"],
                "schedulers": config["schedulers"],
            }
            output_path = config.get("output_path", "outputs")
            os.makedirs(output_path, exist_ok=True)
            self.temp_config_file = str(Path(output_path) / "optim_sched_config.json")
            with open(self.temp_config_file, "w") as f:
                json.dump(optim_sched_config, f, indent=2)
            cmd.extend(["--optim_sched_config", self.temp_config_file])
        return cmd

    def start_training(
        self, config: Dict[str, Any], on_output: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Start training process in background.

        Args:
            config: Training configuration dictionary (or pipeline YAML path)
            on_output: Callback for each line of output

        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._is_running:
                return False

            # Check if using pipeline mode (YAML config)
            if config.get("pipeline_yaml_content"):
                cmd = self._build_pipeline_command(config)
            else:
                # Build command using helper methods (simple mode)
                cmd = ["fluxflow-train"]
                cmd.extend(self._build_data_args(config))
                cmd.extend(self._build_model_args(config))
                cmd.extend(self._build_training_args(config))
                cmd.extend(self._build_output_args(config))
                cmd.extend(self._build_optim_sched_args(config))

            # Start process
            try:
                logger.info(f"Starting training with command: {' '.join(cmd)}")
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )
                self._is_running = True

                # Start output reader thread
                self.thread = threading.Thread(
                    target=self._read_output,
                    args=(on_output,),
                    daemon=True,
                )
                self.thread.start()
                logger.info("Training process started")
                return True

            except FileNotFoundError as e:
                logger.error(f"Training command not found: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to start training: {e}")
                return False

    def _build_pipeline_command(self, config: Dict[str, Any]) -> List[str]:
        """Build command for pipeline mode training.

        Args:
            config: Config with pipeline_yaml_content

        Returns:
            Command list
        """
        # Save YAML to temp file
        yaml_content = config["pipeline_yaml_content"]
        output_path = config.get("output_path", "outputs")
        os.makedirs(output_path, exist_ok=True)

        self.temp_config_file = str(Path(output_path) / "pipeline_config.yaml")
        with open(self.temp_config_file, "w") as f:
            f.write(yaml_content)

        # Build command with --config flag
        cmd = ["fluxflow-train", "--config", self.temp_config_file]
        return cmd

    def _read_output(self, on_output: Optional[Callable[[str], None]]) -> None:
        """Read process output line by line.

        Args:
            on_output: Callback for each line
        """
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            line = line.rstrip()
            self.output_queue.put(line)
            if on_output:
                on_output(line)

        self.process.wait()
        with self._lock:
            self._is_running = False
        logger.info("Training process finished")

    def stop_training(self) -> bool:
        """Stop training process.

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if not self._is_running or not self.process:
                return False

            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                self._is_running = False
                logger.info("Training stopped by user")
                return True
            except subprocess.TimeoutExpired:
                self.process.kill()
                self._is_running = False
                logger.warning("Training process killed after timeout")
                return True
            except Exception as e:
                logger.error(f"Failed to stop training: {e}")
                return False

    def get_output(self) -> List[str]:
        """Get all queued output lines.

        Returns:
            List of output lines
        """
        lines = []
        while not self.output_queue.empty():
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

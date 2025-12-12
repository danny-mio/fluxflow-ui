"""Configuration management for UI."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage training and generation configurations."""

    def __init__(self, config_dir: str = ".ui_configs"):
        """Initialize config manager.

        Args:
            config_dir: Directory to store config files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.training_config_path = self.config_dir / "training_config.json"
        self.generation_config_path = self.config_dir / "generation_config.json"

    def save_training_config(self, config: Dict[str, Any]) -> bool:
        """Save training configuration.

        Args:
            config: Training configuration dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.training_config_path, "w") as f:
                json.dump(config, f, indent=2)
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save training config: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid training config data: {e}")
            return False

    def load_training_config(self) -> Optional[Dict[str, Any]]:
        """Load training configuration.

        Returns:
            Training configuration dict or None if not found/invalid
        """
        if not self.training_config_path.exists():
            return None
        try:
            with open(self.training_config_path, "r") as f:
                data: Dict[str, Any] = json.load(f)
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted training config file: {e}")
            return None
        except (IOError, OSError) as e:
            logger.error(f"Failed to read training config: {e}")
            return None

    def save_generation_config(self, config: Dict[str, Any]) -> bool:
        """Save generation configuration.

        Args:
            config: Generation configuration dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.generation_config_path, "w") as f:
                json.dump(config, f, indent=2)
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save generation config: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid generation config data: {e}")
            return False

    def load_generation_config(self) -> Optional[Dict[str, Any]]:
        """Load generation configuration.

        Returns:
            Generation configuration dict or None if not found/invalid
        """
        if not self.generation_config_path.exists():
            return None
        try:
            with open(self.generation_config_path, "r") as f:
                data: Dict[str, Any] = json.load(f)
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted generation config file: {e}")
            return None
        except (IOError, OSError) as e:
            logger.error(f"Failed to read generation config: {e}")
            return None

    def get_default_training_config(self) -> Dict[str, Any]:
        """Get default training configuration.

        Returns:
            Default training configuration
        """
        return {
            # Data
            "data_path": "",
            "captions_file": "",
            "use_webdataset": False,
            "webdataset_token": "",
            "webdataset_url": "",
            "webdataset_image_key": "jpg",
            "webdataset_caption_key": "prompt",
            # Legacy (deprecated but kept for compatibility)
            "use_tt2m": False,
            "tt2m_token": "",
            # Model
            "output_path": "outputs/flux",
            "model_checkpoint": "",
            "vae_dim": 64,
            "text_embedding_dim": 1024,
            "feature_maps_dim": 64,
            "feature_maps_dim_disc": 128,
            "pretrained_bert_model": "",
            # Training
            "n_epochs": 10,
            "batch_size": 1,
            "lr": 1e-5,
            "lr_min": 0.1,
            "workers": 4,
            "preserve_lr": False,
            "optim_sched_config": "",
            "training_steps": 1,
            "use_fp16": False,
            "initial_clipping_norm": 1.0,
            # Advanced training
            "use_gradient_checkpointing": False,
            "use_lpips": True,
            "lambda_lpips": 0.1,
            "cfg_dropout_prob": 0.0,
            "reduced_min_sizes": "",
            # Training modes
            "train_vae": True,
            "train_no_gan": False,
            "train_spade": True,
            "train_diff": False,
            "train_diff_full": False,
            # KL divergence
            "kl_beta": 1.0,
            "kl_warmup_steps": 100000,
            "kl_free_bits": 0.0,
            # Output
            "lambda_adv": 0.9,
            "sample_interval": 50,
            "log_interval": 10,
            "no_samples": False,
            "test_image_address": "",
            "sample_captions": "a photo of a cat",
            # Misc
            "tokenizer_name": "distilbert-base-uncased",
            "img_size": 1024,
            "channels": 3,
            # Optimizer/Scheduler configurations
            "optimizers": {
                "flow": {
                    "type": "Lion",
                    "lr": 5e-7,
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                    "decoupled_weight_decay": True,
                },
                "vae": {
                    "type": "AdamW",
                    "lr": 5e-7,
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                },
                "text_encoder": {
                    "type": "AdamW",
                    "lr": 5e-8,
                    "betas": [0.9, 0.99],
                    "weight_decay": 0.01,
                },
                "discriminator": {
                    "type": "AdamW",
                    "lr": 5e-7,
                    "betas": [0.0, 0.9],
                    "weight_decay": 0.001,
                    "amsgrad": True,
                },
            },
            "schedulers": {
                "flow": {
                    "type": "CosineAnnealingLR",
                    "eta_min_factor": 0.1,
                },
                "vae": {
                    "type": "CosineAnnealingLR",
                    "eta_min_factor": 0.1,
                },
                "text_encoder": {
                    "type": "CosineAnnealingLR",
                    "eta_min_factor": 0.001,
                },
                "discriminator": {
                    "type": "CosineAnnealingLR",
                    "eta_min_factor": 0.1,
                },
            },
        }

    def get_default_generation_config(self) -> Dict[str, Any]:
        """Get default generation configuration.

        Returns:
            Default generation configuration
        """
        return {
            "model_checkpoint": "",
            "output_path": "generated",
            "img_size": 512,
            "ddim_steps": 50,
            "batch_size": 1,
            "vae_dim": 64,
            "feature_maps_dim": 64,
            "text_embedding_dim": 1024,
            # CFG parameters
            "use_cfg": False,
            "guidance_scale": 5.0,
            "negative_prompt": "",
        }

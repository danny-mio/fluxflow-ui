"""Generation worker for UI."""

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import safetensors.torch
import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fluxflow.models import (  # noqa: E402
    BertTextEncoder,
    FluxCompressor,
    FluxExpander,
    FluxFlowProcessor,
    FluxPipeline,
    detect_architecture_version,
)
from fluxflow.models.versioning import load_versioned_checkpoint  # noqa: E402
from fluxflow.utils import (  # noqa: E402
    build_cfg_null_pair,
    generate_latent_images,
    img_to_random_packet,
)


class GenerationWorker:
    """Worker for generating images from text prompts."""

    def __init__(self):
        """Initialize generation worker."""
        self.model_checkpoint: Optional[str] = None
        self.diffuser: Optional[FluxPipeline] = None
        self.text_encoder: Optional[BertTextEncoder] = None
        self.tokenizer: Any = None
        self.device = self._get_device()
        self.config = {}

    def _get_device(self) -> torch.device:
        """Get available device (CUDA/ROCm > MPS > CPU).

        Returns:
            torch device
        """
        from fluxflow.utils.device import get_device as _core_get_device

        return _core_get_device()

    def _detect_display_version(self, checkpoint_path: str) -> str:
        """Re-detect architecture version from a checkpoint's own keys, for display only.

        No `.version` attribute is ever set on a loaded FluxPipeline, so this
        re-reads the state dict and runs the same key-marker detection used
        during loading. Display-only; failures fall back to "unknown".
        """
        try:
            cp = Path(checkpoint_path)
            weights_path = cp
            if cp.is_dir():
                weights_path = cp / "model.safetensors"
                if not weights_path.exists():
                    weights_path = cp / "flxflow_final.safetensors"
            state_dict = safetensors.torch.load_file(str(weights_path))
            return detect_architecture_version(list(state_dict.keys()))
        except Exception:
            return "unknown"

    def load_model(  # noqa: C901
        self,
        checkpoint_path: str,
        vae_dim: int = 64,
        feature_maps_dim: int = 64,
        text_embedding_dim: int = 1024,
        text_encoder_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load model from checkpoint with automatic version detection.

        Args:
            checkpoint_path: Path to model checkpoint (versioned directory or legacy file)
            vae_dim: VAE latent dimension (ignored for versioned checkpoints)
            feature_maps_dim: Flow processor dimension (ignored for versioned checkpoints)
            text_embedding_dim: Text embedding dimension (ignored for versioned checkpoints)
            text_encoder_path: Optional explicit path to text-encoder weights
                (.safetensors), overriding both the sibling file and any
                bundled copy inside the main checkpoint.

        Returns:
            Tuple of (success, message)
        """
        try:
            if not Path(checkpoint_path).exists():
                return False, f"Checkpoint not found: {checkpoint_path}"

            # Try versioned loading first (new format)
            try:
                if Path(checkpoint_path).is_dir():
                    # Versioned checkpoint (directory with metadata)
                    self.pipeline = load_versioned_checkpoint(
                        Path(checkpoint_path), str(self.device)
                    )
                    self.diffuser = self.pipeline

                    # Load text_encoder separately (not included in versioned checkpoints)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    self.text_encoder = BertTextEncoder(embed_dim=1024)  # Default for UI
                    self.text_encoder.load_with_override(
                        checkpoint_path, override_path=text_encoder_path
                    )

                    # Ensure models are on device
                    self.pipeline.to(self.device).eval()
                    self.text_encoder.to(self.device).eval()

                    # MPS-specific initialization
                    if self.device.type == "mps":
                        import torch

                        torch.mps.empty_cache()

                    # Extract version info for display (no .version attribute is
                    # ever set on a loaded FluxPipeline, so re-detect from the
                    # checkpoint's own state-dict key markers).
                    version_info = self._detect_display_version(checkpoint_path)
                    model_info = f"Version {version_info} model"
                else:
                    # Try legacy versioned loading (single file without metadata)
                    try:
                        self.pipeline = load_versioned_checkpoint(
                            Path(checkpoint_path), str(self.device)
                        )
                        self.diffuser = self.pipeline

                        # Load text_encoder separately (not included in versioned checkpoints)
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
                        )
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                        self.text_encoder = BertTextEncoder(embed_dim=1024)  # Default for UI
                        self.text_encoder.load_with_override(
                            checkpoint_path, override_path=text_encoder_path
                        )

                        # Ensure models are on device
                        self.pipeline.to(self.device).eval()
                        self.text_encoder.to(self.device).eval()

                        # MPS-specific initialization
                        if self.device.type == "mps":
                            import torch

                            torch.mps.empty_cache()

                        version_info = self._detect_display_version(checkpoint_path)
                        model_info = f"Legacy versioned model (v{version_info})"
                    except Exception:
                        # Fall back to manual loading with default v0.3.0
                        return self._load_legacy_model(
                            checkpoint_path,
                            vae_dim,
                            feature_maps_dim,
                            text_embedding_dim,
                            text_encoder_path,
                        )

                model_info = f"Version {version_info} model"

            except Exception as versioned_error:
                # Fall back to manual loading with default v0.3.0
                print(
                    f"Versioned loading failed ({versioned_error}), falling back to legacy v0.3.0"
                )
                try:
                    return self._load_legacy_model(
                        checkpoint_path,
                        vae_dim,
                        feature_maps_dim,
                        text_embedding_dim,
                        text_encoder_path,
                    )
                except Exception as legacy_error:
                    # Try one more fallback: assume v0.7.0 architecture
                    print(
                        f"Legacy loading failed ({legacy_error}), "
                        "trying v0.7.0 architecture fallback"
                    )
                    try:
                        return self._load_v070_fallback(
                            checkpoint_path,
                            vae_dim,
                            feature_maps_dim,
                            text_embedding_dim,
                            text_encoder_path,
                        )
                    except Exception as v070_error:
                        # If all methods fail, provide comprehensive error message
                        return False, (
                            f"Failed to load model with all available methods.\n\n"
                            f"Versioned loading error: {versioned_error}\n"
                            f"Legacy (v0.3.0) loading error: {legacy_error}\n"
                            f"v0.7.0 fallback error: {v070_error}\n\n"
                            f"This likely indicates:\n"
                            "1. The model was trained with custom architecture "
                            "not matching any known version\n"
                            "2. The checkpoint file may be corrupted\n"
                            "3. Model dimensions don't match expected values\n\n"
                            f"Try adjusting vae_dim ({vae_dim}) and "
                            f"feature_maps_dim ({feature_maps_dim}) parameters.\n"
                            "For v0.7.0+ models, ensure the checkpoint was saved "
                            "with proper metadata."
                        )

            # Load tokenizer (shared for all versions)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            self.model_checkpoint = checkpoint_path

            # Extract config info for display
            try:
                vae_dim = int(self.diffuser.compressor.d_model)
            except (AttributeError, TypeError):
                pass
            try:
                feature_maps_dim = int(self.diffuser.flow_processor.d_model)
            except (AttributeError, TypeError):
                pass
            try:
                text_embedding_dim = int(self.text_encoder.embed_dim)  # type: ignore[arg-type]
            except (AttributeError, TypeError):
                pass

            self.config = {
                "vae_dim": vae_dim,
                "feature_maps_dim": feature_maps_dim,
                "text_embedding_dim": text_embedding_dim,
                "version": version_info if "version_info" in locals() else "0.3.0",
                "model_info": model_info,
            }

            success_msg = f"Model loaded successfully on {self.device} ({model_info})"
            return True, success_msg

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def _load_legacy_model(
        self,
        checkpoint_path: str,
        vae_dim: int,
        feature_maps_dim: int,
        text_embedding_dim: int,
        text_encoder_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load model using legacy manual instantiation with automatic architecture detection."""
        # First, inspect checkpoint to detect architecture
        detected_version = "0.3.0"
        try:
            import safetensors.torch

            state_dict = safetensors.torch.load_file(checkpoint_path)
            keys = list(state_dict.keys())
            detected_version = detect_architecture_version(keys)
        except Exception as inspect_error:
            print(f"Checkpoint inspection failed: {inspect_error}")

        if detected_version == "0.10.0":
            print("Detected v0.10.0 features in checkpoint, using v0.10.0 components")
            try:
                return self._load_v100_fallback(
                    checkpoint_path,
                    vae_dim,
                    feature_maps_dim,
                    text_embedding_dim,
                    text_encoder_path,
                )
            except Exception as v100_error:
                return False, (
                    f"Failed to load v0.10.0 model. The checkpoint contains v0.10.0 architecture "
                    f"but loading failed: {str(v100_error)}\n\n"
                    f"This checkpoint requires proper metadata. Please re-save the model using "
                    f"'save_versioned_checkpoint()' to add version information."
                )

        # If v0.7.0 features detected, use v0.7.0 loading only
        if detected_version == "0.7.0":
            print("Detected v0.7.0 features in checkpoint, using v0.7.0 components only")
            try:
                return self._load_v070_fallback(
                    checkpoint_path,
                    vae_dim,
                    feature_maps_dim,
                    text_embedding_dim,
                    text_encoder_path,
                )
            except Exception as v070_error:
                return False, (
                    f"Failed to load v0.7.0 model. The checkpoint contains v0.7.0 architecture "
                    f"but loading failed: {str(v070_error)}\n\n"
                    f"This checkpoint requires proper metadata. Please re-save the model using "
                    f"'save_versioned_checkpoint()' to add version information."
                )

        # Fall back to v0.3.0 loading for older models
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Calculate appropriate attention heads to ensure d_model is divisible
            def get_valid_n_head(d_model, preferred_heads=8):
                """Get number of heads that evenly divides d_model."""
                if d_model % preferred_heads == 0:
                    return preferred_heads
                # Find largest divisor that keeps heads reasonable
                for heads in range(preferred_heads, 0, -1):
                    if d_model % heads == 0:
                        return heads
                return 1  # Fallback, though this shouldn't happen

            # Use flexible attention heads for flow processor (main source of the error)
            flow_attn_heads = get_valid_n_head(feature_maps_dim)

            # Initialize models manually (v0.3.0 defaults)
            self.text_encoder = BertTextEncoder(embed_dim=text_embedding_dim)
            self.diffuser = FluxPipeline(
                FluxCompressor(d_model=vae_dim),
                FluxFlowProcessor(
                    d_model=feature_maps_dim, vae_dim=vae_dim, n_head=flow_attn_heads
                ),
                FluxExpander(d_model=vae_dim),
            )
            self.pipeline = self.diffuser  # For consistency

            # Load checkpoint
            state_dict = safetensors.torch.load_file(checkpoint_path)
            self.diffuser.load_state_dict(
                {
                    k.replace("diffuser.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("diffuser.")
                },
                strict=False,
            )
            self.text_encoder.load_with_override(checkpoint_path, override_path=text_encoder_path)

            self.diffuser.to(self.device).eval()
            self.text_encoder.to(self.device).eval()

            self.model_checkpoint = checkpoint_path
            self.config = {
                "vae_dim": vae_dim,
                "feature_maps_dim": feature_maps_dim,
                "text_embedding_dim": text_embedding_dim,
                "version": "0.3.0",
                "model_info": "Legacy v0.3.0 model (assumed)",
            }

            return True, f"Model loaded successfully on {self.device} (Legacy v0.3.0)"

        except Exception as e:
            return False, f"Failed to load legacy model: {str(e)}"

    def _load_v070_fallback(
        self,
        checkpoint_path: str,
        vae_dim: int,
        feature_maps_dim: int,
        text_embedding_dim: int,
        text_encoder_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load model assuming v0.7.0 architecture (context-enhanced)."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Import v0.7.0 components
            from fluxflow.models.v070.flow import FluxFlowProcessor
            from fluxflow.models.v070.vae import FluxCompressor, FluxExpander

            # Detect actual config from checkpoint instead of using provided dimensions
            state_dict = safetensors.torch.load_file(checkpoint_path)
            config = FluxPipeline._detect_config(state_dict)

            # For v0.7.0 legacy checkpoints, the detected vae_dim includes CONTEXT_DIMS
            # Adjust to get the actual VAE latent dimension
            if config.get("model_version") == "0.7.0":
                from fluxflow.models.v070.vae import CONTEXT_DIMS

                vae_latent_dim = config["vae_dim"] - CONTEXT_DIMS  # VAE latent dimension
            else:
                vae_latent_dim = config["vae_dim"]  # VAE components use this

            # Calculate appropriate attention heads
            def get_valid_n_head(d_model, preferred_heads=8):
                """Get number of heads that evenly divides d_model."""
                if d_model % preferred_heads == 0:
                    return preferred_heads
                # Find largest divisor that keeps heads reasonable
                for heads in range(preferred_heads, 0, -1):
                    if d_model % heads == 0:
                        return heads
                return 1  # Fallback

            vae_attn_heads = get_valid_n_head(vae_latent_dim)
            flow_attn_heads = get_valid_n_head(config["flow_dim"])

            # Initialize models with detected config
            self.text_encoder = BertTextEncoder(
                embed_dim=config.get("text_embed_dim", text_embedding_dim)
            )
            self.diffuser = FluxPipeline(
                FluxCompressor(d_model=vae_latent_dim, attn_heads=vae_attn_heads),
                FluxFlowProcessor(
                    d_model=config["flow_dim"], vae_dim=vae_latent_dim, n_head=flow_attn_heads
                ),
                FluxExpander(d_model=vae_latent_dim),
            )
            self.pipeline = self.diffuser  # For consistency

            # Load checkpoint
            self.diffuser.load_state_dict(
                {
                    k.replace("diffuser.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("diffuser.")
                },
                strict=False,
            )
            self.text_encoder.load_with_override(checkpoint_path, override_path=text_encoder_path)

            self.diffuser.to(self.device).eval()
            self.text_encoder.to(self.device).eval()

            self.model_checkpoint = checkpoint_path
            self.config = {
                "vae_dim": vae_latent_dim,  # actual VAE latent dim
                "feature_maps_dim": config["flow_dim"],
                "text_embedding_dim": config.get("text_embed_dim", text_embedding_dim),
                "version": "0.7.0",
                "model_info": "v0.7.0 auto-detected",
            }

            return True, f"Model loaded successfully on {self.device} (v0.7.0 auto-detected)"

        except Exception as e:
            raise Exception(f"Failed to load v0.7.0 fallback model: {str(e)}")

    def _load_v100_fallback(
        self,
        checkpoint_path: str,
        vae_dim: int,
        feature_maps_dim: int,
        text_embedding_dim: int,
        text_encoder_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load model assuming v0.10.0 architecture (2D RoPE + dual FiLM, bezier-coupled)."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Import v0.10.0 components
            from fluxflow.models.v100.flow import FluxFlowProcessor_v100
            from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

            # Detect actual config from checkpoint instead of using provided dimensions.
            # Unlike v0.7.0, v0.10.0's detected vae_dim is already the true VAE
            # latent dim (no CONTEXT_DIMS offset to subtract).
            state_dict = safetensors.torch.load_file(checkpoint_path)
            config = FluxPipeline._detect_config(state_dict)
            vae_latent_dim = config["vae_dim"]

            # Calculate appropriate attention heads
            def get_valid_n_head(d_model, preferred_heads=8):
                """Get number of heads that evenly divides d_model."""
                if d_model % preferred_heads == 0:
                    return preferred_heads
                for heads in range(preferred_heads, 0, -1):
                    if d_model % heads == 0:
                        return heads
                return 1  # Fallback

            flow_attn_heads = get_valid_n_head(config["flow_dim"], config.get("flow_attn_heads", 8))

            # Initialize models with detected config
            self.text_encoder = BertTextEncoder(
                embed_dim=config.get("text_embed_dim", text_embedding_dim)
            )
            self.diffuser = FluxPipeline(
                FluxCompressor_v100(d_model=vae_latent_dim, downscales=config["downscales"]),
                FluxFlowProcessor_v100(
                    d_model=config["flow_dim"],
                    vae_dim=vae_latent_dim,
                    embedding_size=config.get("text_embed_dim", text_embedding_dim),
                    n_head=flow_attn_heads,
                    n_layers=config.get("flow_transformer_layers", 10),
                ),
                FluxExpander_v100(
                    d_model=vae_latent_dim, upscales=config.get("upscales", config["downscales"])
                ),
            )
            self.pipeline = self.diffuser  # For consistency

            # Load checkpoint
            self.diffuser.load_state_dict(
                {
                    k.replace("diffuser.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("diffuser.")
                },
                strict=False,
            )
            self.text_encoder.load_with_override(checkpoint_path, override_path=text_encoder_path)

            self.diffuser.to(self.device).eval()
            self.text_encoder.to(self.device).eval()

            self.model_checkpoint = checkpoint_path
            self.config = {
                "vae_dim": vae_latent_dim,
                "feature_maps_dim": config["flow_dim"],
                "text_embedding_dim": config.get("text_embed_dim", text_embedding_dim),
                "version": "0.10.0",
                "model_info": "v0.10.0 auto-detected",
            }

            return True, f"Model loaded successfully on {self.device} (v0.10.0 auto-detected)"

        except Exception as e:
            raise Exception(f"Failed to load v0.10.0 fallback model: {str(e)}")

    def generate_image(
        self,
        prompt: str,
        img_width: int = 512,
        img_height: int = 512,
        ddim_steps: int = 50,
        seed: Optional[int] = None,
        use_cfg: bool = False,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Generate image from text prompt.

        Args:
            prompt: Text prompt
            img_width: Image width (must be multiple of 16)
            img_height: Image height (must be multiple of 16)
            ddim_steps: Number of diffusion steps
            seed: Random seed (optional)
            use_cfg: Enable classifier-free guidance
            guidance_scale: CFG strength (only used if use_cfg=True)
            negative_prompt: Negative prompt for CFG (optional)

        Returns:
            Tuple of (image array, status message)
        """
        if not self.diffuser or not self.text_encoder or not self.tokenizer:
            return None, "Model not loaded. Please load a checkpoint first."

        try:
            # MPS-specific preparation
            if self.device.type == "mps":
                import torch

                torch.mps.empty_cache()

            # Validate dimensions are multiples of 16
            if img_width % 16 != 0 or img_height % 16 != 0:
                return (
                    None,
                    f"Width and height must be multiples of 16 (got {img_width}x{img_height})",
                )

            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            with torch.no_grad():
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Encode text — v0.10.0 BertTextEncoder returns per-token (seq, mask).
                text_seq, text_mask = self.text_encoder(input_ids, attention_mask=attention_mask)

                # Encode negative / null prompt if CFG is enabled.
                # NOTE: a zeros-like null with an all-False mask would NaN-out the
                # cross-attention softmax in the v0.10.0 flow. Use an empty-prompt
                # encoding (via build_cfg_null_pair) when no negative prompt is given.
                negative_seq = None
                negative_mask = None
                if use_cfg and guidance_scale > 1.0:
                    if negative_prompt:
                        neg_inputs = self.tokenizer(
                            negative_prompt,
                            padding="max_length",
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        neg_input_ids = neg_inputs["input_ids"].to(self.device)
                        neg_attention_mask = neg_inputs["attention_mask"].to(self.device)
                        negative_seq, negative_mask = self.text_encoder(
                            neg_input_ids, attention_mask=neg_attention_mask
                        )
                    else:
                        null_seq, null_mask = build_cfg_null_pair(
                            self.text_encoder, max_length=int(text_seq.size(1))
                        )
                        negative_seq = null_seq.to(device=self.device, dtype=text_seq.dtype)
                        negative_mask = null_mask.to(device=self.device)

                # Create pure Gaussian noise latent (all dims including context)
                context_dims = self.diffuser.compressor.get_context_dims()
                dummy = torch.zeros(1, 3, img_height, img_width, device=self.device)
                noised_latent = img_to_random_packet(
                    dummy,
                    d_model=self.diffuser.compressor.d_model,
                    context_dims=context_dims,
                    downscales=getattr(self.diffuser.compressor, "downscales", 4),
                    max_hw=getattr(self.diffuser.compressor, "max_hw", 1024),
                ).to(dtype=text_seq.dtype)

                # Denoise with or without CFG
                if (
                    use_cfg
                    and guidance_scale > 1.0
                    and negative_seq is not None
                    and negative_mask is not None
                ):
                    # Use CFG-guided generation
                    denoised_latent = self._generate_with_cfg(
                        noised_latent=noised_latent,
                        text_seq=text_seq,
                        text_mask=text_mask,
                        negative_seq=negative_seq,
                        negative_mask=negative_mask,
                        guidance_scale=guidance_scale,
                        steps=ddim_steps,
                    )
                else:
                    # Standard generation
                    denoised_latent = generate_latent_images(
                        batch_z=noised_latent,
                        text_seq=text_seq,
                        text_mask=text_mask,
                        diffuser=self.diffuser,
                        steps=ddim_steps,
                        prediction_type="v_prediction",
                    )

                # Decode
                decoded_image = self.diffuser.expander(denoised_latent)

                # Convert to numpy array for Gradio
                image = decoded_image[0].cpu()
                image = (image + 1) / 2  # [-1, 1] -> [0, 1]
                image = image.clamp(0, 1)
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)

                return image, "Image generated successfully!"

        except Exception as e:
            return None, f"Generation failed: {str(e)}"

    def _generate_with_cfg(
        self,
        noised_latent: torch.Tensor,
        text_seq: torch.Tensor,
        text_mask: torch.Tensor,
        negative_seq: torch.Tensor,
        negative_mask: torch.Tensor,
        guidance_scale: float,
        steps: int,
    ) -> torch.Tensor:
        """Generate image with classifier-free guidance.

        Args:
            noised_latent: Initial noised latent [B, T+1, D]
            text_seq: Conditional per-token text embeddings [B, T_txt, E]
            text_mask: Conditional bool mask [B, T_txt]
            negative_seq: Unconditional/negative per-token text embeddings [B, T_txt, E]
            negative_mask: Unconditional/negative bool mask [B, T_txt]
            guidance_scale: CFG strength
            steps: Number of denoising steps

        Returns:
            Denoised latent [B, T+1, D]
        """
        from diffusers import DPMSolverMultistepScheduler

        assert self.diffuser is not None, "Diffuser must be loaded before CFG generation"

        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="dpmsolver++",
            solver_order=2,
            prediction_type="v_prediction",
            lower_order_final=True,
            timestep_spacing="trailing",
        )
        scheduler.set_timesteps(steps, device=self.device)  # type: ignore

        # Separate hw_vec (dimension info) from image latent
        hw_vec = noised_latent[:, -1:, :].clone()
        lat = noised_latent[:, :-1, :].clone()

        for t in scheduler.timesteps:  # type: ignore
            # Expand t to batch dimension
            t_batch = torch.full(
                (lat.size(0),), t.item() / 999.0, device=self.device, dtype=torch.float32
            )

            # Reconstruct full latent with hw_vec for model input
            full_input = torch.cat([lat, hw_vec], dim=1)

            # Predict with conditional per-token text
            v_cond = self.diffuser.flow_processor(full_input, text_seq, text_mask, t_batch)
            v_cond_lat = v_cond[:, :-1, :]  # Remove hw_vec from prediction

            # Predict with unconditional per-token text
            v_uncond = self.diffuser.flow_processor(
                full_input, negative_seq, negative_mask, t_batch
            )
            v_uncond_lat = v_uncond[:, :-1, :]  # Remove hw_vec from prediction

            # Apply CFG guidance
            v_guided = v_uncond_lat + guidance_scale * (v_cond_lat - v_uncond_lat)

            # Step the scheduler (only on image latent, not hw_vec)
            lat = scheduler.step(  # type: ignore[attr-defined]
                model_output=v_guided, timestep=int(t.item()), sample=lat
            ).prev_sample

        # Recombine with hw_vec before returning
        return torch.cat([lat, hw_vec], dim=1)

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded
        """
        return self.diffuser is not None and self.text_encoder is not None

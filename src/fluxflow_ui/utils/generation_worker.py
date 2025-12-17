"""Generation worker for UI."""

import sys
from pathlib import Path
from typing import Optional, Tuple

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
)
from fluxflow.utils import generate_latent_images  # noqa: E402


class GenerationWorker:
    """Worker for generating images from text prompts."""

    def __init__(self):
        """Initialize generation worker."""
        self.model_checkpoint: Optional[str] = None
        self.diffuser: Optional[FluxPipeline] = None
        self.text_encoder: Optional[BertTextEncoder] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = self._get_device()
        self.config = {}

    def _get_device(self) -> torch.device:
        """Get available device.

        Returns:
            torch device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load_model(
        self,
        checkpoint_path: str,
        vae_dim: int = 64,
        feature_maps_dim: int = 64,
        text_embedding_dim: int = 1024,
    ) -> Tuple[bool, str]:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            vae_dim: VAE latent dimension
            feature_maps_dim: Flow processor dimension
            text_embedding_dim: Text embedding dimension

        Returns:
            Tuple of (success, message)
        """
        try:
            if not Path(checkpoint_path).exists():
                return False, f"Checkpoint not found: {checkpoint_path}"

            # Load tokenizer (uses cache if present, otherwise downloads)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
            )
            if self.tokenizer.pad_token is None:  # type: ignore[union-attr]
                self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[union-attr]
                self.tokenizer.add_special_tokens(  # type: ignore[union-attr]
                    {"pad_token": "[PAD]"}
                )

            # Initialize models
            self.text_encoder = BertTextEncoder(embed_dim=text_embedding_dim)
            self.diffuser = FluxPipeline(
                FluxCompressor(d_model=vae_dim),
                FluxFlowProcessor(d_model=feature_maps_dim, vae_dim=vae_dim),
                FluxExpander(d_model=vae_dim),
            )

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
            self.text_encoder.load_state_dict(
                {
                    k.replace("text_encoder.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("text_encoder.")
                },
                strict=False,
            )

            self.diffuser.to(self.device).eval()
            self.text_encoder.to(self.device).eval()

            self.model_checkpoint = checkpoint_path
            self.config = {
                "vae_dim": vae_dim,
                "feature_maps_dim": feature_maps_dim,
                "text_embedding_dim": text_embedding_dim,
            }

            return True, f"Model loaded successfully on {self.device}"

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

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
                inputs = self.tokenizer(  # type: ignore[operator]
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Encode text
                text_embeddings = self.text_encoder(input_ids, attention_mask=attention_mask)

                # Encode negative prompt if CFG is enabled
                negative_embeddings = None
                if use_cfg and guidance_scale > 1.0:
                    if negative_prompt:
                        neg_inputs = self.tokenizer(  # type: ignore[operator]
                            negative_prompt,
                            padding="max_length",
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        neg_input_ids = neg_inputs["input_ids"].to(self.device)
                        neg_attention_mask = neg_inputs["attention_mask"].to(self.device)
                        negative_embeddings = self.text_encoder(
                            neg_input_ids, attention_mask=neg_attention_mask
                        )
                    else:
                        # Use null conditioning (zeros)
                        negative_embeddings = torch.zeros_like(text_embeddings)

                # Create random latent with specified dimensions
                z_img = (torch.rand((1, 3, img_height, img_width), device=self.device) * 2) - 1
                latent_z = self.diffuser.compressor(z_img)

                img_seq = latent_z[:, :-1, :].contiguous()
                hw_vec = latent_z[:, -1:, :].contiguous()
                noise_img = torch.randn_like(img_seq)

                # Create noised latent
                from diffusers import DPMSolverMultistepScheduler

                scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
                scheduler.set_timesteps(  # type: ignore[attr-defined]
                    ddim_steps, device=self.device
                )

                t = torch.randint(0, 1000, (1,), device=self.device)
                noised_img = scheduler.add_noise(img_seq, noise_img, t)  # type: ignore
                noised_latent = torch.cat([noised_img, hw_vec], dim=1)

                # Denoise with or without CFG
                if use_cfg and guidance_scale > 1.0 and negative_embeddings is not None:
                    # Use CFG-guided generation
                    denoised_latent = self._generate_with_cfg(
                        noised_latent=noised_latent,
                        text_embeddings=text_embeddings,
                        negative_embeddings=negative_embeddings,
                        guidance_scale=guidance_scale,
                        steps=ddim_steps,
                    )
                else:
                    # Standard generation
                    denoised_latent = generate_latent_images(
                        batch_z=noised_latent,
                        text_embeddings=text_embeddings,
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
        text_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        guidance_scale: float,
        steps: int,
    ) -> torch.Tensor:
        """Generate image with classifier-free guidance.

        Args:
            noised_latent: Initial noised latent [B, T+1, D]
            text_embeddings: Conditional text embeddings
            negative_embeddings: Unconditional/negative text embeddings
            guidance_scale: CFG strength
            steps: Number of denoising steps

        Returns:
            Denoised latent [B, T+1, D]
        """
        from diffusers import DPMSolverMultistepScheduler

        assert self.diffuser is not None, "Diffuser must be loaded before CFG generation"

        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
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
            t_batch = torch.full((lat.size(0),), t.item(), device=self.device, dtype=torch.long)

            # Reconstruct full latent with hw_vec for model input
            full_input = torch.cat([lat, hw_vec], dim=1)

            # Predict with conditional embeddings
            v_cond = self.diffuser.flow_processor(full_input, text_embeddings, t_batch)
            v_cond_lat = v_cond[:, :-1, :]  # Remove hw_vec from prediction

            # Predict with unconditional embeddings
            v_uncond = self.diffuser.flow_processor(full_input, negative_embeddings, t_batch)
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

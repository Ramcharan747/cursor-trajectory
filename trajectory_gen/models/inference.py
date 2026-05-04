"""
End-to-end inference pipeline.

Given start and end points, generates a complete naturalistic trajectory
by sequencing motion primitives through the trained Latent ODE.

Pipeline: (start, end) → Latent ODE → primitive sequence → SIREN decode → (x,y,t)
"""

import torch
import numpy as np
from typing import Optional, Tuple
from .siren import SIREN
from .vqvae import VQVAE
from .latent_ode import LatentODE


class TrajectoryGenerator:
    """
    End-to-end trajectory generation from trained models.
    
    Usage:
        gen = TrajectoryGenerator.load("checkpoint.pt")
        trajectory = gen.generate(start=(100, 200), end=(800, 600))
    """
    
    def __init__(
        self,
        vqvae: VQVAE,
        latent_ode: LatentODE,
        siren_config: dict,
        device: torch.device,
    ):
        self.vqvae = vqvae.eval().to(device)
        self.latent_ode = latent_ode.eval().to(device)
        self.siren_config = siren_config
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[torch.device] = None):
        """Load all models from a single checkpoint."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config = checkpoint['config']
        
        vqvae = VQVAE(
            input_dim=config['siren_param_count'],
            embedding_dim=config.get('embedding_dim', 64),
            num_embeddings=config.get('num_embeddings', 128),
        )
        vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
        
        latent_ode = LatentODE(
            input_dim=config.get('embedding_dim', 64),
            latent_dim=config.get('latent_dim', 16),
        )
        latent_ode.load_state_dict(checkpoint['latent_ode_state_dict'])
        
        return cls(vqvae, latent_ode, config, device)

    @torch.no_grad()
    def generate(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_primitives: int = 5,
        points_per_primitive: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a naturalistic trajectory between two points.
        
        Args:
            start: (x, y) start position in pixels
            end: (x, y) end position in pixels
            num_primitives: Number of motion primitives to sequence
            points_per_primitive: Points to sample from each SIREN
        
        Returns:
            Tuple of (t, x, y) numpy arrays for the full trajectory
        """
        # Sample primitive sequence from Latent ODE
        primitive_embeddings = self.latent_ode.sample(
            num_samples=1,
            seq_len=num_primitives,
            device=self.device,
        )  # (1, num_primitives, embedding_dim)
        
        # Decode each embedding to SIREN weights via VQ-VAE decoder
        embeddings = primitive_embeddings.squeeze(0)  # (num_primitives, embedding_dim)
        
        # Quantize to nearest codebook entry for cleaner primitives
        _, _, indices, _ = self.vqvae.vq(embeddings)
        siren_weights_batch = self.vqvae.decode_indices(indices)
        
        # Build trajectory by evaluating each SIREN
        all_t, all_x, all_y = [], [], []
        
        for i in range(num_primitives):
            # Reconstruct SIREN from weight vector
            siren = SIREN(
                in_features=1,
                hidden_features=self.siren_config.get('hidden_features', 64),
                hidden_layers=self.siren_config.get('hidden_layers', 3),
                out_features=2,
            ).to(self.device)
            siren.set_weight_vector(siren_weights_batch[i])
            
            t_seg, x_seg, y_seg = siren.compute_trajectory(
                num_points=points_per_primitive
            )
            
            # Offset time for continuity
            t_offset = i * 1.0  # 1 second per primitive
            all_t.append(t_seg + t_offset)
            all_x.append(x_seg)
            all_y.append(y_seg)
        
        t = np.concatenate(all_t)
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        
        # Affine transform to match start/end constraints
        x, y = self._apply_boundary_conditions(x, y, start, end)
        
        return t, x, y

    def _apply_boundary_conditions(
        self,
        x: np.ndarray,
        y: np.ndarray,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp trajectory to exactly hit start and end points."""
        # Linear interpolation factor along trajectory
        alpha = np.linspace(0, 1, len(x))
        
        # Baseline straight line
        base_x = start[0] + alpha * (end[0] - start[0])
        base_y = start[1] + alpha * (end[1] - start[1])
        
        # Scale perturbation relative to distance
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        scale = max(dist * 0.3, 10.0)  # Perturbation magnitude
        
        # Add SIREN-generated perturbation that fades at endpoints
        fade = np.sin(np.pi * alpha)  # 0 at start/end, 1 in middle
        x_out = base_x + x * scale * fade
        y_out = base_y + y * scale * fade
        
        return x_out, y_out

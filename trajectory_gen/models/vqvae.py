"""
VQ-VAE — Vector Quantised Variational AutoEncoder

Paper: "Neural Discrete Representation Learning"
       van den Oord, Vinyals, Kavukcuoglu (2017)
       DeepMind — arXiv:1711.00937

Purpose in our pipeline:
    Compress SIREN weight vectors (~8,642 dims) into a discrete codebook
    of ~128 motion primitives. Each primitive is a reusable building block
    of cursor movement (arcs, lines, hooks, micro-corrections, etc.).

Key equations from paper:

    Quantization (Eq. 1-2):
        q(z = k|x) = 1  for k = argmin_j ||z_e(x) - e_j||²
        z_q(x) = e_k     where k = argmin_j ||z_e(x) - e_j||²

    Loss function (Eq. 3):
        L = log p(x|z_q(x)) + ||sg[z_e(x)] - e||² + β·||z_e(x) - sg[e]||²
        
        Term 1: Reconstruction loss (trains encoder + decoder)
        Term 2: VQ loss / dictionary learning (trains codebook embeddings)
        Term 3: Commitment loss (prevents encoder output from growing)
        
        sg[·] = stop-gradient operator
        β = 0.25 (paper: "results did not vary for β from 0.1 to 2.0")

    EMA codebook updates (Appendix A.1, Eq. 6-8):
        N_i^(t) = N_i^(t-1) · γ + n_i^(t) · (1-γ)
        m_i^(t) = m_i^(t-1) · γ + Σ z_{i,j}^(t) · (1-γ)
        e_i^(t) = m_i^(t) / N_i^(t)
        
        γ = 0.99 (paper: "found γ=0.99 to work well in practice")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with EMA codebook updates.
    
    From paper Section 3.1-3.2:
    The encoder output z_e(x) is mapped to its nearest codebook vector e_k.
    Gradients pass through via straight-through estimator (Bengio et al., 2013):
    the gradient ∇_z L is copied directly from decoder input to encoder output.
    
    EMA updates (Appendix A.1) replace the VQ loss term with exponential
    moving averages, which is equivalent to online K-means and is more stable.
    
    Args:
        num_embeddings: K — size of discrete latent space (codebook entries)
        embedding_dim: D — dimensionality of each embedding vector
        commitment_cost: β — weight for commitment loss (default: 0.25)
        decay: γ — EMA decay rate (default: 0.99)
        epsilon: Small value to prevent division by zero in EMA
    """

    def __init__(
        self,
        num_embeddings: int = 128,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings  # K
        self.embedding_dim = embedding_dim    # D
        self.commitment_cost = commitment_cost  # β
        self.decay = decay  # γ
        self.epsilon = epsilon

        # Codebook: K embedding vectors of dimension D
        # e ∈ R^{K×D} (paper Section 3.1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # EMA variables (Appendix A.1)
        # N_i: count of encoder outputs assigned to each embedding
        # m_i: sum of encoder outputs assigned to each embedding
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", self.embedding.weight.data.clone()
        )

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize encoder output to nearest codebook vector.
        
        From paper Eq. 1-2:
            k = argmin_j ||z_e(x) - e_j||²
            z_q(x) = e_k
        
        Straight-through estimator:
            Forward: z_q = e_k (quantized)
            Backward: ∇_z L passes through unaltered
        
        Args:
            z_e: Encoder output, shape (batch, embedding_dim)
            
        Returns:
            Tuple of:
                - z_q: Quantized vectors (batch, embedding_dim)
                - loss: VQ loss + commitment loss (scalar)
                - encoding_indices: Codebook indices (batch,)
                - perplexity: Codebook utilization metric (scalar)
        """
        # Compute distances: ||z_e - e_j||² = ||z_e||² + ||e_j||² - 2·z_e·e_j^T
        # (batch, K)
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(z_e, self.embedding.weight.t())
        )

        # Nearest neighbor lookup (Eq. 1)
        # k = argmin_j ||z_e(x) - e_j||²
        encoding_indices = torch.argmin(distances, dim=1)  # (batch,)

        # Get quantized vectors (Eq. 2)
        # z_q(x) = e_k
        z_q = self.embedding(encoding_indices)  # (batch, D)

        # EMA codebook update (Appendix A.1, Eq. 6-8)
        if self.training:
            # One-hot encodings for assignments
            encodings = F.one_hot(
                encoding_indices, self.num_embeddings
            ).float()  # (batch, K)

            # N_i^(t) = N_i^(t-1) · γ + n_i^(t) · (1-γ)  [Eq. 6]
            self._ema_cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # m_i^(t) = m_i^(t-1) · γ + Σ z_{i,j}^(t) · (1-γ)  [Eq. 7]
            dw = torch.matmul(encodings.t(), z_e)  # (K, D)
            self._ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Laplace smoothing to prevent empty clusters
            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            # e_i^(t) = m_i^(t) / N_i^(t)  [Eq. 8]
            self.embedding.weight.data.copy_(
                self._ema_w / cluster_size.unsqueeze(1)
            )

        # Loss (Eq. 3 — only commitment loss since EMA handles VQ loss)
        # β · ||z_e(x) - sg[e]||²
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator:
        # Forward: z_q (quantized), Backward: gradients flow to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # Codebook perplexity — measures how uniformly the codebook is used
        # Higher perplexity = better utilization, max = num_embeddings
        avg_probs = torch.mean(
            F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, commitment_loss, encoding_indices, perplexity


class VQVAE(nn.Module):
    """
    VQ-VAE for learning a discrete codebook of motion primitives.
    
    From paper Section 3:
    - Encoder: maps SIREN weight vectors → continuous latent space
    - Vector Quantizer: discretizes to nearest codebook entry
    - Decoder: reconstructs SIREN weight vectors from codebook entry
    
    The encoder and decoder are simple MLPs since our "data" is
    already a 1D weight vector (not images/audio).
    
    Architecture:
        Encoder: 8642 → 512 → ReLU → 256 → ReLU → 64 (embedding_dim)
        Codebook: 128 entries × 64 dims (with EMA updates, γ=0.99)
        Decoder: 64 → 256 → ReLU → 512 → ReLU → 8642
    
    The codebook size K=128 gives us 128 distinct motion primitives —
    the "vocabulary" of cursor movement.
    
    Args:
        input_dim: Dimensionality of SIREN weight vectors
        hidden_dim: Hidden layer width in encoder/decoder
        embedding_dim: Dimensionality of codebook vectors (D)
        num_embeddings: Number of codebook entries (K)
        commitment_cost: β for commitment loss
        ema_decay: γ for EMA updates
    """

    def __init__(
        self,
        input_dim: int = 12738,
        hidden_dim: int = 512,
        embedding_dim: int = 64,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Encoder: SIREN weights → continuous embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

        # Vector Quantizer with EMA updates
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
        )

        # Decoder: quantized embedding → reconstructed SIREN weights
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SIREN weight vector to continuous embedding."""
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized embedding back to SIREN weight space."""
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → quantize → decode.
        
        Loss decomposition (paper Eq. 3):
            L = reconstruction_loss + commitment_loss
            
            reconstruction_loss = ||x - x̂||²  (MSE on weight vectors)
            commitment_loss = β · ||z_e - sg[e]||²
            
            The VQ/dictionary loss is handled by EMA updates, not gradient descent.
        
        Args:
            x: SIREN weight vectors, shape (batch, input_dim)
            
        Returns:
            Tuple of:
                - x_recon: Reconstructed weight vectors (batch, input_dim)
                - vq_loss: Commitment loss (scalar)
                - encoding_indices: Codebook indices used (batch,)
                - perplexity: Codebook utilization (scalar)
        """
        # Encode
        z_e = self.encoder(x)

        # Quantize (straight-through)
        z_q, vq_loss, indices, perplexity = self.vq(z_e)

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, indices, perplexity

    def get_codebook(self) -> torch.Tensor:
        """
        Return the full codebook as a tensor.
        Shape: (num_embeddings, embedding_dim) = (128, 64)
        """
        return self.vq.embedding.weight.data.clone()

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices directly to SIREN weight vectors.
        Used during inference: index → embedding → decoder → weights.
        
        Args:
            indices: Codebook indices, shape (batch,)
            
        Returns:
            Reconstructed weight vectors, shape (batch, input_dim)
        """
        z_q = self.vq.embedding(indices)
        return self.decoder(z_q)

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        vq_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total VQ-VAE loss.
        
        From paper Eq. 3:
            L = ||x - x̂||² + β·||z_e - sg[e]||²
        
        The first term trains encoder + decoder (via straight-through).
        The second term (commitment loss) is returned by VectorQuantizer.
        The codebook is updated via EMA (not through this loss).
        """
        recon_loss = F.mse_loss(x_recon, x)
        return recon_loss + vq_loss

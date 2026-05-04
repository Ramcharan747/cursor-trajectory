"""
Latent ODE for trajectory sequence generation.

Papers:
    - "Neural Ordinary Differential Equations" (Chen et al., 2018) — arXiv:1806.07366
    - "Latent ODEs for Irregularly-Sampled Time Series" (Rubanova et al., 2019) — arXiv:1907.03907

Architecture (from Rubanova et al., Algorithm 2 in supplement):
    1. ODE-RNN encoder runs backwards in time on observed primitives
    2. Final hidden state → μ_z0, σ_z0 (approximate posterior)
    3. Sample z0 ~ N(μ_z0, σ_z0)
    4. ODESolve(f, z0, t0...tN) → latent trajectory
    5. Decode each z_i to primitive index or trajectory point

ODE function uses Tanh activations per Rubanova et al. supplement Section 4:
    "Tanh activation constrains the output and prevents the ODE gradients from
    taking large values... we do not recommend using ReLU."

Solver: dopri5 (adaptive Runge-Kutta 4/5) with rtol=1e-3, atol=1e-4
Training: ELBO = E[log p(x|z)] - KL(q(z0|x) || p(z0))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class ODEFunc(nn.Module):
    """
    Neural network parameterizing dz/dt = f_θ(z(t), t).
    
    From Chen et al. 2018, Eq. 2: dh(t)/dt = f(h(t), t, θ)
    Uses time-invariant dynamics: f(z) not f(z,t), following
    Rubanova et al.: "used time-invariant dynamics dh(t)/dt = f_θ(h(t))"
    
    Tanh activations per supplement Section 4 recommendation.
    Scaled: 4 layers × 512 hidden for VRAM target.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        # Final Tanh to bound gradients (supplement recommendation)
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.nfe = 0  # Number of function evaluations (for monitoring)

    def forward(self, t, z):
        self.nfe += 1
        return self.net(z)


class GRUODECell(nn.Module):
    """
    GRU cell for ODE-RNN hidden state updates at observation times.
    
    From Rubanova et al. Algorithm 1:
        h'_i = ODESolve(f_θ, h_{i-1}, (t_{i-1}, t_i))  # ODE between obs
        h_i = GRUCell(h'_i, x_i)                         # Update at obs
    
    Standard GRU equations (supplement Algorithm 1):
        z = σ(f_z([h_prev; x]))       # Update gate
        r = σ(f_r([h_prev; x]))       # Reset gate  
        h' = tanh([r * h_prev; x])    # Candidate
        h = (1-z) * h' + z * h_prev   # New state
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.gru_cell(x, h)


class ODERNNEncoder(nn.Module):
    """
    ODE-RNN recognition network (encoder).
    
    From Rubanova et al. Section 3.1 + Eq. 8:
        q(z0|{x_i, t_i}) = N(μ_z0, σ_z0)
        where μ_z0, σ_z0 = g(ODE-RNN_φ({x_i, t_i}))
    
    Runs backwards in time from t_N to t_0 to get approximate posterior.
    Between observations, hidden state evolves via ODE.
    At each observation, hidden state is updated via GRU.
    
    Hyperparameter choices from supplement Section 5:
        - Recognition model dimensionality > generative model
        - MuJoCo (14D data): 30D recognition, 15D generative
        - We use similar ratio: hidden_dim for recognition > latent_dim
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        ode_hidden_dim: int = 64,
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # ODE dynamics for hidden state between observations
        self.ode_func = ODEFunc(hidden_dim, ode_hidden_dim, num_layers=2)
        
        # GRU for updating hidden state at observations
        self.gru_cell = GRUODECell(input_dim, hidden_dim)
        
        # Map final hidden state to posterior parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.use_adjoint = use_adjoint
        self._odeint = odeint_adjoint if (use_adjoint and HAS_TORCHDIFFEQ) else (odeint if HAS_TORCHDIFFEQ else None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a sequence backwards in time.
        
        Args:
            x: Observations (batch, seq_len, input_dim)
            t: Time points (batch, seq_len) or (seq_len,)
        
        Returns:
            mu, logvar of approximate posterior q(z0|x)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        if self._odeint is not None:
            # Run backwards: from t_N to t_0
            for i in reversed(range(seq_len)):
                # ODE step between observations (if not last)
                if i < seq_len - 1:
                    t_span = torch.tensor([0.0, (t[..., i+1] - t[..., i]).mean().item()], device=device)
                    h = self._odeint(self.ode_func, h, t_span, rtol=1e-3, atol=1e-4)[-1]
                
                # GRU update at observation
                h = self.gru_cell(x[:, i, :], h)
        else:
            # Fallback: simple GRU without ODE dynamics
            for i in reversed(range(seq_len)):
                h = self.gru_cell(x[:, i, :], h)
        
        # Posterior parameters (Eq. 8)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class LatentODE(nn.Module):
    """
    Full Latent ODE model for trajectory primitive sequence generation.
    
    Generative model (Rubanova et al. Eq. 5-7 / Chen et al. Eq. 11-13):
        z0 ~ p(z0) = N(0, I)
        z0, z1, ..., zN = ODESolve(f_θ, z0, (t0, ..., tN))
        x_i ~ p(x_i | z_i)  (independent)
    
    Recognition model (Eq. 8):
        q(z0|{x_i, t_i}) = N(μ_z0, σ_z0) via ODE-RNN encoder
    
    Training objective — ELBO (Eq. 9):
        ELBO = E_{z0~q}[log p(x0,...,xN)] - KL(q(z0|x) || p(z0))
    
    Loss details (supplement Section "Loss"):
        - Reconstruction: negative Gaussian log-likelihood, variance=0.01
        - KL annealing coefficient: 0.99 (supplement Section 6)
        - Optimizer: Adamax, lr=0.01, decay=0.999
    
    Args:
        input_dim: Observation dimensionality (primitive embedding or coords)
        latent_dim: ODE latent state dimensionality
        rec_hidden_dim: Recognition network hidden dim (> latent_dim)
        gen_hidden_dim: Generative ODE function hidden dim
        output_dim: Output dimensionality (same as input_dim usually)
        num_embeddings: VQ-VAE codebook size for primitive selection
        use_adjoint: Use adjoint method for O(1) memory (Chen et al. Section 2)
    """
    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 64,
        rec_hidden_dim: int = 256,
        gen_hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        output_dim = output_dim or input_dim
        
        # Recognition network (ODE-RNN encoder)
        self.encoder = ODERNNEncoder(
            input_dim=input_dim,
            hidden_dim=rec_hidden_dim,
            latent_dim=latent_dim,
            use_adjoint=use_adjoint,
        )
        
        # Generative ODE dynamics: dz/dt = f_θ(z)
        self.ode_func = ODEFunc(latent_dim, gen_hidden_dim, num_layers=3)
        
        # Decoder: z_i → x̂_i
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, gen_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gen_hidden_dim, output_dim),
        )
        
        self.use_adjoint = use_adjoint
        self._odeint = None
        if HAS_TORCHDIFFEQ:
            self._odeint = odeint_adjoint if use_adjoint else odeint

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ·ε, ε ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_latent_trajectory(
        self, z0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve generative ODE forward in time from z0.
        
        From paper Eq. 6: z0,...,zN = ODESolve(f_θ, z0, (t0,...,tN))
        Uses dopri5 solver with rtol=1e-3, atol=1e-4 (supplement Section 4).
        """
        if self._odeint is not None:
            # t should be (seq_len,) — same for all batch elements
            z_traj = self._odeint(
                self.ode_func, z0, t,
                rtol=1e-3, atol=1e-4,
                method='dopri5',
            )  # (seq_len, batch, latent_dim)
            return z_traj.permute(1, 0, 2)  # (batch, seq_len, latent_dim)
        else:
            # Euler fallback when torchdiffeq not available
            batch_size = z0.shape[0]
            z_traj = [z0]
            z = z0
            for i in range(1, len(t)):
                dt = t[i] - t[i-1]
                dz = self.ode_func(t[i-1], z)
                z = z + dt * dz
                z_traj.append(z)
            return torch.stack(z_traj, dim=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample z0 → ODE solve → decode.
        
        Args:
            x: Observations (batch, seq_len, input_dim)
            t: Time points (seq_len,) normalized to [0, 1]
        
        Returns:
            x_recon: Reconstructed sequence (batch, seq_len, output_dim)
            mu: Posterior mean (batch, latent_dim)
            logvar: Posterior log-variance (batch, latent_dim)
        """
        # Encode (backwards in time) → posterior q(z0|x)
        mu, logvar = self.encoder(x, t)
        
        # Sample z0 via reparameterization
        z0 = self.reparameterize(mu, logvar)
        
        # Solve ODE forward → latent trajectory
        z_traj = self.decode_latent_trajectory(z0, t)  # (batch, seq_len, latent)
        
        # Decode each latent state to observation space
        batch, seq_len, _ = z_traj.shape
        z_flat = z_traj.reshape(-1, self.latent_dim)
        x_recon = self.decoder(z_flat).reshape(batch, seq_len, -1)
        
        return x_recon, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
        obs_variance: float = 0.01,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute ELBO loss (Eq. 9).
        
        ELBO = E[log p(x|z)] - KL(q(z0|x) || p(z0))
        
        Reconstruction: Gaussian log-likelihood with fixed variance
            (supplement: "fixed variance of 0.01 for other datasets")
        
        KL: Analytic KL between N(μ, σ²) and N(0, I)
            KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        
        KL annealing: kl_weight ramps from 0 to 1 during training
            (supplement Section 6: "KL annealing with coefficient 0.99")
        """
        # Reconstruction loss (negative Gaussian log-likelihood)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean') / obs_variance
        
        # KL divergence: KL(N(μ, σ²) || N(0, I))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': kl_weight,
        }
        return total_loss, metrics

    def sample(
        self, num_samples: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Sample trajectories from prior p(z0) = N(0, I).
        
        From paper Fig. 4b: "Trajectories sampled from the prior
        p(z0) ~ Normal(z0; 0, I) of the trained model."
        """
        z0 = torch.randn(num_samples, self.latent_dim, device=device)
        t = torch.linspace(0, 1, seq_len, device=device)
        z_traj = self.decode_latent_trajectory(z0, t)
        z_flat = z_traj.reshape(-1, self.latent_dim)
        x_samples = self.decoder(z_flat).reshape(num_samples, seq_len, -1)
        return x_samples

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""
SIREN — Sinusoidal Representation Network

Paper: "Implicit Neural Representations with Periodic Activation Functions"
       Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020)
       Stanford University — arXiv:2006.09661

Architecture (Eq. 4 from paper):
    Φ(x) = W_n(φ_{n-1} ∘ φ_{n-2} ∘ ... ∘ φ_0)(x) + b_n
    where φ_i(x_i) = sin(W_i · x_i + b_i)

Initialization Scheme (Theorem 1.8 from supplementary):
    - Hidden layers: W ~ U(-√(6/fan_in), √(6/fan_in))
      This ensures pre-activations are N(0,1) and post-activations
      are Arcsin(-1,1) distributed, regardless of network depth.

    - First layer:  W ~ U(-1/fan_in, 1/fan_in), then scaled by ω₀
      ω₀ = 30 spans multiple periods over [-1,1] input range.

    - All layers can optionally use ω₀ factorization (Section 1.5):
      W = Ŵ * ω₀, where Ŵ ~ U(-√(6/(ω₀² · fan_in)), √(6/(ω₀² · fan_in)))
      This preserves activation distributions but boosts gradients to Ŵ by ω₀.

Usage for cursor trajectories:
    Input:  t ∈ [0, 1]  (normalized time)
    Output: (x, y) ∈ [-1, 1]  (normalized screen coordinates)
    A single SIREN encodes one trajectory micro-segment (~0.2-2s).
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SineLayer(nn.Module):
    """
    A single SIREN layer: sin(ω₀ · (Wx + b))
    
    From paper Eq. 4: φ_i(x_i) = sin(W_i · x_i + b_i)
    
    The ω₀ factor controls the spatial frequency of the sine activation.
    For the first layer, ω₀ = 30 ensures the network can represent
    high-frequency content. For hidden layers, ω₀ = 30 is factored
    into the weight initialization (Section 1.5 of supplement).
    
    Args:
        in_features: Number of input features (fan_in)
        out_features: Number of output features  
        bias: Whether to include bias term
        is_first: Whether this is the first layer (different init scheme)
        omega_0: Frequency factor ω₀ (default: 30.0, per paper Section 3.2)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """
        Principled initialization from paper Section 3.2 + Supplement Theorem 1.8.
        
        First layer:
            W ~ U(-1/n, 1/n) where n = fan_in
            The ω₀ factor is applied in forward(), so the effective distribution
            of the pre-activation is ω₀ · U(-1/n, 1/n) which spans multiple
            periods of sine over the [-1, 1] input domain.
            
        Hidden layers (Section 1.5 factorization):
            W = Ŵ * ω₀, where Ŵ ~ U(-√(6/(ω₀²·n)), √(6/(ω₀²·n)))
            Equivalently: W ~ U(-√(6/n)/ω₀, √(6/n)/ω₀)
            Pre-multiplying by ω₀ in forward() gives effective:
            ω₀ · W ~ U(-√(6/n), √(6/n))
            Which is exactly the scheme from Theorem 1.8 that makes
            pre-activations N(0,1) distributed.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer: U(-1/n, 1/n)
                # ω₀ multiplication happens in forward()
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: factorized scheme (Section 1.5)
                # W ~ U(-√(6/n)/ω₀, √(6/n)/ω₀)
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sin(ω₀ · (Wx + b))
        
        The ω₀ scaling is applied to the linear output before the sine,
        matching Eq. 4 of the paper where the first layer uses
        sin(ω₀ · Wx + b) to span multiple periods.
        """
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both pre-activation (for gradient analysis) and activation."""
        intermediate = self.omega_0 * self.linear(x)
        return torch.sin(intermediate), intermediate


class SIREN(nn.Module):
    """
    Full SIREN network for trajectory segment representation.
    
    From paper Eq. 4:
        Φ(x) = W_n(φ_{n-1} ∘ φ_{n-2} ∘ ... ∘ φ_0)(x) + b_n
    
    The final layer is a standard linear layer (no sine activation),
    as the output should be in coordinate space, not [-1, 1].
    
    For cursor trajectories:
        - Input dim = 1 (normalized time t)
        - Output dim = 2 (x, y coordinates)
        - Hidden layers = 3, width = 64 (from implementation plan)
        - Total parameters ≈ 12,738 (1×64 + 64×64×3 + 64×2 + biases)
    
    Key property (Section 3.1): Any derivative of a SIREN is itself a SIREN,
    since d/dx sin(x) = cos(x) = sin(x + π/2). This means velocity and
    acceleration of fitted trajectories are analytic — no finite differences needed.
    
    Args:
        in_features: Input dimensionality (1 for time, 2 for spatial coords)
        hidden_features: Width of hidden layers (default: 64)
        hidden_layers: Number of hidden layers (default: 3)
        out_features: Output dimensionality (2 for x,y trajectory)
        outermost_linear: If True, final layer has no sine activation (default: True)
        first_omega_0: ω₀ for first layer (default: 30.0, paper recommendation)
        hidden_omega_0: ω₀ for hidden layers (default: 30.0)
    """

    def __init__(
        self,
        in_features: int = 1,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        out_features: int = 2,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        # Build network layer by layer
        layers = []

        # First layer: different initialization, uses first_omega_0
        layers.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        # Hidden layers: standard SIREN initialization with hidden_omega_0
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        # Final layer: linear (no sine) for coordinate output
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            # Initialize final layer with same scheme as hidden layers
            # (Section 1.5: U(-√(6/n)/ω₀, √(6/n)/ω₀))
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map normalized time to trajectory coordinates.
        
        Args:
            x: Input tensor of shape (batch, 1) with values in [0, 1]
               (or [-1, 1] for spatial coordinates)
        
        Returns:
            Coordinates tensor of shape (batch, 2) for (x, y) positions
        """
        return self.net(x)

    def forward_with_activations(self, x: torch.Tensor):
        """
        Forward pass returning all intermediate activations.
        Useful for analyzing the distribution of activations per layer
        (should be Arcsin distributed per Theorem 1.8).
        """
        activations = {"input": x}
        current = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                current, intermediate = layer.forward_with_intermediate(current)
                activations[f"layer_{i}_pre"] = intermediate
                activations[f"layer_{i}_post"] = current
            else:
                current = layer(current)
                activations[f"layer_{i}_linear"] = current
        return current, activations

    def get_weight_vector(self) -> torch.Tensor:
        """
        Flatten all parameters into a single vector.
        This is the "DNA" of the trajectory segment — used as input to VQ-VAE.
        
        Returns:
            1D tensor of all network parameters concatenated
        """
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_weight_vector(self, weights: torch.Tensor):
        """
        Load parameters from a flattened weight vector.
        Inverse of get_weight_vector() — used to reconstruct SIREN from VQ-VAE output.
        
        Args:
            weights: 1D tensor matching total parameter count
        """
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(weights[offset : offset + numel].view(p.shape))
            offset += numel

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def compute_trajectory(
        self,
        num_points: int = 100,
        t_start: float = 0.0,
        t_end: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a trajectory by evaluating SIREN at evenly-spaced time points.
        
        Args:
            num_points: Number of points to sample
            t_start: Start of time range (normalized)
            t_end: End of time range (normalized)
            device: Device to run on
            
        Returns:
            Tuple of (t, x, y) as numpy arrays
        """
        if device is None:
            device = next(self.parameters()).device

        t = torch.linspace(t_start, t_end, num_points, device=device).unsqueeze(-1)
        with torch.no_grad():
            coords = self.forward(t)

        t_np = t.cpu().numpy().flatten()
        x_np = coords[:, 0].cpu().numpy()
        y_np = coords[:, 1].cpu().numpy()
        return t_np, x_np, y_np

    def compute_velocity(
        self, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity analytically using autograd.
        
        Key SIREN property (Section 3.1): derivatives of SIRENs are SIRENs.
        d/dt sin(ωt) = ω·cos(ωt) = ω·sin(ωt + π/2)
        
        This gives exact velocities without finite-difference approximation.
        
        Args:
            t: Time tensor of shape (batch, 1), requires_grad=True
            
        Returns:
            Velocity tensor of shape (batch, 2) for (dx/dt, dy/dt)
        """
        t = t.requires_grad_(True)
        coords = self.forward(t)
        velocity = torch.autograd.grad(
            coords,
            t,
            grad_outputs=torch.ones_like(coords),
            create_graph=True,
        )[0]
        return velocity


class SIRENFitter:
    """
    Fits a SIREN to a single trajectory micro-segment.
    
    Takes raw (x, y, t) data, normalizes it, trains a fresh SIREN,
    and returns the fitted model with its flattened weight vector.
    
    Training uses Adam optimizer per paper's recommendation (Section 3.2):
    "The proposed initialization scheme yielded fast and robust convergence 
    using the ADAM optimizer for all experiments in this work."
    
    Args:
        hidden_features: SIREN hidden width (default: 64)
        hidden_layers: SIREN depth (default: 3)
        omega_0: Frequency factor (default: 30.0)
        lr: Learning rate for Adam (default: 1e-4)
        num_iterations: Training iterations (default: 500)
    """

    def __init__(
        self,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        lr: float = 1e-4,
        num_iterations: int = 500,
        device: Optional[torch.device] = None,
    ):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = omega_0
        self.lr = lr
        self.num_iterations = num_iterations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[SIREN, torch.Tensor, dict]:
        """
        Fit a SIREN to a trajectory segment.
        
        Args:
            t: Time array (microseconds or any units)
            x: X-coordinate array (pixels)
            y: Y-coordinate array (pixels)
            verbose: Print loss every 100 iterations
            
        Returns:
            Tuple of:
                - Trained SIREN model
                - Flattened weight vector (for VQ-VAE input)
                - Metadata dict with normalization params and final loss
        """
        # Normalize time to [0, 1]
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-10)

        # Normalize coordinates to [-1, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        coord_range = max(x_max - x_min, y_max - y_min, 1.0)
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        x_norm = (x - x_center) / (coord_range / 2 + 1e-10)
        y_norm = (y - y_center) / (coord_range / 2 + 1e-10)

        # Convert to tensors
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)
        coords_tensor = torch.tensor(
            np.stack([x_norm, y_norm], axis=-1),
            dtype=torch.float32,
            device=self.device,
        )

        # Create and train SIREN
        model = SIREN(
            in_features=1,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=2,
            first_omega_0=self.omega_0,
            hidden_omega_0=self.omega_0,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training loop — simple MSE, per paper Section 3.1:
        # L = Σ_i ||Φ(x_i) - f(x_i)||²
        final_loss = float("inf")
        for step in range(self.num_iterations):
            pred = model(t_tensor)
            loss = torch.nn.functional.mse_loss(pred, coords_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{self.num_iterations}: MSE = {final_loss:.6f}")

        # Compute reconstruction error in pixel space
        with torch.no_grad():
            pred_final = model(t_tensor).cpu().numpy()
            pred_x = pred_final[:, 0] * (coord_range / 2) + x_center
            pred_y = pred_final[:, 1] * (coord_range / 2) + y_center
            pixel_error = np.sqrt((pred_x - x) ** 2 + (pred_y - y) ** 2)

        metadata = {
            "t_min": float(t_min),
            "t_max": float(t_max),
            "x_center": float(x_center),
            "y_center": float(y_center),
            "coord_range": float(coord_range),
            "final_mse": final_loss,
            "mean_pixel_error": float(pixel_error.mean()),
            "max_pixel_error": float(pixel_error.max()),
            "num_points": len(t),
            "num_parameters": model.num_parameters,
        }

        weight_vector = model.get_weight_vector().detach()

        return model, weight_vector, metadata

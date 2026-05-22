"""
TrajectoryGen — Continuous Trajectory Generation Pipeline

A 6-stage ML pipeline that learns human cursor movement patterns
and generates naturalistic trajectories between arbitrary points.

Pipeline: Collect → Segment → SIREN INR → VQ-VAE → Latent ODE → Inference

Based on:
    - SIREN (Sitzmann et al., 2020): Periodic activation functions for INR
    - VQ-VAE (van den Oord et al., 2017): Discrete latent representation learning
    - Neural ODE (Chen et al., 2018): Continuous-depth models
    - Latent ODE (Rubanova et al., 2019): Irregularly-sampled time series
"""

__version__ = "2.0.1"

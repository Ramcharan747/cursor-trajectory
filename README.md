<div align="center">
  <h1>Mouse Trajectory ML Engine</h1>
  <p><b>Advanced Human-Mimicry Cursor Generation using Latent ODEs & Diffusion Models</b></p>
  
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
</div>

<br/>

This repository contains state-of-the-art machine learning models designed to learn, encode, and autonomously generate human-like mouse trajectories. It is specifically designed to bypass advanced bot detection heuristics (like Cloudflare Turnstile, DataDome, and reCAPTCHA v3) by mimicking the biological noise, latency, and acceleration curves of real human input.

> **Note:** The Go-based web scraping/extraction pipeline that utilizes these models has been separated into its own repository: [UltraSearch](https://github.com/Ramcharan747/UltraSearch).

## 🧠 Model Architecture

The repository contains three different generative approaches to solving the biological mimicry problem, located in the `trajectory_gen/models/` directory:

1. **Latent ODEs (`latent_ode.py`)**: Uses continuous-time neural ordinary differential equations to model the smooth acceleration and deceleration mechanics of human cursor movement across a screen.
2. **Conditional VAE (`cvae.py`)**: A Variational Autoencoder conditioned on start and end coordinates. It learns the distribution of possible human paths between two points and samples from that latent space.
3. **Diffusion Models (`diffusion.py`)**: Uses a denoising diffusion probabilistic model to iteratively refine a noisy path into a highly realistic, human-verified trajectory sequence.

## 🎯 Dataset Collection (`cursor_capture/`)

High-quality trajectory generation requires high-fidelity biological training data. 
We provide a high-performance **Rust-based** daemon that runs natively on MacOS and Windows to quietly capture thousands of natural human mouse movements. 

- Written in Rust for minimal CPU overhead.
- Operates strictly at the OS-level API to capture raw HID coordinates, velocity, and click latency.

### Building the Capture Daemon
```bash
cd cursor_capture
cargo build --release
./target/release/cursor_capture
```

## 🧪 Training

The models are trained on continuous sequences of `(x, y, timestamp)` coordinates. The training pipeline includes synthetic noise injection to ensure the model doesn't overfit to specific screen dimensions.

```bash
cd trajectory_gen
python train.py --model diffusion --epochs 100
```

## 🤝 Integration

To use the generated trajectories in your automation tools (like Puppeteer, Playwright, or Chromedp), use the output JSON payload from the inference engine. 

*(For a full end-to-end integration of these models into a stealth browser, check out the [UltraSearch](https://github.com/Ramcharan747/UltraSearch) project).*

## 📝 License
Distributed under the MIT License.

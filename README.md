<div align="center">

# 🖱️ Cursor Trajectory

**Learn how humans move — then generate it.**

A research system that captures real cursor movement at **per-pixel resolution**, decomposes it into mathematical primitives, and trains a Neural ODE to generate naturalistic trajectories between any two points.

[![Build](https://github.com/Ramcharan747/cursor-trajectory/actions/workflows/build.yml/badge.svg)](https://github.com/Ramcharan747/cursor-trajectory/actions)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey)]()

---

**CursorCapture** records every pixel your cursor visits · **TrajectoryGen** learns your movement patterns · **Generate** realistic paths on demand

</div>

---

## 🎯 What is this?

Most trajectory generation uses straight lines or Bézier curves. Real human cursor movement is far more complex — it has acceleration phases, micro-corrections, overshoots, and a rhythmic quality unique to each person.

This project takes a different approach:

1. **Record** every pixel the cursor visits during daily computer use (per-pixel, μs timestamps)
2. **Decompose** trajectories into mathematical building blocks (motion primitives) using SIREN neural networks
3. **Learn** the vocabulary of how you move using VQ-VAE
4. **Generate** new trajectories using Latent ODEs that are statistically indistinguishable from real movement

The result: given any two points on screen, output a continuous, naturalistic trajectory in under 500ms — with no seams, no straight lines, and no unnatural transitions.

> **Domain-agnostic by design.** Cursor movement is the test domain. The same architecture generalizes to game NPC movement, animation, handwriting synthesis, and robotics.

---

## 📦 CursorCapture — Data Collection

A **1.6MB Rust binary** that silently records cursor movement. Install once, forget forever.

### Why per-pixel?

Most recorders sample at a fixed rate (e.g. 60Hz = every 16ms). This **loses data** — if your cursor moves 200 pixels in 16ms, you only see the start and end, missing 198 points of the actual trajectory.

CursorCapture takes a different approach: **record every distinct pixel the cursor visits.** No time-based throttling. The OS reports a new position → we record it. Period.

Each event includes a **microsecond-precision timestamp** so you can compute velocity, acceleration, and jerk from the data without any interpolation guesswork.

### Features

| Feature | Detail |
|---------|--------|
| 🔬 **Per-pixel capture** | Records every distinct pixel position — zero spatial data loss |
| ⏱️ **Microsecond timestamps** | μs-precision timing for velocity & acceleration analysis |
| 💾 **Efficient storage** | JSONL format, hourly file rotation, auto-gzip after 24h |
| 🔒 **Privacy first** | Position + timestamp only. No screenshots, keystrokes, or window titles |
| 🔄 **Auto-start** | Runs on every login — macOS LaunchAgent / Windows Startup |
| 🛡️ **Self-healing** | Crash recovery via `KeepAlive` (macOS) / auto-restart |
| 📊 **Storage cap** | 500MB limit, oldest compressed files auto-deleted |
| 🪶 **Tiny footprint** | < 5MB RAM, single static binary, zero dependencies |

### Quick Install

<details>
<summary><b>🍎 macOS (Apple Silicon & Intel)</b></summary>

**Option 1: One-click installer**
1. Download the latest `.tar.gz` from [**Releases**](../../releases)
2. Extract and double-click **`install_mac.command`**
3. Grant Accessibility permission when the settings window opens
4. Done ✓ — runs in the background forever

**Option 2: Manual**
```bash
# Download and extract
tar -xzf cursor_capture-macos-*.tar.gz
cd dist

# Install (registers auto-start, opens permission dialog)
./cursor_capture install
```

> **Note:** macOS requires Accessibility permission for cursor monitoring. The installer opens the settings panel automatically — just add and enable `cursor_capture`.

</details>

<details>
<summary><b>🪟 Windows</b></summary>

**Option 1: One-click installer**
1. Download the latest `.zip` from [**Releases**](../../releases)
2. Extract and double-click **`install_win.bat`**
3. Done ✓ — no special permissions needed on Windows

**Option 2: Manual**
```powershell
# Just run the exe — it auto-installs on first launch
.\cursor_capture.exe
```

</details>

<details>
<summary><b>🔧 Build from source</b></summary>

```bash
git clone https://github.com/Ramcharan747/cursor-trajectory.git
cd cursor-trajectory/cursor_capture
cargo build --release

# Binary at: target/release/cursor_capture
./target/release/cursor_capture install
```

</details>

### Usage

```bash
cursor_capture              # Smart default: auto-install if needed, then run
cursor_capture status       # Check if running, data size, etc.
cursor_capture uninstall    # Remove auto-start (preserves data)
```

### Data Format

Each line is a distinct pixel position the cursor visited, with a microsecond timestamp:

```jsonl
{"x":1024.0,"y":768.0,"t":1714857600123456}
{"x":1025.0,"y":769.0,"t":1714857600124012}
{"x":1026.0,"y":770.0,"t":1714857600124589}
```

| Field | Type | Description |
|-------|------|-------------|
| `x` | `f64` | Horizontal position (integer pixels) |
| `y` | `f64` | Vertical position (integer pixels) |
| `t` | `i64` | Microseconds since Unix epoch (μs precision) |

**Why microseconds?** At per-pixel capture rates, consecutive events can be <1ms apart. Microsecond precision lets you compute instantaneous velocity and acceleration without rounding artifacts.

**Storage estimates (8 hours active use/day):**
- Standard mouse (125Hz): ~160MB/day raw → ~30MB compressed
- 500MB cap ≈ 2-3 weeks of continuous collection
- Auto-compresses files older than 24h, auto-deletes oldest when cap reached

---

## 🧠 TrajectoryGen — ML Pipeline

> **Status: Model architecture complete. Training begins after data collection.**

### Architecture Overview

```
Raw Data ──→ Segmentation ──→ SIREN INR ──→ VQ-VAE ──→ Latent ODE ──→ Trajectory
 (x,y,μs)     micro-cuts      per-segment    primitive    sequence      continuous
  per-pixel    at velocity     weight vector  library      generator    output
               dips, turns     compression    128 codes    ODE-RNN      (x,y,t)
```

### Model Details

All model implementations are paper-verified against the original research:

<details>
<summary><b>Stage 1: Segmentation</b></summary>

Cut continuous per-pixel recordings at meaningful boundaries:
- **Direction reversals** (>45° angle change between velocity vectors)
- **Velocity dips** (<5% of local peak speed — near-pauses)
- **Curvature inflection points** (2nd derivative magnitude > 3σ)
- Segments shorter than 50ms merged, longer than 3s split

</details>

<details>
<summary><b>Stage 2: SIREN INR Fitting</b></summary>

**Paper:** [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) (Sitzmann et al., 2020)

Each micro-segment (0.05–3s) is compressed into a SIREN network:

| Parameter | Value | Source |
|-----------|-------|--------|
| Layers | 1 input + 3 hidden + 1 linear output | — |
| Hidden width | 64 neurons | — |
| Activation | `sin(ω₀ · (Wx + b))` | Paper Eq. 4 |
| ω₀ (first layer) | 30.0 | Paper Section 3.2 |
| ω₀ (hidden layers) | 30.0 | Supplement Section 1.5 |
| Init (first layer) | `U(-1/fan_in, 1/fan_in)` | Paper Section 3.2 |
| Init (hidden) | `U(-√(6/n)/ω₀, √(6/n)/ω₀)` | Supplement Theorem 1.8 |
| Total params | ~12,738 per segment | — |
| Optimizer | Adam, lr=1e-4 | Paper Section 3.2 |
| Training | 500 iterations | — |

**Key property:** Derivatives of SIRENs are SIRENs (since d/dx sin = cos = sin(· + π/2)), so velocity and acceleration are computed analytically — no finite differences.

</details>

<details>
<summary><b>Stage 3: VQ-VAE Primitive Library</b></summary>

**Paper:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (van den Oord et al., 2017)

Compresses SIREN weight vectors into a discrete codebook of motion primitives:

| Parameter | Value | Source |
|-----------|-------|--------|
| Encoder | 12738 → 2048 → ReLU → 1024 → ReLU → 512 → ReLU → 256 | — |
| Codebook (K) | 512 entries × 256 dims | — |
| Decoder | 256 → 512 → ReLU → 1024 → ReLU → 2048 → ReLU → 12738 | — |
| Gradient | Straight-through estimator | Paper Section 3.2 |
| Codebook updates | EMA, γ=0.99 | Appendix A.1, Eq. 6-8 |
| Commitment cost (β) | 0.25 | Paper Section 3.2 |
| Loss | MSE + β·commitment | Paper Eq. 3 |

Each codebook entry = a reusable motion primitive (arc, line, hook, correction, etc.)

</details>

<details>
<summary><b>Stage 4: Latent ODE Generator</b></summary>

**Papers:**
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Chen et al., 2018)
- [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907) (Rubanova et al., 2019)

Sequences motion primitives using continuous-time dynamics:

| Parameter | Value | Source |
|-----------|-------|--------|
| Encoder | ODE-RNN (backwards in time) | Rubanova Eq. 8, Algorithm 1 |
| Recognition dim | 256 (> latent dim) | Supplement Section 5 |
| Latent dim | 64 | — |
| ODE function | 4-layer MLP × 512 hidden, **Tanh** activation | Supplement Section 4 |
| ODE solver | `dopri5` (adaptive RK 4/5) | Supplement Section 4 |
| Tolerances | rtol=1e-3, atol=1e-4 | Supplement Section 4 |
| Training | ELBO with KL annealing (coeff 0.99) | Supplement Section 6 |
| Memory | O(1) via adjoint method | Chen et al. Section 2 |
| Optimizer | Adamax, lr=0.01, decay=0.999 | Supplement Section 6 |

**Why Tanh?** From Rubanova supplement: *"Tanh activation constrains the output and prevents the ODE gradients from taking large values... we do not recommend using ReLU."*

</details>

### Training Strategy

Optimized for Google Colab (16GB VRAM, 4-hour sessions):
- Checkpoints every 10 minutes (survives disconnects)
- Mixed precision (fp16) for ~2× speedup on T4
- Gradient accumulation (4 steps) for larger effective batch
- VQ-VAE: batch 4096, effective 16384 with grad accumulation (~3-4GB)
- Latent ODE: batch 256, seq_len 20, adjoint method (~10-12GB)
- Precomputed SIREN weight vectors (no redundant INR fitting during training)
- Checkpoint storage on HuggingFace Hub

---

## 🏗️ Architecture

### CursorCapture (Rust)

```
┌─────────────────────┐                      ┌───────────────────┐
│  Listener Thread     │    mpsc channel      │  Writer Thread     │
│                      │ ──────────────────→  │                    │
│  rdev::listen()      │   CursorEvent        │  BufWriter<File>   │
│  • Per-pixel capture │   {x, y, t_μs}       │  • Batch writes    │
│  • Pixel dedup only  │                      │  • File rotation   │
│  • No time throttle  │                      │  • Hourly gzip     │
└─────────────────────┘                      │  • 500MB cap       │
         │                                    └───────────────────┘
  ┌──────┴──────┐
  │  Watchdog    │  Detects missing permissions
  │  Thread      │  Periodic health logging
  └─────────────┘
```

### TrajectoryGen Pipeline

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│  Record   │──→│  Segment    │──→│  SIREN     │──→│  VQ-VAE   │──→│  Latent  │
│  Per-pixel│   │  Direction  │   │  3×64      │   │  512 codes │   │  Latent  │
│  (x,y,μs) │   │  Velocity   │   │  sin(ω₀x)  │   │  EMA+CL   │   │  ODE-RNN │
│           │   │  Curvature  │   │ ~12.7K wts │   │ 256d embed │   │  Adjoint │
└──────────┘    └─────────────┘    └───────────┘    └───────────┘    └──────────┘
```

### Research Papers

The model architecture is built on these four papers:

| Paper | Year | Role in Pipeline | Key Contribution |
|-------|------|------------------|------------------|
| [SIREN](https://arxiv.org/abs/2006.09661) | 2020 | Trajectory segment encoding | Periodic activations + principled init (ω₀=30) |
| [VQ-VAE](https://arxiv.org/abs/1711.00937) | 2017 | Motion primitive codebook | Discrete latent learning + EMA codebook |
| [Neural ODE](https://arxiv.org/abs/1806.07366) | 2018 | Continuous dynamics backbone | Adjoint method for O(1) memory |
| [Latent ODE](https://arxiv.org/abs/1907.03907) | 2019 | Sequence generation | ODE-RNN encoder + VAE framework |

---

## 📁 Project Structure

```
cursor-trajectory/
├── cursor_capture/              # Rust data collection daemon
│   ├── src/
│   │   ├── main.rs              # CLI + smart auto-install
│   │   ├── recorder.rs          # Per-pixel capture, no time throttle
│   │   ├── storage.rs           # JSONL writer, rotation, compression
│   │   └── platform.rs          # Cross-platform auto-start
│   ├── install_mac.command      # macOS one-click installer
│   ├── install_win.bat          # Windows one-click installer
│   ├── Cargo.toml
│   └── README.md
├── trajectory_gen/              # ML pipeline
│   ├── models/
│   │   ├── siren.py             # SIREN INR (Sitzmann et al. 2020)
│   │   ├── vqvae.py             # VQ-VAE codebook (van den Oord et al. 2017)
│   │   ├── latent_ode.py        # Latent ODE generator (Rubanova et al. 2019)
│   │   └── inference.py         # End-to-end generation pipeline
│   ├── data/
│   │   ├── preprocessing.py     # JSONL loading, idle filtering
│   │   └── segmentation.py      # Direction/velocity/curvature cuts
│   ├── training/
│   │   └── colab_trainer.py     # Colab-optimized training loop
│   └── requirements.txt         # Python dependencies
├── papers/                      # Source research papers
├── .github/workflows/build.yml  # CI: auto-build Mac + Windows
└── README.md                    # ← You are here
```

---

## 🚀 Quick Start (ML Pipeline)

```bash
# Install dependencies
pip install -r trajectory_gen/requirements.txt

# Load and segment your data
python -c "
from trajectory_gen.data.preprocessing import load_all_recordings
from trajectory_gen.data.segmentation import segment_trajectory

x, y, t = load_all_recordings()
segments = segment_trajectory(x, y, t)
print(f'Found {len(segments)} segments from {len(x)} points')
"

# Fit SIRENs to segments (example)
python -c "
from trajectory_gen.models.siren import SIREN, SIRENFitter
fitter = SIRENFitter()
print(f'SIREN parameters: {SIREN().num_parameters}')
"
```

---

## 🤝 Contributing

This is a research project in active development. Contributions welcome:

- **Data collection improvements** — Multi-monitor support, click events
- **Segmentation algorithms** — New cut-point heuristics
- **Model architecture** — Alternative primitive representations
- **Platform support** — Linux support, system tray UI
- **Training recipes** — Hyperparameter tuning, new datasets

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If this project is useful to you, consider giving it a ⭐**

Built with 🦀 Rust and 🔥 PyTorch

</div>

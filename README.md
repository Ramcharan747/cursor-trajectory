<div align="center">

# 🖱️ Cursor Trajectory

**Learn how humans move — then generate it.**

A research system that captures real cursor movement at **per-pixel resolution**, decomposes it into mathematical primitives, and trains a Neural ODE to generate naturalistic trajectories between any two points.

[![Build](https://github.com/Ramcharan747/cursor-trajectory/actions/workflows/build.yml/badge.svg)](https://github.com/Ramcharan747/cursor-trajectory/actions)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey)]()

---

**CursorCapture** records every pixel your cursor visits · **TrajectoryGen** learns your movement patterns · **Generate** realistic paths on demand

</div>

---

## 🎯 What is this?

Most trajectory generation uses straight lines or Bézier curves. Real human cursor movement is far more complex — it has acceleration phases, micro-corrections, overshoots, and a rhythmic quality unique to each person.

This project takes a different approach:

1. **Record** every pixel the cursor visits during daily computer use
2. **Decompose** trajectories into mathematical building blocks (motion primitives) using SIREN neural networks
3. **Learn** the vocabulary of how you move using VQ-VAE
4. **Generate** new trajectories using Neural ODEs that are statistically indistinguishable from real movement

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

**Why microseconds?** At high sample rates (125–1000Hz), consecutive events can be <1ms apart. Microsecond precision lets you compute instantaneous velocity and acceleration without rounding artifacts.

**Storage estimates (8 hours active use/day):**
- Standard mouse (125Hz): ~160MB/day raw → ~30MB compressed
- 500MB cap ≈ 2-3 weeks of continuous collection
- Auto-compresses files older than 24h, auto-deletes oldest when cap reached

---

## 🧠 TrajectoryGen — ML Pipeline

> **Status: In development.** Data collection is live. The pipeline below is being built.

### Architecture

```
Raw Data ──→ Segmentation ──→ SIREN INR ──→ VQ-VAE ──→ Neural ODE ──→ Trajectory
 (x,y,t)      micro-cuts      per-segment    primitive    sequence      continuous
               at velocity     weight vector  library      generator    output
               dips, turns     compression    128 codes    Latent ODE   (x,y,t)
```

### Stage 1: Segmentation
Cut continuous recordings at meaningful boundaries:
- **Direction reversals** (>45° angle change)
- **Velocity dips** (<5% of peak speed)
- **Curvature inflection points** (3σ spike)

### Stage 2: SIREN INR Fitting
Each micro-segment (0.2–2s) is compressed into a tiny SIREN network:
- 3 layers × 64 neurons, sinusoidal activations
- Input: normalized time `t ∈ [0,1]` → Output: `(x, y)`
- ~8,500 parameters per segment = the segment's DNA
- FOMAML meta-learning for 100× faster fitting

### Stage 3: VQ-VAE Primitive Library
Cluster INR weight vectors into a codebook of ~128 motion primitives:
- Encoder: 8500 → 512 → 256 → 64 dims
- Codebook: 128 entries, EMA updates, commitment loss
- Each primitive = a reusable building block of movement

### Stage 4: Neural ODE Generator
A Latent ODE that takes start/end coordinates and sequences primitives:
- Adjoint method for O(1) memory training
- Soft attention over the primitive codebook
- C² continuity guaranteed by integration-based architecture
- Target: <500ms inference on CPU

### Training Strategy
Optimized for Google Colab (16GB VRAM, 4-hour sessions):
- Checkpoints to HuggingFace Hub every 10 minutes
- Mixed precision (fp16) + gradient accumulation
- Precomputed INR weights (no redundant fitting during training)

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
│  Record   │──→│  Segment    │──→│  SIREN     │──→│  VQ-VAE   │──→│  Neural  │
│  Per-pixel│   │  Direction  │   │  3×64      │   │  128 codes │   │  ODE     │
│  (x,y,μs) │   │  Velocity   │   │  sin(ωx)   │   │  EMA+CL   │   │  Latent  │
│           │   │  Curvature  │   │  ~8.5K wts │   │  64d embed │   │  Adjoint │
└──────────┘    └─────────────┘    └───────────┘    └───────────┘    └──────────┘
```

---

## 📁 Project Structure

```
cursor-trajectory/
├── cursor_capture/              # Rust data collection daemon
│   ├── src/
│   │   ├── main.rs              # CLI + smart auto-install
│   │   ├── recorder.rs          # Per-pixel capture, no throttle
│   │   ├── storage.rs           # JSONL writer, rotation, compression
│   │   └── platform.rs          # Cross-platform auto-start
│   ├── install_mac.command      # macOS one-click installer
│   ├── install_win.bat          # Windows one-click installer
│   ├── Cargo.toml
│   └── README.md
├── trajectory_gen/              # ML pipeline (coming soon)
│   ├── models/                  # SIREN, VQ-VAE, Latent ODE
│   ├── data/                    # Preprocessing, segmentation
│   └── training/                # Colab training scripts
├── notebooks/                   # Experiment notebooks
├── .github/workflows/build.yml  # CI: auto-build Mac + Windows
└── README.md                    # ← You are here
```

---

## 🤝 Contributing

This is a research project in active development. Contributions welcome:

- **Data collection improvements** — Multi-monitor support, click events
- **Segmentation algorithms** — New cut-point heuristics
- **Model architecture** — Alternative to VQ-VAE for primitive library
- **Platform support** — Linux support, system tray UI

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If this project is useful to you, consider giving it a ⭐**

Built with 🦀 Rust and 🔥 PyTorch

</div>

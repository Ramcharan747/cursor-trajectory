# CursorCapture

A lightweight, cross-platform cursor movement recorder built in Rust. Records every pixel your cursor visits with microsecond timestamps — zero spatial data loss.

**1.6MB binary. Zero dependencies. Fire and forget.**

## Quick Start

### macOS

```bash
# Download or build the binary, then:
./cursor_capture install
```

This does two things:
1. Registers auto-start on login (via macOS LaunchAgent)
2. Starts recording immediately

**⚠️ Required: Grant Accessibility Permission**

After first run, go to:
> **System Settings → Privacy & Security → Accessibility**

Add and enable the `cursor_capture` binary. Without this, macOS blocks input monitoring.

### Windows

```bat
cursor_capture.exe install
```

This registers a Startup entry and starts recording. No special permissions needed.

## Commands

| Command | Description |
|---------|-------------|
| `cursor_capture` | Smart default: auto-install if needed, then run |
| `cursor_capture run` | Start recording |
| `cursor_capture install` | Register auto-start + begin recording |
| `cursor_capture uninstall` | Remove auto-start registration |
| `cursor_capture status` | Show current status, data directory, disk usage |

## Data Format

Events are stored as JSONL (one JSON object per line) in `~/cursor_capture_data/`:

```json
{"x":1024.0,"y":768.0,"t":1714857600123456}
{"x":1025.0,"y":769.0,"t":1714857600124012}
{"x":1026.0,"y":770.0,"t":1714857600124589}
```

- `x`, `y` — cursor position in integer screen pixels
- `t` — microseconds since Unix epoch (μs precision)

### Per-pixel capture

Unlike fixed-rate samplers (60Hz, 120Hz), CursorCapture records **every distinct pixel** the cursor visits. There is no time-based throttle — if your cursor moves through 200 pixels, all 200 are recorded.

The only filter is position deduplication: if the cursor hasn't moved to a new pixel, nothing is recorded.

### Microsecond timestamps

Timestamps are in microseconds (not milliseconds) for sub-millisecond precision. This is critical for computing accurate velocity and acceleration at high sample rates.

### File Rotation

- New file every hour: `cursor_2026-05-04_14.jsonl`
- Files older than 24 hours are auto-compressed to `.jsonl.gz`
- Total storage capped at 500MB (oldest compressed files deleted first)

## Building from Source

```bash
cargo build --release
# Binary at: target/release/cursor_capture
```

## Architecture

```
┌─────────────────────┐     channel      ┌──────────────┐
│  rdev::listen        │ ──────────────→  │ Writer Thread │
│  (mouse events)      │  CursorEvent    │ (batch JSONL) │
│  per-pixel capture   │  {x, y, t_μs}  │ file rotation │
│  pixel dedup only    │                  │ compression   │
└─────────────────────┘                  └──────────────┘
         │
  ┌──────┴──────┐
  │  Watchdog    │  Permission detection
  │  Thread      │  Health logging
  └─────────────┘
```

## Privacy

This tool records cursor positions only (no screenshots, no keystrokes, no window titles). Data stays local on your machine in `~/cursor_capture_data/`. No data is transmitted anywhere.

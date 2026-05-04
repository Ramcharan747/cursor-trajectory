# CursorCapture

A lightweight, cross-platform cursor movement recorder built in Rust. Records mouse movement at 60Hz to JSONL files for trajectory analysis research.

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

This registers a Run entry in the Windows Registry and starts recording.

## Commands

| Command | Description |
|---------|-------------|
| `cursor_capture run` | Start recording (default if no command given) |
| `cursor_capture install` | Register auto-start + begin recording |
| `cursor_capture uninstall` | Remove auto-start registration |
| `cursor_capture status` | Show current status, data directory, disk usage |

## Data Format

Events are stored as JSONL (one JSON object per line) in `~/cursor_capture_data/`:

```json
{"x":1024.0,"y":768.0,"t":1714857600123}
{"x":1025.5,"y":770.2,"t":1714857600139}
```

- `x`, `y` — cursor position in screen coordinates (pixels)
- `t` — milliseconds since Unix epoch (UTC)

### File Rotation

- New file every hour: `cursor_2026-05-04_14.jsonl`
- Files older than 24 hours are auto-compressed to `.jsonl.gz`
- Total storage capped at 500MB (oldest compressed files deleted first)

## Data Volume Estimates

At 60Hz recording:
- ~1 hour active use ≈ 5MB raw JSONL
- ~8 hours/day ≈ 40MB/day raw → ~8MB/day compressed
- ~500MB cap ≈ ~2 months of data

## Building from Source

```bash
# macOS / Linux
cargo build --release
# Binary at: target/release/cursor_capture

# Windows (cross-compile from Mac/Linux requires toolchain setup)
# Best to build natively on Windows:
cargo build --release
# Binary at: target\release\cursor_capture.exe
```

## Architecture

```
┌─────────────────┐     channel      ┌──────────────┐
│  rdev::listen   │ ──────────────→  │ Writer Thread │
│  (mouse events) │  CursorEvent     │ (batch JSONL) │
│  60Hz throttle  │                  │ file rotation │
│  idle detection │                  │ compression   │
└─────────────────┘                  └──────────────┘
```

- **Listener thread**: captures `MouseMove` events via `rdev`, throttles to 60Hz, filters idle periods
- **Writer thread**: receives events via channel, buffers 1000 events, batch-writes to JSONL, handles file rotation and storage maintenance

## Privacy

This tool records cursor positions only (no screenshots, no keystrokes, no window titles). Data stays local on your machine in `~/cursor_capture_data/`. No data is transmitted anywhere.

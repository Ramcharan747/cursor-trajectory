"""
Data preprocessing — loads raw JSONL cursor recordings.

Handles the per-pixel capture format with microsecond timestamps:
    {"x":1024.0,"y":768.0,"t":1714857600123456}
"""

import json
import gzip
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_recording(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single JSONL recording file.
    
    Supports both plain .jsonl and compressed .jsonl.gz files.
    
    Returns:
        Tuple of (x, y, t) numpy arrays
    """
    x_list, y_list, t_list = [], [], []
    
    path = Path(filepath)
    opener = gzip.open if path.suffix == '.gz' else open
    
    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                x_list.append(float(event['x']))
                y_list.append(float(event['y']))
                t_list.append(int(event['t']))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    
    return np.array(x_list), np.array(y_list), np.array(t_list, dtype=np.int64)


def load_all_recordings(
    data_dir: str = "~/cursor_capture_data",
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all recording files from the data directory.
    
    Returns concatenated (x, y, t) arrays sorted by timestamp.
    """
    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = sorted(
        list(data_dir.glob("cursor_*.jsonl")) + list(data_dir.glob("cursor_*.jsonl.gz"))
    )
    
    if max_files:
        files = files[:max_files]
    
    all_x, all_y, all_t = [], [], []
    for f in files:
        x, y, t = load_recording(str(f))
        if len(x) > 0:
            all_x.append(x)
            all_y.append(y)
            all_t.append(t)
    
    if not all_x:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)
    
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    
    # Sort by timestamp
    order = np.argsort(t)
    return x[order], y[order], t[order]


def filter_idle_periods(
    x: np.ndarray, y: np.ndarray, t: np.ndarray,
    idle_threshold_s: float = 2.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split recording at idle periods (no movement for > threshold).
    
    Returns list of (x, y, t) tuples for each active period.
    """
    if len(t) < 2:
        return [(x, y, t)]
    
    dt = np.diff(t) / 1e6  # microseconds → seconds
    idle_mask = dt > idle_threshold_s
    split_indices = np.where(idle_mask)[0] + 1
    
    periods = []
    boundaries = [0] + split_indices.tolist() + [len(x)]
    
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= 5:  # Minimum 5 points
            periods.append((x[s:e], y[s:e], t[s:e]))
    
    return periods

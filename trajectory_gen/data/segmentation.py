"""
Trajectory Segmentation — cuts continuous cursor recordings at natural boundaries.

Detects segment boundaries using three complementary signals:
    1. Direction reversals (angle change > threshold)
    2. Velocity dips (speed < fraction of local peak)
    3. Curvature inflection points (2nd derivative spikes)

Each segment becomes one SIREN fitting target.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Segment:
    """A trajectory micro-segment ready for SIREN fitting."""
    t: np.ndarray       # Timestamps (microseconds)
    x: np.ndarray       # X coordinates (pixels)
    y: np.ndarray       # Y coordinates (pixels)
    start_idx: int      # Index in original recording
    end_idx: int        # Index in original recording
    duration_s: float   # Duration in seconds
    num_points: int     # Number of data points


def compute_velocity(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocity components and speed from position + time arrays."""
    dt = np.diff(t).astype(np.float64)
    dt = np.maximum(dt, 1.0)  # Prevent div-by-zero (min 1 microsecond)
    
    vx = np.diff(x) / dt * 1e6  # pixels/second (t is in microseconds)
    vy = np.diff(y) / dt * 1e6
    speed = np.sqrt(vx**2 + vy**2)
    
    return vx, vy, speed


def compute_direction(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Compute direction angle (radians) from velocity components."""
    return np.arctan2(vy, vx)


def compute_curvature(vx: np.ndarray, vy: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Compute unsigned curvature from velocity components."""
    if len(vx) < 2:
        return np.array([0.0])
    
    dvx = np.diff(vx)
    dvy = np.diff(vy)
    
    # κ = |v × a| / |v|³ (2D cross product)
    cross = np.abs(vx[:-1] * dvy - vy[:-1] * dvx)
    speed_cubed = np.maximum(speed[:-1] ** 3, 1e-10)
    
    return cross / speed_cubed


def detect_cut_points(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    angle_threshold: float = 45.0,
    velocity_dip_fraction: float = 0.05,
    curvature_sigma: float = 3.0,
    min_segment_duration_s: float = 0.1,
    max_segment_duration_s: float = 3.0,
) -> List[int]:
    """
    Detect natural boundary points in a trajectory.
    
    Returns indices where segments should be split.
    """
    if len(x) < 5:
        return []
    
    vx, vy, speed = compute_velocity(x, y, t)
    
    if len(speed) < 3:
        return []
    
    cuts = set()
    
    # 1. Direction reversals
    direction = compute_direction(vx, vy)
    angle_change = np.abs(np.diff(direction))
    # Wrap angles to [0, π]
    angle_change = np.minimum(angle_change, 2 * np.pi - angle_change)
    threshold_rad = np.radians(angle_threshold)
    
    for i in np.where(angle_change > threshold_rad)[0]:
        cuts.add(i + 1)  # +1 because diff reduces length
    
    # 2. Velocity dips
    if len(speed) > 10:
        # Use rolling window to find local peak speed
        window = min(20, len(speed) // 3)
        for i in range(window, len(speed)):
            local_peak = np.max(speed[max(0, i-window):i])
            if local_peak > 0 and speed[i] < velocity_dip_fraction * local_peak:
                cuts.add(i + 1)
    
    # 3. Curvature spikes
    curvature = compute_curvature(vx, vy, speed)
    if len(curvature) > 5:
        mean_k = np.mean(curvature)
        std_k = np.std(curvature)
        if std_k > 0:
            for i in np.where(curvature > mean_k + curvature_sigma * std_k)[0]:
                cuts.add(i + 2)
    
    # Sort and filter
    cuts = sorted(cuts)
    cuts = [c for c in cuts if 0 < c < len(x)]
    
    return cuts


def segment_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    min_points: int = 10,
    min_duration_s: float = 0.05,
    max_duration_s: float = 3.0,
    **kwargs,
) -> List[Segment]:
    """
    Segment a full trajectory recording into micro-segments.
    
    Args:
        x, y, t: Full recording arrays
        min_points: Minimum points per segment
        min_duration_s: Minimum segment duration (seconds)
        max_duration_s: Maximum segment duration (seconds)
        **kwargs: Passed to detect_cut_points
    
    Returns:
        List of Segment objects
    """
    cuts = detect_cut_points(x, y, t, **kwargs)
    
    # Add boundaries
    boundaries = [0] + cuts + [len(x)]
    
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        
        if end - start < min_points:
            continue
        
        seg_t = t[start:end]
        duration = (seg_t[-1] - seg_t[0]) / 1e6  # microseconds → seconds
        
        if duration < min_duration_s:
            continue
        
        # Split oversized segments
        if duration > max_duration_s:
            mid = (start + end) // 2
            sub_boundaries = [start, mid, end]
            for j in range(len(sub_boundaries) - 1):
                s, e = sub_boundaries[j], sub_boundaries[j + 1]
                if e - s >= min_points:
                    st = t[s:e]
                    dur = (st[-1] - st[0]) / 1e6
                    segments.append(Segment(
                        t=st, x=x[s:e], y=y[s:e],
                        start_idx=s, end_idx=e,
                        duration_s=dur, num_points=e - s,
                    ))
        else:
            segments.append(Segment(
                t=seg_t, x=x[start:end], y=y[start:end],
                start_idx=start, end_idx=end,
                duration_s=duration, num_points=end - start,
            ))
    
    return segments

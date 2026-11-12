"""Stage 2 — Netflix h4 motion representation (Chen et al., WACV 2023, Sec. 3.3),
plus a pooled flow time series used by the montage stages.

h4: for a shot with frames f_1..f_l the paper computes

    q_j = OF(f_j, f_{j+1})                        dense flow, H x W x 2
    Q   = sum_j q_j / (l - 1)                     temporal mean (Eq. 5)
    r   = flatten(Q)                              shot representation

and compares shots with cosine similarity. OF is OpenCV's Farneback method.
"In practice, we sample every four frames": we take flow start frames
j = 0, stride, 2*stride, ... and compute flow from f_j to f_{j+gap}
(gap = 1 by default, i.e. adjacent frames, matching Eq. 5).

Time series (our addition, not h4): adjacent-frame flow every `ts_stride`
frames, average-pooled to a grid_w x grid_h grid, so a cut can be scored on
the motion right before / after a specific frame instead of the whole clip.

Outputs: outputs/motion/<clip_id>.npz, <clip_id>_flow.jpg, summary.csv, config.json
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import FlowConfig, OutputPaths
from .video import read_frames


@dataclass
class MotionRep:
    clip_id: str
    mean_flow: np.ndarray   # (H, W, 2) float32, units: pixels/frame at flow resolution
    n_pairs: int            # how many flow fields were averaged for h4
    ts_flow: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 1, 2), np.float32))
    ts_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # start frame j of each ts flow

    @property
    def vector(self) -> np.ndarray:
        """r = flatten(Q)."""
        return self.mean_flow.reshape(-1)

    @property
    def mean_dx(self) -> float:
        return float(self.mean_flow[..., 0].mean())

    @property
    def mean_dy(self) -> float:
        return float(self.mean_flow[..., 1].mean())

    @property
    def mean_magnitude(self) -> float:
        """Mean per-pixel length of the time-averaged flow (net, coherent motion)."""
        return float(np.linalg.norm(self.mean_flow, axis=-1).mean())

    def save(self, path: Path, cfg: FlowConfig) -> None:
        np.savez_compressed(
            path, mean_flow=self.mean_flow, n_pairs=self.n_pairs, clip_id=self.clip_id,
            ts_flow=self.ts_flow, ts_frame=self.ts_frame, config=json.dumps(asdict(cfg)),
        )

    @classmethod
    def load(cls, path: Path) -> "MotionRep":
        d = np.load(path)
        return cls(str(d["clip_id"]), d["mean_flow"], int(d["n_pairs"]), d["ts_flow"], d["ts_frame"])


def farneback(prev: np.ndarray, nxt: np.ndarray, cfg: FlowConfig) -> np.ndarray:
    """Dense flow (H, W, 2) from grayscale `prev` to `nxt`: where each pixel of prev moves to."""
    return cv2.calcOpticalFlowFarneback(
        prev, nxt, None,
        cfg.pyr_scale, cfg.levels, cfg.winsize, cfg.iterations,
        cfg.poly_n, cfg.poly_sigma, 0,
    )


def analyze_frames(
    frames: Sequence[np.ndarray], cfg: FlowConfig, mask: Optional[np.ndarray] = None,
    time_series: bool = True,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """One pass over a clip's frames.

    frames: grayscale uint8 frames, all at the flow resolution.
    mask:   optional (H, W) array, 1 inside the picture, 0 on letterbox bars.
    Returns (mean_flow, n_pairs, ts_flow, ts_frame):
      mean_flow  h4 temporal mean (Eq. 5), divided by `gap` so units stay px/frame
      ts_flow    (T, grid_h, grid_w, 2) pooled adjacent-frame flows
      ts_frame   (T,) start frame j of each ts flow (flow from j to j+1)
    """
    n = len(frames)
    h4_starts = set(range(0, n - cfg.gap, cfg.stride))
    ts_starts = set(range(0, n - 1, cfg.ts_stride)) if time_series else set()
    if not h4_starts:
        raise ValueError(f"clip too short: {n} frames, need more than gap={cfg.gap}")
    m = None if mask is None else mask[..., None]

    total = None
    ts_flow, ts_frame = [], []
    for j in sorted(h4_starts | ts_starts):
        adjacent = farneback(frames[j], frames[j + 1], cfg) if (j in ts_starts or cfg.gap == 1) else None
        if j in h4_starts:
            q = adjacent if cfg.gap == 1 else farneback(frames[j], frames[j + cfg.gap], cfg)
            total = q if total is None else total + q
        if j in ts_starts:
            f = adjacent if m is None else adjacent * m
            ts_flow.append(cv2.resize(f, (cfg.grid_w, cfg.grid_h), interpolation=cv2.INTER_AREA))
            ts_frame.append(j)

    mean = total / (len(h4_starts) * cfg.gap)
    if m is not None:
        mean = mean * m
    ts = np.array(ts_flow, np.float32) if ts_flow else np.zeros((0, cfg.grid_h, cfg.grid_w, 2), np.float32)
    return mean.astype(np.float32), len(h4_starts), ts, np.array(ts_frame, np.int32)


def mean_flow_from_frames(
    frames: Sequence[np.ndarray], cfg: FlowConfig, mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, int]:
    """h4 only: (temporal-mean flow, number of flow pairs averaged)."""
    mean, n_pairs, _, _ = analyze_frames(frames, cfg, mask, time_series=False)
    return mean, n_pairs


def content_mask(box: Sequence[int], frame_size: tuple[int, int], flow_size: tuple[int, int]) -> np.ndarray:
    """(H, W) mask at flow resolution that is 1 inside the content box, 0 on padding."""
    x, y, w, h = box
    sx = flow_size[0] / frame_size[0]
    sy = flow_size[1] / frame_size[1]
    mask = np.zeros((flow_size[1], flow_size[0]), np.float32)
    x0, y0 = int(np.ceil(x * sx)), int(np.ceil(y * sy))
    x1, y1 = int(np.floor((x + w) * sx)), int(np.floor((y + h) * sy))
    mask[y0:y1, x0:x1] = 1.0
    return mask


def extract_motion(clip: dict, cfg: FlowConfig) -> MotionRep:
    """MotionRep for one preprocessed clip (a manifest entry)."""
    flow_size = (cfg.flow_width, cfg.flow_height)
    frames = list(read_frames(Path(clip["path"]), size=flow_size, gray=True))
    mask = content_mask(clip["content_box"], (clip["width"], clip["height"]), flow_size)
    mean_flow, n_pairs, ts_flow, ts_frame = analyze_frames(frames, cfg, mask)
    return MotionRep(clip["id"], mean_flow, n_pairs, ts_flow, ts_frame)


def run_motion(manifest: dict, paths: OutputPaths, cfg: FlowConfig) -> list[MotionRep]:
    from .visualize import flow_panel  # local import keeps motion.py usable without matplotlib

    paths.motion.mkdir(parents=True, exist_ok=True)
    reps = []
    for i, clip in enumerate(manifest["clips"], 1):
        rep = extract_motion(clip, cfg)
        rep.save(paths.motion / f"{rep.clip_id}.npz", cfg)
        cv2.imwrite(
            str(paths.motion / f"{rep.clip_id}_flow.jpg"),
            flow_panel(rep.mean_flow, label=f"{rep.clip_id}  |v|={rep.mean_magnitude:.2f} px/frame"),
        )
        print(f"[motion] ({i}/{len(manifest['clips'])}) {rep.clip_id}: "
              f"{rep.n_pairs} h4 flow pairs, {len(rep.ts_frame)} time-series flows, "
              f"mean |v| = {rep.mean_magnitude:.3f} px/frame")
        reps.append(rep)

    with open(paths.motion / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "n_pairs", "mean_dx", "mean_dy", "mean_magnitude"])
        for r in reps:
            writer.writerow([r.clip_id, r.n_pairs, f"{r.mean_dx:.4f}", f"{r.mean_dy:.4f}", f"{r.mean_magnitude:.4f}"])
    (paths.motion / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"[motion] wrote {len(reps)} representations -> {paths.motion}")
    return reps


def load_reps(manifest: dict, paths: OutputPaths) -> list[MotionRep]:
    reps = []
    for clip in manifest["clips"]:
        p = paths.motion / f"{clip['id']}.npz"
        if not p.exists():
            raise SystemExit(f"{p} not found — run the 'motion' stage first")
        reps.append(MotionRep.load(p))
    return reps

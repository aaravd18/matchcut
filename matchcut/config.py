"""Default parameters for every pipeline stage, plus the output folder layout.

All defaults live here so they are easy to find and change. The CLI overrides a
subset of them; every stage records the values it used next to its outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessConfig:
    width: int = 640          # output frame width (pixels)
    height: int = 360         # output frame height (pixels)
    fps: int = 30             # output frame rate
    duration: float = 20.0    # max seconds kept from each source clip (room to choose cut points)
    window: str = "center"    # which part of the source to keep: "center" or "start"
    crf: int = 18             # libx264 quality (lower = better, 18 is visually lossless)


@dataclass
class FlowConfig:
    """Netflix h4: Farneback flow -> temporal mean -> flatten -> cosine.

    The paper does not state resolution, sampling details or Farneback
    parameters; these are our choices (see README "Deviations from the paper").
    """
    flow_width: int = 256     # flow is computed on frames downscaled to this size
    flow_height: int = 144
    stride: int = 4           # h4: use every `stride`-th frame as a flow start frame ("every four frames")
    gap: int = 1              # h4: flow between frame j and j+gap (1 = adjacent frames, as in Eq. 5)
    # cv2.calcOpticalFlowFarneback parameters (OpenCV tutorial defaults)
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    # Flow time series (used to score cuts at specific moments, not part of h4)
    ts_stride: int = 2        # adjacent-frame flow every `ts_stride` frames
    grid_w: int = 16          # each flow field is average-pooled to grid_w x grid_h cells
    grid_h: int = 9


@dataclass
class SequenceConfig:
    """Montage: cut-point scoring, clip ordering and segment placement."""
    min_motion: float = 0.05      # px/frame (at flow resolution) below which motion is treated as absent
    window: float = 0.5           # seconds of flow compared on each side of a cut
    cut_step: float = 0.25        # spacing of candidate cut points (seconds)
    seg_min: float = 1.5          # allowed length of each clip in the montage (seconds)
    seg_max: float = 3.5
    seg_target: float = 2.5
    length_weight: float = 0.1    # penalty per unit of |len - target| / target
    w_motion: float = 0.6         # transition score = w_motion*motion + w_comp*composition + w_color*color
    w_comp: float = 0.25
    w_color: float = 0.15
    max_clips: Optional[int] = None   # None = use every usable clip; N = best-connected N clips
    order: Optional[list] = None      # fixed clip order (ids or short names); skips the search


@dataclass
class RenderConfig:
    top_n: int = 5            # render h4 pair previews for the best N pairs
    seconds_a: float = 2.5    # preview: clip A before the cut
    seconds_b: float = 2.5    # preview: clip B after the cut
    crf: int = 18


@dataclass(frozen=True)
class OutputPaths:
    """Folder layout under the output root (default: outputs/)."""
    root: Path

    @property
    def preprocessed(self) -> Path:
        return self.root / "preprocessed"

    @property
    def manifest(self) -> Path:
        return self.preprocessed / "manifest.json"

    @property
    def motion(self) -> Path:
        return self.root / "motion"

    @property
    def similarity(self) -> Path:
        return self.root / "similarity"

    @property
    def ranked_pairs(self) -> Path:
        return self.similarity / "ranked_pairs.json"

    @property
    def transitions(self) -> Path:
        return self.root / "transitions"

    @property
    def montage(self) -> Path:
        return self.root / "montage"

    @property
    def edl(self) -> Path:
        return self.montage / "edl.json"

    @property
    def comparisons(self) -> Path:
        return self.root / "comparisons"

    @property
    def report(self) -> Path:
        return self.root / "report"

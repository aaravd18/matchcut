"""Flow and similarity visualizations (images only; HTML lives in report.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

# Diverging scale for cosine similarity: blue (-1) -> neutral gray (0) -> red (+1).
DIVERGING_NEG = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_POS = "#e34948"


def flow_to_bgr(flow: np.ndarray, max_mag: Optional[float] = None) -> np.ndarray:
    """Standard HSV flow coding: hue = direction, brightness = magnitude.

    max_mag sets which magnitude maps to full brightness; default is the
    99th percentile of this field, so each panel uses its own scale.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99))
    max_mag = max(max_mag, 1e-6)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.int32) % 180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_panel(
    flow: np.ndarray,
    size: tuple[int, int] = (640, 360),
    arrow_spacing: int = 40,
    label: Optional[str] = None,
) -> np.ndarray:
    """HSV flow image upscaled to `size` with a grid of arrows on top.

    Arrows show the local average flow, rescaled so the longest arrow is about
    half the grid spacing (direction is exact; length is relative within the panel).
    """
    img = cv2.resize(flow_to_bgr(flow), size, interpolation=cv2.INTER_LINEAR)
    h, w = flow.shape[:2]
    sx = size[0] / w
    cell = max(1, int(round(arrow_spacing / sx)))

    starts, vecs = [], []
    for cy in range(cell // 2, h, cell):
        for cx in range(cell // 2, w, cell):
            v = flow[max(0, cy - cell // 2): cy + cell // 2 + 1,
                     max(0, cx - cell // 2): cx + cell // 2 + 1].reshape(-1, 2).mean(0)
            starts.append((cx * sx, cy * sx))
            vecs.append(v)
    vecs = np.array(vecs)
    longest = float(np.linalg.norm(vecs, axis=1).max()) if len(vecs) else 0.0
    if longest > 1e-3:
        scale = 0.5 * arrow_spacing / longest
        for (x, y), v in zip(starts, vecs):
            p0 = (int(x), int(y))
            p1 = (int(x + v[0] * scale), int(y + v[1] * scale))
            if p0 == p1:
                continue
            cv2.arrowedLine(img, p0, p1, (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.3)
            cv2.arrowedLine(img, p0, p1, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)

    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(img, label, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def color_wheel(size: int = 96) -> np.ndarray:
    """Legend for flow_to_bgr: the colour at each point is the colour of flow pointing that way."""
    c = (size - 1) / 2
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    flow = np.dstack([xs - c, ys - c])
    img = flow_to_bgr(flow, max_mag=c)
    outside = (xs - c) ** 2 + (ys - c) ** 2 > c ** 2
    img[outside] = 255
    return img


def similarity_heatmap(
    sim: np.ndarray, labels: Sequence[str], path: Path,
    title: str = "Motion similarity (h4)", cbar_label: str = "cosine similarity of mean flow (h4)",
    xlabel: Optional[str] = None, ylabel: Optional[str] = None,
) -> None:
    """Annotated similarity matrix as a PNG (diverging scale centred on 0; NaN cells left blank)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])
    n = len(labels)
    side = max(4.0, 0.7 * n + 2.0)
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    im = ax.imshow(sim, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), labels, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    if n <= 25:
        for i in range(n):
            for j in range(n):
                if np.isfinite(sim[i, j]):
                    ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#1f1f1e")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

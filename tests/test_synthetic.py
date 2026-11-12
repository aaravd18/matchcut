"""Synthetic sanity check for the h4 pipeline (no video files needed).

Generates short grayscale clips with known motion over random textures
(Farneback needs texture; flat backgrounds give flow only at edges), runs the
same flow + cosine code as the real pipeline, and checks the expected ordering.

Run:   python tests/test_synthetic.py          (also writes outputs/synthetic/)
  or:  python -m pytest tests/test_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matchcut.config import FlowConfig  # noqa: E402
from matchcut.motion import mean_flow_from_frames  # noqa: E402
from matchcut.similarity import cosine_matrix  # noqa: E402

W, H, N_FRAMES = 256, 144, 40
CFG = FlowConfig(flow_width=W, flow_height=H)


def texture(seed: int, w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 1.5)
    t = (t - t.min()) / (t.max() - t.min()) * 255
    return t.astype(np.uint8)


def pan(seed: int, vx: float, vy: float) -> list[np.ndarray]:
    """Camera translating by (vx, vy) px/frame over a large texture: content moves by (-vx, -vy)."""
    canvas = texture(seed, W + 200, H + 200)
    frames = []
    for t in range(N_FRAMES):
        x0, y0 = 100 + vx * t, 100 + vy * t
        m = np.float32([[1, 0, -x0], [0, 1, -y0]])  # sub-pixel crop
        frames.append(cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_LINEAR))
    return frames


def zoom(seed: int, rate: float) -> list[np.ndarray]:
    """Zoom about the centre: rate > 0 zooms in (content expands outward)."""
    canvas = texture(seed, W, H)
    frames = []
    for t in range(N_FRAMES):
        s = 1.0 + rate * t
        m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        frames.append(cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return frames


def moving_object(seed: int, y_center: int, vx: float) -> list[np.ndarray]:
    """Static background with a 40x40 textured square moving horizontally at height y_center."""
    bg = texture(seed, W, H)
    patch = 255 - texture(seed + 100, 40, 40)
    frames = []
    for t in range(N_FRAMES):
        f = bg.copy()
        x = int(round(20 + vx * t))
        f[y_center - 20:y_center + 20, x:x + 40] = patch
        frames.append(f)
    return frames


def static(seed: int) -> list[np.ndarray]:
    return [texture(seed, W, H)] * N_FRAMES


def build_clips() -> dict[str, list[np.ndarray]]:
    return {
        "panR_a": pan(1, 2.0, 0.0),        # camera pans right, content moves left
        "panR_b": pan(2, 3.0, 0.0),        # same direction, different texture & speed
        "panL": pan(3, -2.0, 0.0),
        "tiltDown": pan(4, 0.0, 2.0),
        "zoomIn_a": zoom(5, 0.006),
        "zoomIn_b": zoom(6, 0.008),
        "obj_top": moving_object(7, 30, 4.0),      # same subject motion, top of frame
        "obj_bottom": moving_object(8, 114, 4.0),  # same subject motion, bottom of frame
        "static": static(9),
    }


def compute(clips: dict[str, list[np.ndarray]]):
    names = list(clips)
    flows = {n: mean_flow_from_frames(clips[n], CFG)[0] for n in names}
    sim = cosine_matrix(np.stack([flows[n].reshape(-1) for n in names]))
    return names, flows, sim


_CACHE = {}


def _results():
    if not _CACHE:
        _CACHE["r"] = compute(build_clips())
    return _CACHE["r"]


def _s(a: str, b: str) -> float:
    names, _, sim = _results()
    return float(sim[names.index(a), names.index(b)])


def test_same_pan_direction_is_best_match():
    assert _s("panR_a", "panR_b") > 0.8
    names, _, sim = _results()
    row = sim[names.index("panR_a")].copy()
    row[names.index("panR_a")] = -np.inf
    assert names[int(row.argmax())] == "panR_b"


def test_opposite_pan_is_negative():
    assert _s("panR_a", "panL") < -0.8


def test_orthogonal_motion_is_near_zero():
    assert abs(_s("panR_a", "tiltDown")) < 0.3


def test_zoom_matches_zoom_better_than_pan():
    z = _s("zoomIn_a", "zoomIn_b")
    assert z > 0.5
    assert z > max(_s("zoomIn_a", "panR_a"), _s("zoomIn_a", "panL"), _s("zoomIn_a", "tiltDown"))


def test_static_clip_has_no_motion():
    _, flows, _ = _results()
    assert np.linalg.norm(flows["static"], axis=-1).mean() < 0.05


def test_known_limitation_position_sensitivity():
    """Documented h4 limitation (not a bug): identical subject motion in different
    screen regions scores low, because the representation is a per-pixel field."""
    assert _s("obj_top", "obj_bottom") < 0.5


def _save_visuals(out_dir: Path) -> None:
    from matchcut.visualize import flow_panel, similarity_heatmap

    names, flows, sim = _results()
    out_dir.mkdir(parents=True, exist_ok=True)
    similarity_heatmap(sim, names, out_dir / "synthetic_heatmap.png")
    clips = build_clips()
    tiles = []
    for n in names:
        frame = cv2.cvtColor(cv2.resize(clips[n][N_FRAMES // 2], (320, 180)), cv2.COLOR_GRAY2BGR)
        flow = flow_panel(flows[n], size=(320, 180), arrow_spacing=28, label=n)
        tiles.append(np.hstack([frame, flow]))
    cv2.imwrite(str(out_dir / "synthetic_flows.jpg"), np.vstack(tiles))


if __name__ == "__main__":
    names, _, sim = _results()
    print("cosine similarity (h4) on synthetic clips:")
    print(" " * 11 + "".join(f"{n[:9]:>10}" for n in names))
    for n, row in zip(names, sim):
        print(f"{n:>11}" + "".join(f"{x:>10.2f}" for x in row))
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError:
                failures += 1
                print(f"FAIL  {name}")
    _save_visuals(ROOT / "outputs" / "synthetic")
    print(f"visuals -> {ROOT / 'outputs' / 'synthetic'}")
    sys.exit(1 if failures else 0)

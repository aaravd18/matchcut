"""Synthetic check for the montage stages: cut scoring, ordering, cut placement.

Seven 4-second synthetic clips: two right pans, two left pans, two zooms and a
static shot, given in shuffled order. A good montage should put clips with the
same motion next to each other, and every segment must respect seg_min/seg_max.

Run:   python tests/test_sequence.py     (or python -m pytest tests/test_sequence.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matchcut.config import FlowConfig, SequenceConfig  # noqa: E402
from matchcut.motion import MotionRep, analyze_frames  # noqa: E402
from matchcut.sequence import choose_order, edge_matrix, path_score, place_segments  # noqa: E402
from matchcut.transitions import clip_features, transition_scores  # noqa: E402
from test_synthetic import texture  # noqa: E402

W, H, FPS, N = 256, 144, 30, 120          # 4 s clips
MARGIN = 320
FLOW = FlowConfig(flow_width=W, flow_height=H)
CFG = SequenceConfig()


def pan(seed, vx):
    canvas = texture(seed, W + 2 * MARGIN, H)
    return [cv2.warpAffine(canvas, np.float32([[1, 0, -(MARGIN + vx * t)], [0, 1, 0]]), (W, H)) for t in range(N)]


def zoom(seed, rate):
    canvas = texture(seed, W, H)
    return [cv2.warpAffine(canvas, cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + rate * t), (W, H),
                           borderMode=cv2.BORDER_REFLECT) for t in range(N)]


CLIPS = {   # deliberately shuffled
    "zoom_a": lambda: zoom(11, 0.004),
    "panR_a": lambda: pan(12, 2.0),
    "static": lambda: [texture(13, W, H)] * N,
    "panL_a": lambda: pan(14, -2.0),
    "zoom_b": lambda: zoom(15, 0.005),
    "panR_b": lambda: pan(16, 2.5),
    "panL_b": lambda: pan(17, -2.5),
}
PARTNER = {"panR_a": "panR_b", "panL_a": "panL_b", "zoom_a": "zoom_b"}

_CACHE = {}


def _setup():
    if not _CACHE:
        names, feats = list(CLIPS), []
        for name in names:
            frames = CLIPS[name]()
            mean, n_pairs, ts, ts_frame = analyze_frames(frames, FLOW)
            rep = MotionRep(name, mean, n_pairs, ts, ts_frame)
            feats.append(clip_features(rep, [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames], FPS, CFG))
        _CACHE.update(names=names, feats=feats, edges=edge_matrix(feats, CFG))
    return _CACHE["names"], _CACHE["feats"], _CACHE["edges"]


def test_cut_score_prefers_same_motion():
    names, feats, _ = _setup()
    f = dict(zip(names, feats))
    same = transition_scores(f["panR_a"], f["panR_b"], CFG)
    opposite = transition_scores(f["panR_a"], f["panL_a"], CFG)
    assert same["motion_valid"].any() and same["total"].max() > opposite["total"].max() + 0.5


def test_static_clip_has_no_motion_evidence():
    names, feats, _ = _setup()
    f = dict(zip(names, feats))
    assert not transition_scores(f["static"], f["panR_a"], CFG)["motion_valid"].any()


def test_order_puts_matching_motion_together():
    names, _, edges = _setup()
    order, _ = choose_order(edges, len(names))
    seq = [names[i] for i in order]
    for a, b in PARTNER.items():
        assert abs(seq.index(a) - seq.index(b)) == 1, f"{a} and {b} not adjacent in {seq}"


def test_max_clips_two_picks_a_matching_pair():
    names, _, edges = _setup()
    order, _ = choose_order(edges, 2)
    pair = {names[i] for i in order}
    assert any(pair == {a, b} for a, b in PARTNER.items()), pair


def test_segments_respect_length_limits():
    names, feats, edges = _setup()
    order, _ = choose_order(edges, len(names))
    for c, (ki, ko) in zip(order, place_segments(order, feats, CFG)):
        length = (feats[c].boundaries[ko] - feats[c].boundaries[ki]) / FPS
        assert CFG.seg_min - 1e-6 <= length <= CFG.seg_max + 1e-6


def test_local_search_finds_planted_path():
    """More than 8 clips uses greedy + local search instead of exhaustive search."""
    rng = np.random.default_rng(0)
    n = 12
    e = rng.uniform(0.0, 0.3, (n, n))
    planted = list(rng.permutation(n))
    for a, b in zip(planted, planted[1:]):
        e[a, b] = 1.0
    np.fill_diagonal(e, -np.inf)
    order, method = choose_order(e, n)
    assert method != "exhaustive"
    assert path_score(order, e) >= path_score(planted, e) - 1e-9


if __name__ == "__main__":
    names, _, edges = _setup()
    order, method = choose_order(edges, len(names))
    print(f"order ({method}): {' -> '.join(names[i] for i in order)}")
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}  {e}")
    sys.exit(1 if failures else 0)

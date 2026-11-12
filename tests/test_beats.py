"""Synthetic checks for beat analysis and beat-synchronised cut placement.

1. A generated 120 BPM track (kick on beat 1, hi-hat clicks on every beat, a
   chord change every bar) must give the right tempo, beats and downbeats.
2. With the synthetic clips from test_sequence.py, every cut placed by the
   beat DP must land exactly on a beat frame, each clip must last a whole
   number of beats, and the montage must end on a bar line — also at a tempo
   whose beat is not a whole number of frames (128 BPM).

Run:   python tests/test_beats.py      (or python -m pytest tests/test_beats.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from matchcut.audio import analyze_signal, beat_grid  # noqa: E402
from matchcut.config import MusicConfig, SequenceConfig  # noqa: E402
from matchcut.sequence import allowed_beat_counts, choose_order, place_segments_on_beats  # noqa: E402

SR = 22050
BPM = 120.0
FIRST_BEAT = 0.5          # seconds; beat 0 is a downbeat
N_BEATS = 64


def synth_track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signal, true beat times, true downbeat times)."""
    period = 60.0 / BPM
    beats = FIRST_BEAT + period * np.arange(N_BEATS)
    n = int((beats[-1] + 1.0) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    rng = np.random.default_rng(0)
    for i, b in enumerate(beats):
        s = int(b * SR)
        hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.linspace(0, 8, int(0.03 * SR)))
        y[s:s + len(hat)] += 0.3 * hat
        if i % 4 == 0:   # kick on beat 1
            k = np.arange(int(0.15 * SR)) / SR
            y[s:s + len(k)] += 0.9 * np.sin(2 * np.pi * 55 * k) * np.exp(-k * 25)
    roots = [220.0, 174.6, 261.6, 196.0]          # one chord per bar
    for bar in range(N_BEATS // 4):
        s, e = int(beats[bar * 4] * SR), int((beats[bar * 4] + 4 * period) * SR)
        f0 = roots[bar % 4]
        seg = t[s:e]
        y[s:e] += 0.08 * sum(np.sin(2 * np.pi * f0 * r * seg) for r in (1, 1.25, 1.5))
    return y, beats, beats[::4]


_CACHE = {}


def _analysis():
    if not _CACHE:
        y, beats, downs = synth_track()
        _CACHE.update(info=analyze_signal(y, SR, MusicConfig()), beats=beats, downs=downs)
    return _CACHE["info"], _CACHE["beats"], _CACHE["downs"]


def test_tempo():
    info, _, _ = _analysis()
    assert abs(info["tempo"] - BPM) / BPM < 0.01, info["tempo"]


def test_beats_land_on_true_beats():
    info, beats, _ = _analysis()
    det = np.array(info["beat_times"])
    err = np.abs(det[:, None] - beats[None, :]).min(1)
    assert len(det) >= N_BEATS - 4 and err.max() < 0.025, (len(det), err.max())


def test_downbeats():
    info, _, downs = _analysis()
    det = np.array(info["beat_times"])[np.array(info["is_downbeat"])]
    err = np.abs(det[:, None] - downs[None, :]).min(1)
    assert err.max() < 0.025, f"downbeats off by up to {err.max():.3f}s"


def _grid_beats(bpm: float, seconds: float = 120.0) -> dict:
    period = 60.0 / bpm
    bt = np.arange(0, seconds, period)
    return {"beat_times": bt.tolist(), "is_downbeat": (np.arange(len(bt)) % 4 == 0).tolist(),
            "beat_strength": [1.0] * len(bt), "beat_period": period, "downbeat_offset": 0}


def _place(bpm: float, music: MusicConfig):
    from test_sequence import _setup
    names, feats, edges = _setup()
    cfg = SequenceConfig()
    order, _ = choose_order(edges, len(names))
    grid = beat_grid(_grid_beats(bpm), 0.0, feats[0].fps)
    return order, feats, cfg, grid, place_segments_on_beats(order, feats, cfg, grid, music)


def _check_on_grid(bpm: float, music: MusicConfig):
    order, feats, cfg, grid, placement = _place(bpm, music)
    allowed = allowed_beat_counts(grid, cfg, music)
    assert placement[0]["m_start"] == 0
    for c, p in zip(order, placement):
        n_beats = p["m_end"] - p["m_start"]
        assert n_beats in allowed
        assert p["out_frame"] - p["in_frame"] == grid.frames[p["m_end"]] - grid.frames[p["m_start"]]
        assert 0 <= p["in_frame"] and p["out_frame"] <= feats[c].n_frames
    for a, b in zip(placement, placement[1:]):
        assert a["m_end"] == b["m_start"]          # contiguous: each cut is exactly one beat
    assert grid.is_down[placement[-1]["m_end"]]    # ends on a bar line
    return grid, placement


def test_cuts_on_beats_120bpm():
    _check_on_grid(120.0, MusicConfig())


def test_cuts_on_beats_128bpm_fractional_frames():
    grid, placement = _check_on_grid(128.0, MusicConfig())
    assert np.all(grid.frames == np.round(grid.times * 30).astype(int))


def test_bars_mode_cuts_only_on_downbeats():
    grid, placement = _check_on_grid(120.0, MusicConfig(cut_on="bars"))
    assert all(grid.is_down[p["m_start"]] for p in placement)


if __name__ == "__main__":
    info, _, _ = _analysis()
    print(f"synthetic track: {info['tempo']} BPM, {info['n_beats']} beats, downbeat scores {info['downbeat_scores']['combined']}")
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

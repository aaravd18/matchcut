"""Stage 4 — score cuts between specific moments of two clips (departs from the paper).

h4 compares whole-clip average motion. A real cut joins one moment of A (its
out-point) to one moment of B (its in-point), so here every candidate cut point
(every `cut_step` seconds) gets features describing what is on screen there:

  motion       mean pooled flow (grid_w x grid_h) over the `window` seconds
               before the cut (for an out-point) or after it (for an in-point)
  composition  "h2-style" frame match generalised to scenery: layout of
               brightness and of edges in the frame at the cut (32x18)
  color        Lab colour histogram of the frame at the cut

Score of cutting from A at out-point i to B at in-point j:

  motion_ij = cos(A.out_motion[i], B.in_motion[j]) * min(|a|,|b|)/max(|a|,|b|)
              (0 if either side moves less than `min_motion`: no motion evidence)
  comp_ij   = mean of ZNCC(brightness layouts) and ZNCC(edge layouts)   in [-1, 1]
  color_ij  = Bhattacharyya coefficient of colour histograms           in [0, 1]
  total_ij  = w_motion*motion + w_comp*comp + w_color*color

Outputs (outputs/transitions/): <clip_id>.npz features, edges.json/csv with the
best directed cut for every ordered pair, edges_heatmap.png.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from .config import OutputPaths, SequenceConfig
from .motion import MotionRep, load_reps
from .video import read_frames

FEAT_SIZE = (64, 36)      # frames are analysed at this size
LAYOUT_SIZE = (32, 18)    # brightness / edge layout grid
# Lab histogram bins (L, a, b). a/b use an odd count so neutral grey (a = b = 128
# in OpenCV's 8-bit Lab) is a bin centre; with an even count a bin edge sits
# exactly at neutral and near-grey frames (sky, foam) split arbitrarily.
COLOR_BINS = (8, 7, 7)


def _unit(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-length rows (all-constant rows become zeros)."""
    x = x - x.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 1e-8)


def frame_features(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(brightness layout, edge layout, sqrt colour histogram) of one frame."""
    small = cv2.resize(frame_bgr, FEAT_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    lum = cv2.resize(gray, LAYOUT_SIZE, interpolation=cv2.INTER_AREA).ravel()
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edge = cv2.resize(np.sqrt(gx * gx + gy * gy), LAYOUT_SIZE, interpolation=cv2.INTER_AREA).ravel()
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, list(COLOR_BINS), [0, 256] * 3).ravel()
    hist /= max(hist.sum(), 1e-8)
    return lum, edge, np.sqrt(hist).astype(np.float32)


@dataclass
class ClipFeatures:
    clip_id: str
    fps: float
    n_frames: int
    boundaries: np.ndarray   # (K,) candidate cut positions as frame indices: a cut at b shows ... b-1 | b ...
    out_ok: np.ndarray       # (K,) a full motion window fits before b
    in_ok: np.ndarray        # (K,) a full motion window fits after b
    out_motion: np.ndarray   # (K, D) mean pooled flow in the window before b
    in_motion: np.ndarray    # (K, D) mean pooled flow in the window after b
    out_mag: np.ndarray      # (K,) mean cell speed of out_motion (px/frame)
    in_mag: np.ndarray       # (K,)
    out_lum: np.ndarray      # (K, 576) features of frame b-1 (last frame shown before the cut)
    out_edge: np.ndarray
    out_color: np.ndarray
    in_lum: np.ndarray       # (K, 576) features of frame b (first frame shown after the cut)
    in_edge: np.ndarray
    in_color: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return self.boundaries / self.fps

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> "ClipFeatures":
        d = np.load(path)
        kw = {f.name: d[f.name] for f in fields(cls)}
        kw["clip_id"], kw["fps"], kw["n_frames"] = str(kw["clip_id"]), float(kw["fps"]), int(kw["n_frames"])
        return cls(**kw)


def _window_motion(rep: MotionRep, first: int, last: int) -> np.ndarray | None:
    """Mean pooled flow of time-series flows j->j+1 with first <= j and j+1 <= last."""
    sel = (rep.ts_frame >= first) & (rep.ts_frame + 1 <= last)
    if not sel.any():
        return None
    return rep.ts_flow[sel].mean(0)


def clip_features(rep: MotionRep, frames_bgr: list[np.ndarray], fps: float, cfg: SequenceConfig) -> ClipFeatures:
    """Features at every candidate cut point of one clip. frames_bgr: every frame (any size)."""
    n = len(frames_bgr)
    step = cfg.cut_step * fps
    bounds = sorted({int(round(k * step)) for k in range(int(n / step) + 1)} | {n})
    bounds = np.array([b for b in bounds if 0 <= b <= n], np.int32)
    win = int(round(cfg.window * fps))
    d = rep.ts_flow.shape[1] * rep.ts_flow.shape[2] * 2

    cols = {k: [] for k in ("out_ok", "in_ok", "out_motion", "in_motion", "out_mag", "in_mag")}
    for b in bounds:
        for side, lo, hi, fits in (("out", b - win, b - 1, b - win >= 0), ("in", b, b + win - 1, b + win <= n)):
            m = _window_motion(rep, lo, hi) if fits else None
            cols[f"{side}_ok"].append(m is not None)
            vec = np.zeros(d, np.float32) if m is None else m.ravel()
            cols[f"{side}_motion"].append(vec)
            cols[f"{side}_mag"].append(0.0 if m is None else float(np.linalg.norm(m, axis=-1).mean()))

    cache: dict[int, tuple] = {}

    def feats(i: int):
        i = min(max(i, 0), n - 1)
        if i not in cache:
            cache[i] = frame_features(frames_bgr[i])
        return cache[i]

    out_f = [feats(b - 1) for b in bounds]
    in_f = [feats(b) for b in bounds]
    return ClipFeatures(
        clip_id=rep.clip_id, fps=float(fps), n_frames=n, boundaries=bounds,
        out_ok=np.array(cols["out_ok"]), in_ok=np.array(cols["in_ok"]),
        out_motion=np.array(cols["out_motion"], np.float32), in_motion=np.array(cols["in_motion"], np.float32),
        out_mag=np.array(cols["out_mag"], np.float32), in_mag=np.array(cols["in_mag"], np.float32),
        out_lum=_unit(np.array([f[0] for f in out_f], np.float32)),
        out_edge=_unit(np.array([f[1] for f in out_f], np.float32)),
        out_color=np.array([f[2] for f in out_f], np.float32),
        in_lum=_unit(np.array([f[0] for f in in_f], np.float32)),
        in_edge=_unit(np.array([f[1] for f in in_f], np.float32)),
        in_color=np.array([f[2] for f in in_f], np.float32),
    )


def transition_scores(a: ClipFeatures, b: ClipFeatures, cfg: SequenceConfig) -> dict[str, np.ndarray]:
    """All score terms for cutting from A (rows: out-points) to B (cols: in-points).

    Infeasible cuts (no room for a motion window) get total = -inf.
    """
    na = np.linalg.norm(a.out_motion, axis=1, keepdims=True)
    nb = np.linalg.norm(b.in_motion, axis=1, keepdims=True)
    ua = np.divide(a.out_motion, na, out=np.zeros_like(a.out_motion), where=na > 1e-12)
    ub = np.divide(b.in_motion, nb, out=np.zeros_like(b.in_motion), where=nb > 1e-12)
    cos = ua @ ub.T
    ma, mb = a.out_mag[:, None], b.in_mag[None, :]
    speed = np.minimum(ma, mb) / np.maximum(np.maximum(ma, mb), 1e-12)
    valid = (ma >= cfg.min_motion) & (mb >= cfg.min_motion)
    motion = np.where(valid, cos * speed, 0.0)
    comp = 0.5 * (a.out_lum @ b.in_lum.T + a.out_edge @ b.in_edge.T)
    color = np.clip(a.out_color @ b.in_color.T, 0.0, 1.0)
    total = cfg.w_motion * motion + cfg.w_comp * comp + cfg.w_color * color
    feasible = a.out_ok[:, None] & b.in_ok[None, :]
    return {
        "total": np.where(feasible, total, -np.inf), "motion": motion, "cos": cos, "speed": speed,
        "motion_valid": valid & feasible, "composition": comp, "color": color,
    }


def best_cut(a: ClipFeatures, b: ClipFeatures, cfg: SequenceConfig,
             min_before: float = 0.0, min_after: float = 0.0) -> tuple[int, int, float]:
    """Best (out-point index in A, in-point index in B, total score), leaving at least
    `min_before` seconds of A before the cut and `min_after` seconds of B after it."""
    total = transition_scores(a, b, cfg)["total"].copy()
    total[a.times < min_before - 1e-6, :] = -np.inf
    total[:, b.times > b.duration - min_after + 1e-6] = -np.inf
    i, j = np.unravel_index(int(np.argmax(total)), total.shape)
    return int(i), int(j), float(total[i, j])


def breakdown(a: ClipFeatures, b: ClipFeatures, i: int, j: int, cfg: SequenceConfig) -> dict:
    """Score terms of one specific cut, as plain floats (for EDLs and reports)."""
    s = transition_scores(a, b, cfg)
    return {
        "total": round(float(s["total"][i, j]), 4),
        "motion": round(float(s["motion"][i, j]), 4),
        "motion_cos": round(float(s["cos"][i, j]), 4),
        "speed_ratio": round(float(s["speed"][i, j]), 4),
        "motion_valid": bool(s["motion_valid"][i, j]),
        "composition": round(float(s["composition"][i, j]), 4),
        "color": round(float(s["color"][i, j]), 4),
        "out_speed": round(float(a.out_mag[i]), 4),
        "in_speed": round(float(b.in_mag[j]), 4),
    }


def run_transitions(manifest: dict, paths: OutputPaths, cfg: SequenceConfig) -> list[ClipFeatures]:
    from .similarity import short_names
    from .visualize import similarity_heatmap

    out = paths.transitions
    out.mkdir(parents=True, exist_ok=True)
    reps = {r.clip_id: r for r in load_reps(manifest, paths)}
    feats = []
    for i, clip in enumerate(manifest["clips"], 1):
        frames = list(read_frames(Path(clip["path"]), size=FEAT_SIZE))
        f = clip_features(reps[clip["id"]], frames, clip["fps"], cfg)
        f.save(out / f"{f.clip_id}.npz")
        moving = float(np.mean(np.r_[f.out_mag[f.out_ok], f.in_mag[f.in_ok]] >= cfg.min_motion))
        print(f"[transitions] ({i}/{len(manifest['clips'])}) {f.clip_id}: "
              f"{len(f.boundaries)} cut points, {moving:.0%} of motion windows above min_motion")
        feats.append(f)

    ids = [f.clip_id for f in feats]
    names = short_names(ids)
    n = len(feats)
    edges = np.full((n, n), np.nan)
    rows = []
    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            i, j, s = best_cut(feats[x], feats[y], cfg, cfg.seg_min, cfg.seg_min)
            if not np.isfinite(s):
                continue
            edges[x, y] = s
            rows.append({"from": ids[x], "to": ids[y], "out_s": round(float(feats[x].times[i]), 3),
                         "in_s": round(float(feats[y].times[j]), 3),
                         **breakdown(feats[x], feats[y], i, j, cfg)})
    rows.sort(key=lambda r: r["total"], reverse=True)
    (out / "edges.json").write_text(json.dumps({"config": asdict(cfg), "clips": ids, "edges": rows}, indent=2))
    with open(out / "edges.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    similarity_heatmap(edges, names, out / "edges_heatmap.png", title="Best cut score, row → column",
                       cbar_label="transition score (motion + composition + colour)",
                       xlabel="cut into (B)", ylabel="cut from (A)")
    print(f"[transitions] best directed cut for {len(rows)} ordered pairs -> {out}")
    return feats


def load_features(manifest: dict, paths: OutputPaths) -> list[ClipFeatures]:
    feats = []
    for clip in manifest["clips"]:
        p = paths.transitions / f"{clip['id']}.npz"
        if not p.exists():
            raise SystemExit(f"{p} not found — run the 'transitions' stage first")
        feats.append(ClipFeatures.load(p))
    return feats

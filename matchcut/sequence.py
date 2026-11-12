"""Stage 5 — choose the clip order and the exact cut points for the montage.

1. Edge weight E[a, b] = best transition score for cutting a -> b (any cut
   points that leave each clip at least seg_min seconds).
2. Order: shortlist the paths with the highest total E, then keep the one whose
   exact cut placement (step 3) scores best.
   - max_clips = None: use every usable clip; max_clips = N: the best-connected N
     clips, i.e. the N-clip path with the highest total score.
   - <= 8 clips: exhaustive search over all orderings (exact).
   - more: greedy from every start clip, then local search (reverse a run,
     move a clip, swap two clips, swap in an unused clip) until nothing improves.
   - or a fixed order given by the user.
3. Cut points: dynamic programming over the chosen order picks each clip's in-
   and out-point to maximise the sum of transition scores, with every segment
   seg_min..seg_max long and a small penalty for straying from seg_target.
   - with music (outputs/beats/beats.json): every cut lands exactly on a beat,
     each clip lasts a whole number of beats, cuts on downbeats/strong beats
     earn a bonus, and the montage ends on a bar line (place_segments_on_beats).
   - without music: free timing on the cut_step grid (place_segments).

Output: outputs/montage/edl.json and edl.csv (edit decision list).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from itertools import permutations
from typing import Optional, Sequence

import numpy as np

from .audio import BeatGrid, beat_grid
from .config import MusicConfig, OutputPaths, SequenceConfig
from .similarity import short_names
from .transitions import ClipFeatures, best_cut, breakdown, load_features, transition_scores

EXHAUSTIVE_MAX = 8


def edge_matrix(feats: Sequence[ClipFeatures], cfg: SequenceConfig) -> np.ndarray:
    n = len(feats)
    e = np.full((n, n), -np.inf)
    for a in range(n):
        for b in range(n):
            if a != b:
                e[a, b] = best_cut(feats[a], feats[b], cfg, cfg.seg_min, cfg.seg_min)[2]
    return e


def path_score(path: Sequence[int], e: np.ndarray) -> float:
    return float(sum(e[path[i], path[i + 1]] for i in range(len(path) - 1)))


def _local_search(path: list[int], e: np.ndarray, n: int) -> list[int]:
    best, best_s = list(path), path_score(path, e)
    improved = True
    while improved:
        improved = False
        k = len(best)
        unused = [u for u in range(n) if u not in best]
        candidates = []
        for i in range(k):
            for j in range(i + 1, k):
                candidates.append(best[:i] + best[i:j + 1][::-1] + best[j + 1:])     # reverse a run
                s = list(best); s[i], s[j] = s[j], s[i]; candidates.append(s)       # swap two clips
            rest = best[:i] + best[i + 1:]
            for p in range(k):
                if p != i:
                    candidates.append(rest[:p] + [best[i]] + rest[p:])              # move a clip
            for u in unused:
                candidates.append(best[:i] + [u] + best[i + 1:])                     # swap in unused clip
        for c in candidates:
            s = path_score(c, e)
            if s > best_s + 1e-9:
                best, best_s, improved = c, s, True
                break
    return best


def candidate_orders(e: np.ndarray, k: int, shortlist: int = 50) -> tuple[list[list[int]], str]:
    """Highest-scoring paths (by edge weights) visiting k of the n clips, best first."""
    n = len(e)
    w = np.where(np.isfinite(e), e, -1e6)
    if k <= 1:
        return [[int(np.argmax(w.max(axis=1)))]], "single"
    if n <= EXHAUSTIVE_MAX:
        scored = sorted(permutations(range(n), k), key=lambda p: path_score(p, w), reverse=True)
        return [list(p) for p in scored[:shortlist]], "exhaustive"
    found = {}
    for start in range(n):
        path, used = [start], {start}
        while len(path) < k:
            nxt = max((j for j in range(n) if j not in used), key=lambda j: w[path[-1], j])
            path.append(nxt)
            used.add(nxt)
        path = _local_search(path, w, n)
        found[tuple(path)] = path_score(path, w)
    ranked = sorted(found, key=found.get, reverse=True)
    return [list(p) for p in ranked], "greedy + local search"


def choose_order(e: np.ndarray, k: int) -> tuple[list[int], str]:
    """Highest-scoring path by edge weights alone. Returns (path, method)."""
    n = len(e)
    w = np.where(np.isfinite(e), e, -1e6)
    if k <= 1:
        return [int(np.argmax(w.max(axis=1)))], "single"
    if n <= EXHAUSTIVE_MAX:
        best = max(permutations(range(n), k), key=lambda p: path_score(p, w))
        return list(best), "exhaustive"
    best, best_s = None, -np.inf
    for start in range(n):
        path, used = [start], {start}
        while len(path) < k:
            nxt = max((j for j in range(n) if j not in used), key=lambda j: w[path[-1], j])
            path.append(nxt)
            used.add(nxt)
        path = _local_search(path, w, n)
        s = path_score(path, w)
        if s > best_s:
            best, best_s = path, s
    return best, "greedy + local search"


def place_segments(order: Sequence[int], feats: Sequence[ClipFeatures], cfg: SequenceConfig,
                   return_value: bool = False):
    """DP over the fixed order. Returns [(in_index, out_index)] into each clip's boundaries
    (and the DP objective: sum of cut scores minus length penalties, if return_value)."""
    fps = feats[order[0]].fps
    smin, smax = cfg.seg_min * fps - 1e-6, cfg.seg_max * fps + 1e-6
    tgt = cfg.seg_target * fps

    def seg_terms(f: ClipFeatures):
        b = f.boundaries.astype(float)
        length = b[None, :] - b[:, None]                        # (in, out) in frames
        ok = (length >= smin) & (length <= smax)
        return ok, -cfg.length_weight * np.abs(length - tgt) / tgt

    back = []
    v = None
    for pos, c in enumerate(order):
        f = feats[c]
        ok, pen = seg_terms(f)
        if pos == 0:
            u = np.zeros(len(f.boundaries))
            prev_arg = None
        else:
            m = v[:, None] + transition_scores(feats[order[pos - 1]], f, cfg)["total"]   # (prev_out, in)
            u, prev_arg = m.max(0), m.argmax(0)
        seg = np.where(ok, u[:, None] + pen, -np.inf)            # (in, out)
        v, in_arg = seg.max(0), seg.argmax(0)
        back.append((in_arg, prev_arg))
    if not np.isfinite(v).any():
        if return_value:
            return [], -np.inf
        raise ValueError("no feasible cut placement; try a wider --seg-min/--seg-max range")

    result = []
    out_k = int(np.argmax(v))
    best_value = float(v[out_k])
    for pos in range(len(order) - 1, -1, -1):
        in_arg, prev_arg = back[pos]
        in_k = int(in_arg[out_k])
        result.append((in_k, out_k))
        if prev_arg is not None:
            out_k = int(prev_arg[in_k])
    return (result[::-1], best_value) if return_value else result[::-1]


def allowed_beat_counts(grid: BeatGrid, cfg: SequenceConfig, music: MusicConfig) -> list[int]:
    """Whole numbers of beats a clip may last (seg_min..seg_max; multiples of 4 for --cut-on bars)."""
    nmin = max(1, int(np.ceil(cfg.seg_min / grid.period - 1e-6)))
    nmax = max(nmin, int(np.floor(cfg.seg_max / grid.period + 1e-6)))
    if music.cut_on == "bars":
        return [n for n in range(4, max(nmax, 4) + 1, 4) if n >= nmin] or [4]
    return list(range(nmin, nmax + 1))


def _nearest(sorted_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the nearest element of sorted_vals for each x."""
    i = np.clip(np.searchsorted(sorted_vals, x), 1, len(sorted_vals) - 1)
    return np.where(np.abs(sorted_vals[i - 1] - x) <= np.abs(sorted_vals[i] - x), i - 1, i)


def place_segments_on_beats(order: Sequence[int], feats: Sequence[ClipFeatures], cfg: SequenceConfig,
                            grid: BeatGrid, music: MusicConfig, return_value: bool = False):
    """DP placing every cut exactly on a beat of the soundtrack.

    State after clip i: (m, k) = the montage beat it ends on, and the candidate
    out-point in the clip whose features describe its last frames. Clip i
    starts at beat m_prev with in-point j and lasts n beats (an allowed count),
    i.e. L = grid.frames[m_prev + n] - grid.frames[m_prev] frames; its exact out
    frame is in_frame + L (features are read at the nearest 0.1 s candidate).

    Objective = sum of transition scores + w_beat * reward of every cut beat
                (by position in the bar, see audio.METRIC_WEIGHT) - length penalties.
    The last clip must end on a downbeat (bar line) when possible.

    Returns a list of per-clip dicts {k_in, k_out, in_frame, out_frame, m_start, m_end}.
    """
    fps = feats[order[0]].fps
    ns = allowed_beat_counts(grid, cfg, music)
    M = len(grid.frames)

    def pen(L):
        return cfg.length_weight * abs(L / fps - cfg.seg_target) / cfg.seg_target

    V = None
    back = []   # per position: (j_in, m_prev, k_prev) arrays of shape (M, K)
    for pos, c in enumerate(order):
        f = feats[c]
        B = f.boundaries.astype(np.int64)
        K = len(B)
        if pos == 0:
            U = np.full((M, K), -np.inf)
            U[0] = 0.0
            argU = np.full((M, K), -1)
            rows = [0]
        else:
            T = transition_scores(feats[order[pos - 1]], f, cfg)["total"]
            rows = [m for m in range(M) if np.isfinite(V[m]).any()]
            U = np.full((M, K), -np.inf)
            argU = np.full((M, K), -1)
            for m in rows:
                tmp = V[m][:, None] + T
                U[m], argU[m] = tmp.max(0), tmp.argmax(0)

        newV = np.full((M, K), -np.inf)
        bj, bm, bk = (np.full((M, K), -1) for _ in range(3))
        for m in rows:
            for n in ns:
                m2 = m + n
                if m2 >= M:
                    continue
                L = int(grid.frames[m2] - grid.frames[m])
                vals = U[m] - pen(L) + music.w_beat * grid.reward[m2]
                ok = np.isfinite(vals) & (B + L <= f.n_frames)
                idx = np.nonzero(ok)[0]
                if not len(idx):
                    continue
                idx = idx[np.argsort(-vals[idx])]
                kout = _nearest(B, B[idx] + L)
                kout, first = np.unique(kout, return_index=True)   # best j per out-candidate
                idx = idx[first]
                better = vals[idx] > newV[m2, kout]
                kout, idx = kout[better], idx[better]
                newV[m2, kout] = vals[idx]
                bj[m2, kout], bm[m2, kout], bk[m2, kout] = idx, m, argU[m, idx]
        V = newV
        back.append((bj, bm, bk, B))

    finite = np.isfinite(V)
    if not finite.any():
        if return_value:
            return [], -np.inf
        raise ValueError("clips cannot be placed on the beat grid; try --max-clips, a later "
                         "--music-start, or wider --seg-min/--seg-max")
    ends = V.copy()
    if (finite & grid.is_down[:, None]).any():
        ends[~grid.is_down] = -np.inf                       # end on a bar line
    m, k = np.unravel_index(int(np.argmax(ends)), ends.shape)
    value = float(ends[m, k])

    placement = []
    for pos in range(len(order) - 1, -1, -1):
        bj, bm, bk, B = back[pos]
        j, mp, kp = int(bj[m, k]), int(bm[m, k]), int(bk[m, k])
        L = int(grid.frames[m] - grid.frames[mp])
        placement.append({"k_in": j, "k_out": int(k), "in_frame": int(B[j]), "out_frame": int(B[j] + L),
                          "m_start": mp, "m_end": int(m)})
        m, k = mp, kp
    placement.reverse()
    return (placement, value) if return_value else placement


def place_free(order, feats, cfg, return_value=False):
    """place_segments() in the same per-clip dict format as place_segments_on_beats()."""
    res = place_segments(order, feats, cfg, return_value=True)
    idx, value = res
    placement = [{"k_in": ki, "k_out": ko, "in_frame": int(feats[c].boundaries[ki]),
                  "out_frame": int(feats[c].boundaries[ko]), "m_start": None, "m_end": None}
                 for c, (ki, ko) in zip(order, idx)]
    if not placement and not return_value:
        raise ValueError("no feasible cut placement; try a wider --seg-min/--seg-max range")
    return (placement, value) if return_value else placement


def refine_order(candidates: Sequence[Sequence[int]], place) -> list[int]:
    """Pick the candidate order whose exact cut placement (DP) scores best.

    Edge weights score each cut on its own, but a middle clip's in- and
    out-point must be seg_min..seg_max (or a whole number of beats) apart, so
    the best order by edge weights is not always the best once cuts are placed
    jointly. `place(order)` returns (placement, value).
    """
    best, best_v = list(candidates[0]), -np.inf
    for order in candidates:
        _, value = place(order)
        if value > best_v + 1e-9:
            best, best_v = list(order), value
    return best


def _resolve_order(names: Sequence[str], ids: Sequence[str], short: Sequence[str]) -> list[int]:
    lookup = {**{s: i for i, s in enumerate(short)}, **{c: i for i, c in enumerate(ids)}}
    missing = [x for x in names if x not in lookup]
    if missing:
        raise SystemExit(f"--order: unknown clip(s) {missing}; use ids or short names {list(short)}")
    order = [lookup[x] for x in names]
    if len(set(order)) != len(order):
        raise SystemExit("--order lists a clip twice")
    return order


def build_edl(manifest: dict, feats: list[ClipFeatures], cfg: SequenceConfig,
              beats: Optional[dict] = None, music: Optional[MusicConfig] = None) -> dict:
    """Edit decision list. With `beats` (beats.json) every cut lands on a beat."""
    clips = {c["id"]: c for c in manifest["clips"]}
    ids_all = [f.clip_id for f in feats]
    usable = [i for i, f in enumerate(feats) if f.duration >= cfg.seg_min]
    too_short = [ids_all[i] for i in range(len(feats)) if i not in usable]
    sub = [feats[i] for i in usable]
    ids = [f.clip_id for f in sub]
    short = short_names(ids)
    fps = sub[0].fps

    grid = None
    if beats is not None:
        music = music or MusicConfig()
        grid = beat_grid(beats, music.start, fps)

        def place(order):
            return place_segments_on_beats(order, sub, cfg, grid, music, return_value=True)
    else:
        def place(order):
            return place_free(order, sub, cfg, return_value=True)

    e = edge_matrix(sub, cfg)
    if cfg.order:
        order, method = _resolve_order(cfg.order, ids, short), "manual"
    else:
        k = len(sub) if cfg.max_clips is None else max(1, min(cfg.max_clips, len(sub)))
        candidates, method = candidate_orders(e, k, shortlist=50 if grid is None else 20)
        order = refine_order(candidates, place)
        method += f", best of {len(candidates)} by exact cut placement"
    placement, _ = place(order)
    if not placement:
        raise SystemExit("no feasible placement for this order; try --max-clips, --seg-min/--seg-max, or --music-start")

    segments, cuts = [], []
    for pos, (c, p) in enumerate(zip(order, placement)):
        f = sub[c]
        start = (grid.frames[p["m_start"]] if grid is not None
                 else (segments[-1]["montage_start_frame"] + segments[-1]["frames"] if segments else 0))
        n_fr = p["out_frame"] - p["in_frame"]
        seg = {
            "index": pos + 1, "clip_id": f.clip_id, "short_name": short[c], "path": clips[f.clip_id]["path"],
            "in_frame": p["in_frame"], "out_frame": p["out_frame"], "frames": n_fr,
            "in_s": round(p["in_frame"] / fps, 3), "out_s": round(p["out_frame"] / fps, 3),
            "duration": round(n_fr / fps, 3), "montage_start_frame": int(start),
            "montage_start": round(int(start) / fps, 3),
        }
        if grid is not None:
            seg["beats"] = p["m_end"] - p["m_start"]
        segments.append(seg)
        if pos > 0:
            pc, pp = order[pos - 1], placement[pos - 1]
            cut = {
                "index": pos, "from": sub[pc].clip_id, "to": f.clip_id,
                "from_short": short[pc], "to_short": short[c],
                "montage_time": seg["montage_start"], "out_frame": pp["out_frame"], "in_frame": p["in_frame"],
                "out_k": pp["k_out"], "in_k": p["k_in"],
                **breakdown(sub[pc], f, pp["k_out"], p["k_in"], cfg),
            }
            if grid is not None:
                m = p["m_start"]
                cut.update({"beat": int(m), "downbeat": bool(grid.is_down[m]),
                            "beat_reward": round(float(grid.reward[m]), 3),
                            "music_time": round(grid.music_start + float(grid.times[m]), 3)})
            cuts.append(cut)

    total_frames = segments[-1]["montage_start_frame"] + segments[-1]["frames"]
    edl = {
        "config": asdict(cfg), "order_method": method,
        "n_available": len(feats), "n_used": len(order),
        "unused": [ids[i] for i in range(len(sub)) if i not in order],
        "too_short": too_short,
        "path_score": round(sum(c["total"] for c in cuts), 4),
        "mean_cut_score": round(float(np.mean([c["total"] for c in cuts])), 4) if cuts else None,
        "fps": fps, "total_frames": int(total_frames), "total_duration": round(total_frames / fps, 3),
        "segments": segments, "cuts": cuts, "music": None,
    }
    if grid is not None:
        m_end = placement[-1]["m_end"]
        edl["music"] = {
            "path": beats["path"], "tempo": beats["tempo"], "config": asdict(music),
            "allowed_beats": allowed_beat_counts(grid, cfg, music),
            "start": round(grid.music_start, 3), "end": round(grid.music_start + float(grid.times[m_end]), 3),
            "ends_on_downbeat": bool(grid.is_down[m_end]),
            "cuts_on_downbeats": int(sum(c["downbeat"] for c in cuts)),
            # beats inside the montage (montage time), for click tracks and the report timeline
            "beat_times": [round(float(t), 4) for t in grid.times[: m_end + 1]],
            "is_downbeat": [bool(d) for d in grid.is_down[: m_end + 1]],
        }
    return edl


def run_sequence(manifest: dict, paths: OutputPaths, cfg: SequenceConfig,
                 music: Optional[MusicConfig] = None) -> dict:
    from .audio import load_beats

    feats = load_features(manifest, paths)
    music = music or MusicConfig()
    beats = load_beats(paths) if music.enabled else None
    edl = build_edl(manifest, feats, cfg, beats, music)
    paths.montage.mkdir(parents=True, exist_ok=True)
    paths.edl.write_text(json.dumps(edl, indent=2))
    with open(paths.montage / "edl.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "clip_id", "in_s", "out_s", "duration", "montage_start",
                    "cut_in_total", "cut_in_motion", "cut_in_composition", "cut_in_color"])
        cut_by_to = {c["index"] + 1: c for c in edl["cuts"]}
        for s in edl["segments"]:
            c = cut_by_to.get(s["index"], {})
            w.writerow([s["index"], s["clip_id"], s["in_s"], s["out_s"], s["duration"], s["montage_start"],
                        c.get("total", ""), c.get("motion", ""), c.get("composition", ""), c.get("color", "")])

    print(f"[sequence] {edl['n_used']}/{edl['n_available']} clips ({edl['order_method']}), "
          f"{edl['total_duration']:.1f}s, mean cut score {edl['mean_cut_score']}")
    mu = edl["music"]
    if mu:
        print(f"    music {mu['start']:.2f}–{mu['end']:.2f}s at {mu['tempo']:.1f} BPM; clip lengths {mu['allowed_beats']} beats; "
              f"{mu['cuts_on_downbeats']}/{len(edl['cuts'])} cuts on downbeats; ends on a bar line: {mu['ends_on_downbeat']}")
    for s in edl["segments"]:
        beats_txt = f", {s['beats']} beats" if "beats" in s else ""
        print(f"    {s['index']:>2}. {s['short_name']:<12} {s['in_s']:6.2f}–{s['out_s']:6.2f}s "
              f"({s['duration']:.2f}s{beats_txt})")
    for c in edl["cuts"]:
        flag = "" if c["motion_valid"] else "  (no motion evidence)"
        beat_txt = (f"  @beat {c['beat']}{' (downbeat)' if c['downbeat'] else ''}" if "beat" in c else "")
        print(f"    cut {c['index']}: {c['from_short']} → {c['to_short']}  total {c['total']:+.3f}  "
              f"motion {c['motion']:+.3f}  comp {c['composition']:+.3f}  color {c['color']:.3f}{beat_txt}{flag}")
    if edl["unused"]:
        print(f"    not used (max_clips): {', '.join(edl['unused'])}")
    if edl["too_short"]:
        print(f"    too short for a {cfg.seg_min}s segment: {', '.join(edl['too_short'])}")
    return edl


def load_edl(paths: OutputPaths) -> Optional[dict]:
    return json.loads(paths.edl.read_text()) if paths.edl.exists() else None

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
   and out-point (multiples of cut_step) to maximise the sum of transition
   scores, with every segment seg_min..seg_max long and a small penalty for
   straying from seg_target.

Output: outputs/montage/edl.json and edl.csv (edit decision list).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from itertools import permutations
from typing import Optional, Sequence

import numpy as np

from .config import OutputPaths, SequenceConfig
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


def refine_order(candidates: Sequence[Sequence[int]], feats: Sequence[ClipFeatures], cfg: SequenceConfig
                 ) -> list[int]:
    """Pick the candidate order whose exact cut placement (DP) scores best.

    Edge weights score each cut on its own, but a middle clip's in- and
    out-point must be seg_min..seg_max apart, so the best order by edge weights
    is not always the best once cuts are placed jointly.
    """
    best, best_v = list(candidates[0]), -np.inf
    for order in candidates:
        _, value = place_segments(order, feats, cfg, return_value=True)
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


def build_edl(manifest: dict, feats: list[ClipFeatures], cfg: SequenceConfig) -> dict:
    clips = {c["id"]: c for c in manifest["clips"]}
    ids_all = [f.clip_id for f in feats]
    min_len = cfg.seg_min
    usable = [i for i, f in enumerate(feats) if f.duration >= min_len]
    too_short = [ids_all[i] for i in range(len(feats)) if i not in usable]
    sub = [feats[i] for i in usable]
    ids = [f.clip_id for f in sub]
    short = short_names(ids)

    e = edge_matrix(sub, cfg)
    if cfg.order:
        order, method = _resolve_order(cfg.order, ids, short), "manual"
    else:
        k = len(sub) if cfg.max_clips is None else max(1, min(cfg.max_clips, len(sub)))
        candidates, method = candidate_orders(e, k)
        order = refine_order(candidates, sub, cfg)
        method += f", best of {len(candidates)} by exact cut placement"
    cuts_idx = place_segments(order, sub, cfg)

    segments, cuts, t = [], [], 0.0
    for pos, (c, (ki, ko)) in enumerate(zip(order, cuts_idx)):
        f = sub[c]
        b_in, b_out = int(f.boundaries[ki]), int(f.boundaries[ko])
        dur = (b_out - b_in) / f.fps
        segments.append({
            "index": pos + 1, "clip_id": f.clip_id, "short_name": short[c], "path": clips[f.clip_id]["path"],
            "in_frame": b_in, "out_frame": b_out, "in_s": round(b_in / f.fps, 3), "out_s": round(b_out / f.fps, 3),
            "duration": round(dur, 3), "montage_start": round(t, 3),
        })
        t += dur
        if pos > 0:
            pc = order[pos - 1]
            pko = cuts_idx[pos - 1][1]
            cuts.append({
                "index": pos, "from": sub[pc].clip_id, "to": f.clip_id,
                "from_short": short[pc], "to_short": short[c],
                "montage_time": round(segments[-1]["montage_start"], 3),
                "out_frame": int(sub[pc].boundaries[pko]), "in_frame": b_in,
                "out_k": pko, "in_k": ki,
                **breakdown(sub[pc], f, pko, ki, cfg),
            })
    return {
        "config": asdict(cfg), "order_method": method,
        "n_available": len(feats), "n_used": len(order),
        "unused": [ids[i] for i in range(len(sub)) if i not in order],
        "too_short": too_short,
        "path_score": round(sum(c["total"] for c in cuts), 4),
        "mean_cut_score": round(float(np.mean([c["total"] for c in cuts])), 4) if cuts else None,
        "total_duration": round(t, 3), "segments": segments, "cuts": cuts,
    }


def run_sequence(manifest: dict, paths: OutputPaths, cfg: SequenceConfig) -> dict:
    feats = load_features(manifest, paths)
    edl = build_edl(manifest, feats, cfg)
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
    for s in edl["segments"]:
        print(f"    {s['index']:>2}. {s['short_name']:<12} {s['in_s']:6.2f}–{s['out_s']:6.2f}s ({s['duration']:.2f}s)")
    for c in edl["cuts"]:
        flag = "" if c["motion_valid"] else "  (no motion evidence)"
        print(f"    cut {c['index']}: {c['from_short']} → {c['to_short']}  total {c['total']:+.3f}  "
              f"motion {c['motion']:+.3f}  comp {c['composition']:+.3f}  color {c['color']:.3f}{flag}")
    if edl["unused"]:
        print(f"    not used (max_clips): {', '.join(edl['unused'])}")
    if edl["too_short"]:
        print(f"    too short for a {cfg.seg_min}s segment: {', '.join(edl['too_short'])}")
    return edl


def load_edl(paths: OutputPaths) -> Optional[dict]:
    return json.loads(paths.edl.read_text()) if paths.edl.exists() else None

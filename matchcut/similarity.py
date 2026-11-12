"""Stage 3 — pairwise cosine similarity of h4 representations and pair ranking.

Outputs (outputs/similarity/):
  matrix.npy, matrix.csv    N x N cosine similarity (diagonal = 1)
  heatmap.png               annotated matrix
  ranked_pairs.csv / .json  every unordered pair i<j, best first
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import OutputPaths
from .motion import MotionRep, load_reps


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows. An all-zero row (no motion at all) gets 0 against everything."""
    v = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    unit = np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-12)
    return np.clip(unit @ unit.T, -1.0, 1.0)


def rank_pairs(sim: np.ndarray, reps: Sequence[MotionRep], min_motion: float = 0.0) -> list[dict]:
    """Unordered pairs (i < j) of clips that both move, sorted by similarity, highest first.

    Minimum-motion filter (our addition): a clip whose mean flow magnitude is
    below `min_motion` has a noise-dominated direction, so its cosine scores
    are meaningless; pairs involving it are left out of the ranking.
    """
    pairs = []
    n = len(reps)
    moving = [r.mean_magnitude >= min_motion for r in reps]
    for i in range(n):
        for j in range(i + 1, n):
            if not (moving[i] and moving[j]):
                continue
            pairs.append({
                "clip_a": reps[i].clip_id,
                "clip_b": reps[j].clip_id,
                "score": round(float(sim[i, j]), 6),
                "magnitude_a": round(reps[i].mean_magnitude, 4),
                "magnitude_b": round(reps[j].mean_magnitude, 4),
            })
    pairs.sort(key=lambda p: p["score"], reverse=True)
    for rank, p in enumerate(pairs, 1):
        p["rank"] = rank
    return [{"rank": p.pop("rank"), **p} for p in pairs]


def run_similarity(manifest: dict, paths: OutputPaths, min_motion: float = 0.0) -> tuple[np.ndarray, list[dict]]:
    from .visualize import similarity_heatmap

    reps = load_reps(manifest, paths)
    sim = cosine_matrix(np.stack([r.vector for r in reps]))
    ranked = rank_pairs(sim, reps, min_motion)
    ids = [r.clip_id for r in reps]
    excluded = [{"clip_id": r.clip_id, "mean_magnitude": round(r.mean_magnitude, 4)}
                for r in reps if r.mean_magnitude < min_motion]

    out = paths.similarity
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "matrix.npy", sim)
    with open(out / "matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + ids)
        for cid, row in zip(ids, sim):
            writer.writerow([cid] + [f"{x:.4f}" for x in row])
    with open(out / "ranked_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked[0].keys()) if ranked else ["rank"])
        writer.writeheader()
        writer.writerows(ranked)
    paths.ranked_pairs.write_text(json.dumps(
        {"clips": ids, "min_motion": min_motion, "excluded": excluded, "pairs": ranked}, indent=2))
    static = {e["clip_id"] for e in excluded}
    labels = [s + (" (static)" if cid in static else "") for cid, s in zip(ids, short_names(ids))]
    similarity_heatmap(sim, labels, out / "heatmap.png")

    print(f"[similarity] {len(ids)} clips, {len(ranked)} ranked pairs -> {out}")
    for e in excluded:
        print(f"    excluded (mean |v| {e['mean_magnitude']:.3f} < {min_motion}): {e['clip_id']}")
    for p in ranked[:5]:
        print(f"    #{p['rank']:<3} {p['score']:+.3f}  {p['clip_a']}  <->  {p['clip_b']}")
    return sim, ranked


def short_names(ids: Sequence[str]) -> list[str]:
    """Shorter display names (e.g. Pexels '5601797-hd_1920_1080_30fps' -> '5601797') if still unique."""
    short = [cid.replace("-", "_").split("_")[0] for cid in ids]
    return short if len(set(short)) == len(short) else list(ids)


def load_ranked(paths: OutputPaths) -> dict:
    if not paths.ranked_pairs.exists():
        raise SystemExit(f"{paths.ranked_pairs} not found — run the 'similarity' stage first")
    return json.loads(Path(paths.ranked_pairs).read_text())

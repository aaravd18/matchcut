"""Stage 6 — render videos with ffmpeg (hard cuts, no audio).

- outputs/montage/montage.mp4: every segment of the EDL joined with hard cuts.
- outputs/comparisons/rankNN__<A>__<B>.mp4: previews for the top h4 pairs.
  Each preview uses the better cut direction (A->B or B->A) and the best cut
  points from the transition scores, with `seconds_a` before the cut and
  `seconds_b` after it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from .config import OutputPaths, RenderConfig, SequenceConfig
from .similarity import short_names
from .transitions import breakdown, best_cut, load_features


def concat_segments(segments: Sequence[tuple[str, int, int]], out_path: Path, fps: float, crf: int = 18) -> None:
    """Join [start_frame, end_frame) ranges of (identically formatted) videos with hard cuts."""
    inputs, graph = [], []
    for i, (path, start, end) in enumerate(segments):
        inputs += ["-i", str(path)]
        graph.append(f"[{i}:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{i}]")
    graph.append("".join(f"[v{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[out]")
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
           "-map", "[out]", "-r", str(fps), "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)


def run_montage(manifest: dict, edl: dict, paths: OutputPaths, crf: int = 18) -> Path:
    fps = manifest["clips"][0]["fps"]
    segs = [(s["path"], s["in_frame"], s["out_frame"]) for s in edl["segments"]]
    out = paths.montage / "montage.mp4"
    concat_segments(segs, out, fps, crf)
    print(f"[render] montage: {len(segs)} clips, {edl['total_duration']:.1f}s -> {out}")
    return out


def run_previews(manifest: dict, ranked: dict, paths: OutputPaths, cfg: RenderConfig,
                 seq_cfg: SequenceConfig) -> list[dict]:
    total = cfg.seconds_a + cfg.seconds_b
    if not 4.0 <= total <= 6.0:
        print(f"[render] note: preview length {total:.1f}s is outside the 4–6 s target")
    clips = {c["id"]: c for c in manifest["clips"]}
    feats = {f.clip_id: f for f in load_features(manifest, paths)}
    names = dict(zip(ranked["clips"], short_names(ranked["clips"])))
    paths.comparisons.mkdir(parents=True, exist_ok=True)
    for old in paths.comparisons.glob("rank*.mp4"):
        old.unlink()

    rendered = []
    for pair in ranked["pairs"][: cfg.top_n]:
        options = []
        for x, y in ((pair["clip_a"], pair["clip_b"]), (pair["clip_b"], pair["clip_a"])):
            i, j, s = best_cut(feats[x], feats[y], seq_cfg, cfg.seconds_a, cfg.seconds_b)
            options.append((s, x, y, i, j))
        s, a, b, i, j = max(options, key=lambda o: o[0])
        fa, fb = feats[a], feats[b]
        fps = fa.fps
        out_b, in_b = int(fa.boundaries[i]), int(fb.boundaries[j])
        n_a, n_b = int(round(cfg.seconds_a * fps)), int(round(cfg.seconds_b * fps))
        name = f"rank{pair['rank']:02d}__{names[a]}__{names[b]}.mp4"
        concat_segments([(clips[a]["path"], out_b - n_a, out_b), (clips[b]["path"], in_b, in_b + n_b)],
                        paths.comparisons / name, fps, cfg.crf)
        info = {"rank": pair["rank"], "h4_score": pair["score"], "clip_a": a, "clip_b": b, "file": name,
                "a_out_s": round(out_b / fps, 3), "b_in_s": round(in_b / fps, 3),
                "cut_at_seconds": round(n_a / fps, 3), "duration": round((n_a + n_b) / fps, 3),
                "transition": breakdown(fa, fb, i, j, seq_cfg)}
        rendered.append(info)
        print(f"[render] #{pair['rank']} {name}  (A out {info['a_out_s']:.2f}s, B in {info['b_in_s']:.2f}s, "
              f"transition {info['transition']['total']:+.3f})")

    (paths.comparisons / "index.json").write_text(json.dumps(rendered, indent=2))
    return rendered


def load_rendered(paths: OutputPaths) -> list[dict]:
    p = paths.comparisons / "index.json"
    return json.loads(p.read_text()) if p.exists() else []

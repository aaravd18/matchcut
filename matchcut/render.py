"""Stage 6 — render videos with ffmpeg (hard cuts, no audio).

- outputs/montage/montage.mp4: every segment of the EDL joined with hard cuts,
  with the soundtrack (from the EDL's music start, faded out at the end) when
  the montage was cut to music. --click also writes montage_click.mp4 with a
  metronome on the detected beats, to check the sync by ear.
- outputs/comparisons/rankNN__<A>__<B>.mp4: previews for the top h4 pairs.
  Each preview uses the better cut direction (A->B or B->A) and the best cut
  points from the transition scores, with `seconds_a` before the cut and
  `seconds_b` after it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import OutputPaths, RenderConfig, SequenceConfig
from .similarity import short_names
from .transitions import breakdown, best_cut, load_features


def concat_segments(segments: Sequence[tuple[str, int, int]], out_path: Path, fps: float, crf: int = 18,
                    audio: Optional[dict] = None) -> None:
    """Join [start_frame, end_frame) ranges of (identically formatted) videos with hard cuts.

    audio (optional): {"path", "start", "duration", "fade", "click_wav"} — music
    from `start` seconds for `duration` seconds, faded out over the last `fade`
    seconds, optionally mixed with a click track.
    """
    inputs, graph = [], []
    for i, (path, start, end) in enumerate(segments):
        inputs += ["-i", str(path)]
        graph.append(f"[{i}:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{i}]")
    graph.append("".join(f"[v{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[out]")
    maps = ["-map", "[out]"]
    if audio:
        a = len(segments)
        dur, fade = audio["duration"], min(audio["fade"], audio["duration"])
        inputs += ["-ss", f"{audio['start']:.4f}", "-t", f"{dur:.4f}", "-i", str(audio["path"])]
        music = f"[{a}:a]aformat=sample_rates=44100:channel_layouts=stereo,afade=t=out:st={dur - fade:.3f}:d={fade:.3f}"
        if audio.get("click_wav"):
            inputs += ["-i", str(audio["click_wav"])]
            graph.append(music + "[m]")
            graph.append(f"[{a + 1}:a]aformat=sample_rates=44100:channel_layouts=stereo[c]")
            graph.append("[m][c]amix=inputs=2:normalize=0:duration=first[aout]")
        else:
            graph.append(music + "[aout]")
        maps += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
           *maps, "-r", str(fps), "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)


def write_click_track(beat_times: Sequence[float], is_down: Sequence[bool], duration: float, path: Path,
                      sr: int = 44100) -> None:
    """Metronome: 1 kHz tick on every beat, louder 2 kHz tick on downbeats."""
    import librosa
    import soundfile as sf

    n = int(round(duration * sr))
    bt = np.asarray(beat_times, float)
    down = np.asarray(is_down, bool)
    y = 0.35 * librosa.clicks(times=bt[~down], sr=sr, click_freq=1000, click_duration=0.05, length=n)
    y += 0.6 * librosa.clicks(times=bt[down], sr=sr, click_freq=2000, click_duration=0.07, length=n)
    sf.write(str(path), y.astype(np.float32), sr)


def run_montage(manifest: dict, edl: dict, paths: OutputPaths, crf: int = 18, click: bool = False) -> Path:
    fps = edl.get("fps", manifest["clips"][0]["fps"])
    segs = [(s["path"], s["in_frame"], s["out_frame"]) for s in edl["segments"]]
    out = paths.montage / "montage.mp4"
    stale_click = paths.montage / "montage_click.mp4"
    mu = edl.get("music")
    audio = None
    if mu:
        audio = {"path": mu["path"], "start": mu["start"], "duration": edl["total_duration"],
                 "fade": mu["config"]["fade"]}
    concat_segments(segs, out, fps, crf, audio)
    print(f"[render] montage: {len(segs)} clips, {edl['total_duration']:.1f}s"
          + (f", music {mu['start']:.2f}–{mu['end']:.2f}s" if mu else ", no audio") + f" -> {out}")
    if click and mu:
        wav = paths.montage / "clicks.wav"
        write_click_track(mu["beat_times"], mu["is_downbeat"], edl["total_duration"], wav)
        concat_segments(segs, stale_click, fps, crf, {**audio, "click_wav": wav})
        print(f"[render] click-track check: {stale_click}")
    elif stale_click.exists():
        stale_click.unlink()
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

"""Stage 1 — standardize raw clips to 640x360, 30 fps, at most N seconds (default 20), without distortion.

Each clip is scaled to fit inside the target frame (aspect ratio preserved) and
padded with black bars if its aspect ratio differs (16:9 sources get no bars).
The picture rectangle inside the frame ("content box") is stored in the
manifest so the motion stage can ignore the bars.

Outputs: outputs/preprocessed/<clip_id>.mp4 and outputs/preprocessed/manifest.json
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from .config import OutputPaths, PreprocessConfig
from .video import count_frames, list_videos, probe


def fit_box(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the largest even-sized, aspect-preserving rectangle centred in the output."""
    scale = min(out_w / src_w, out_h / src_h)
    w = min(out_w, max(2, 2 * round(src_w * scale / 2)))
    h = min(out_h, max(2, 2 * round(src_h * scale / 2)))
    x = (out_w - w) // 2
    y = (out_h - h) // 2
    return x - x % 2, y - y % 2, w, h


def choose_window(duration: float, target: float, mode: str) -> tuple[float, float]:
    """(start, length) in seconds of the part of the source clip to keep."""
    if duration <= target:
        return 0.0, duration
    if mode == "center":
        return (duration - target) / 2.0, target
    if mode == "start":
        return 0.0, target
    raise ValueError(f"unknown window mode: {mode!r} (use 'center' or 'start')")


def preprocess_clip(src: Path, dst: Path, cfg: PreprocessConfig) -> dict:
    info = probe(src)
    x, y, w, h = fit_box(info["width"], info["height"], cfg.width, cfg.height)
    start, length = choose_window(info["duration"], cfg.duration, cfg.window)
    # Stop slightly before the source ends so we never ask for frames that do not exist.
    n_target = max(1, int(round(length * cfg.fps)) - (1 if length < cfg.duration else 0))
    vf = (
        f"fps={cfg.fps},"                                   # resample time first (cheaper for 60 fps sources)
        f"scale={w}:{h}:flags=area,"                        # aspect-preserving downscale
        f"pad={cfg.width}:{cfg.height}:{x}:{y}:color=black,"  # letterbox/pillarbox if needed
        "setsar=1"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-vf", vf, "-frames:v", str(n_target), "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(cfg.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    n_frames = count_frames(dst)
    return {
        "id": dst.stem,
        "source": str(src),
        "path": str(dst),
        "source_width": info["width"],
        "source_height": info["height"],
        "source_fps": round(info["fps"], 3),
        "source_duration": round(info["duration"], 3),
        "window_start": round(start, 3),
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
        "n_frames": n_frames,
        "duration": round(n_frames / cfg.fps, 3),
        "content_box": [x, y, w, h],
    }


def run_preprocess(input_dir: Path, paths: OutputPaths, cfg: PreprocessConfig) -> dict:
    sources = list_videos(input_dir)
    if not sources:
        raise SystemExit(f"no video files found in {input_dir}")
    paths.preprocessed.mkdir(parents=True, exist_ok=True)
    ids = [s.stem for s in sources]
    if len(set(ids)) != len(ids):
        raise SystemExit("two input files share the same name stem; rename one")

    clips = []
    for i, src in enumerate(sources, 1):
        dst = paths.preprocessed / f"{src.stem}.mp4"
        print(f"[preprocess] ({i}/{len(sources)}) {src.name}")
        clips.append(preprocess_clip(src, dst, cfg))

    manifest = {"config": asdict(cfg), "input_dir": str(input_dir), "clips": clips}
    paths.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[preprocess] wrote {len(clips)} clips -> {paths.preprocessed}")
    return manifest


def load_manifest(paths: OutputPaths) -> dict:
    if not paths.manifest.exists():
        raise SystemExit(f"{paths.manifest} not found — run the 'preprocess' stage first")
    return json.loads(paths.manifest.read_text())

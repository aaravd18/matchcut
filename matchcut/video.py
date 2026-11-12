"""Small video I/O helpers: list files, probe metadata (ffprobe), decode frames (OpenCV)."""
from __future__ import annotations

import json
import subprocess
from collections import deque
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def list_videos(directory: Path) -> list[Path]:
    """Video files directly inside `directory` (not recursive), sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"input directory not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def probe(path: Path) -> dict:
    """Display width/height (rotation applied), frame rate and duration of a video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]
    info = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    stream = info["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    if abs(rotation) % 180 == 90:  # phone footage: ffmpeg auto-rotates, so swap
        width, height = height, width
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": width,
        "height": height,
        "fps": float(num) / float(den),
        "duration": float(info["format"]["duration"]),
    }


def count_frames(path: Path) -> int:
    """Exact frame count by decoding (container metadata can be wrong)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_frames(
    path: Path, size: Optional[Tuple[int, int]] = None, gray: bool = False
) -> Iterator[np.ndarray]:
    """Yield BGR (or grayscale) uint8 frames, optionally resized to size=(width, height)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
                frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame
    finally:
        cap.release()


def read_frames_at(
    path: Path, indices, size: Optional[Tuple[int, int]] = None
) -> dict[int, np.ndarray]:
    """{index: BGR frame} for the requested frame indices, decoding sequentially once.

    Only the requested frames are kept in memory.
    """
    wanted = set(int(i) for i in indices)
    last = max(wanted) if wanted else -1
    out: dict[int, np.ndarray] = {}
    for i, frame in enumerate(read_frames(path, size=size)):
        if i in wanted:
            out[i] = frame
        if i >= last:
            break
    missing = wanted - set(out)
    if missing:
        raise IndexError(f"frames {sorted(missing)[:5]} out of range for {path}")
    return out


def read_frame(path: Path, index: int) -> np.ndarray:
    """One BGR frame by index; negative indices count from the end (-1 = last frame).

    Decodes sequentially, which is exact and cheap for our 5-second clips.
    """
    if index >= 0:
        for i, frame in enumerate(read_frames(path)):
            if i == index:
                return frame
        raise IndexError(f"frame {index} out of range for {path}")
    tail: deque = deque(maxlen=-index)
    for frame in read_frames(path):
        tail.append(frame)
    if len(tail) < -index:
        raise IndexError(f"frame {index} out of range for {path}")
    return tail[0]

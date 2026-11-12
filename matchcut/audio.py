"""Stage — soundtrack beat analysis (librosa).

1. Decode the music with ffmpeg to mono float32 (so any format ffmpeg reads works).
2. Onset strength: librosa.onset.onset_strength (spectral flux of a log-mel
   spectrogram, median across bands): peaks where drum hits / notes start.
3. Beats: librosa.beat.beat_track — Ellis (2007) dynamic programming: picks
   beat times on strong onsets while keeping the spacing near the tempo, with
   a tempo prior centred on `start_bpm` (avoids half/double-tempo errors).
4. Downbeats (heuristic, assumes 4/4): for each of the 4 possible bar phases,
   average (a) low-frequency (< 150 Hz, kick drum) onset strength on those
   beats, (b) chord change (chroma novelty) going into those beats and
   (c) the large loudness jumps that mark new sections; the phase with the
   highest combined z-score is "beat 1". Override with --downbeat-offset if
   the click preview sounds wrong.
   Note: librosa trims leading beats with no percussion (e.g. a pad-only
   intro), so the beat grid can start well after 0 s.

Outputs: outputs/beats/beats.json, beats.png
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MusicConfig, OutputPaths

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def find_music(input_dir: Path) -> Optional[Path]:
    folder = Path(input_dir) / "music"
    if not folder.is_dir():
        return None
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if len(files) > 1:
        print(f"[beats] several audio files in {folder}; using {files[0].name} (choose with --music)")
    return files[0] if files else None


def load_audio(path: Path, sr: int) -> np.ndarray:
    """Mono float32 samples at `sr`, decoded by ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def analyze_signal(y: np.ndarray, sr: int, cfg: MusicConfig) -> dict:
    """Tempo, beats, beat strengths and downbeats of a mono signal."""
    import librosa

    hop = cfg.hop_length
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop, start_bpm=cfg.start_bpm, bpm=cfg.bpm, units="frames")
    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 8:
        raise ValueError("fewer than 8 beats detected; is this music with a steady beat?")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # Strength of each beat: onset peak within +-2 frames, scaled so the 99th percentile = 1.
    env = np.clip(onset / (np.percentile(onset, 99) + 1e-9), 0, 1)
    strength = np.array([env[max(0, f - 2): f + 3].max() for f in beat_frames])

    # Downbeat phase. (a) kick drum: onset strength of mel bands centred below 150 Hz.
    n_mels = 64
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=n_mels)
    centres = librosa.mel_frequencies(n_mels=n_mels + 2, fmax=sr / 2)[1:-1]
    low = librosa.onset.onset_strength(S=librosa.power_to_db(mel[centres < 150]), sr=sr, hop_length=hop)
    low_at = np.array([low[max(0, f - 2): f + 3].max() for f in beat_frames])
    # (b) harmony: chroma change from the previous beat to this one.
    chroma = librosa.util.sync(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop), beat_frames, aggregate=np.median)
    chroma = chroma[:, 1:len(beat_frames) + 1] if chroma.shape[1] > len(beat_frames) else chroma
    c = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    novelty = np.r_[0.0, 1.0 - np.sum(c[:, 1:] * c[:, :-1], axis=0)][:len(beat_frames)]
    # (c) sections: loudness change (dB) between the 4 beats after and the 4 beats before;
    #     new sections (drops, breakdowns) almost always start on beat 1 of a bar.
    rms = librosa.util.sync(librosa.feature.rms(y=y, hop_length=hop), beat_frames, aggregate=np.mean)[0]
    rms = rms[1:len(beat_frames) + 1] if rms.shape[0] > len(beat_frames) else rms
    db = 20 * np.log10(rms + 1e-6)
    change = np.array([abs(db[i:i + 4].mean() - db[i - 4:i].mean()) if 4 <= i <= len(db) - 4 else 0.0
                       for i in range(len(db))])
    big = change >= np.percentile(change, 95)   # only the largest jumps carry section information
    low_score = np.array([low_at[p::4].mean() for p in range(4)])
    harm_score = np.array([novelty[p::4][1:].mean() if len(novelty[p::4]) > 1 else 0.0 for p in range(4)])
    sect_score = np.array([(change * big)[p::4].sum() for p in range(4)])
    combined = _z(low_score) + _z(harm_score) + _z(sect_score)
    offset = int(cfg.downbeat_offset) if cfg.downbeat_offset is not None else int(np.argmax(combined))
    is_down = (np.arange(len(beat_frames)) - offset) % 4 == 0

    ibi = np.diff(beat_times)
    return {
        "sr": sr, "hop_length": hop, "duration": round(len(y) / sr, 3),
        "tempo_librosa": round(float(np.atleast_1d(tempo)[0]), 2),
        "tempo": round(float(60.0 / np.median(ibi)), 2),          # from the beats themselves
        "beat_period": round(float(np.median(ibi)), 4),
        "ibi_std_ms": round(float(ibi.std() * 1000), 1),
        "n_beats": int(len(beat_times)),
        "downbeat_offset": offset,
        "downbeat_scores": {"kick": low_score.round(4).tolist(), "harmony": harm_score.round(4).tolist(),
                            "sections": sect_score.round(3).tolist(),
                            "combined": [round(float(x), 3) for x in combined], "forced": cfg.downbeat_offset is not None},
        "beat_times": beat_times.round(4).tolist(),
        "beat_strength": strength.round(4).tolist(),
        "is_downbeat": is_down.tolist(),
        "_onset_env": env, "_onset_times": librosa.times_like(env, sr=sr, hop_length=hop),
    }


def plot_beats(info: dict, path: Path, zoom_seconds: float = 8.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, env = info["_onset_times"], info["_onset_env"]
    bt = np.array(info["beat_times"])
    down = np.array(info["is_downbeat"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [1, 1.3]})
    for ax, (lo, hi) in zip(axes, [(0, info["duration"]), (bt[0], bt[0] + zoom_seconds)]):
        sel = (t >= lo) & (t <= hi)
        ax.plot(t[sel], env[sel], color="#6b6a66", linewidth=0.8)
        if hi - lo <= 2 * zoom_seconds:
            for b, d in zip(bt, down):
                if lo <= b <= hi:
                    ax.axvline(b, color="#e34948" if d else "#2a78d6", linewidth=2 if d else 1, alpha=0.9 if d else 0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.05)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_title(f"Onset strength — tempo {info['tempo']:.1f} BPM, {info['n_beats']} beats", loc="left", fontsize=11)
    axes[1].set_title(f"First {zoom_seconds:.0f} s: beats (blue) and downbeats (red)", loc="left", fontsize=11)
    axes[1].set_xlabel("music time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_beats(input_dir: Path, paths: OutputPaths, cfg: MusicConfig) -> Optional[dict]:
    music = Path(cfg.path) if cfg.path else find_music(input_dir)
    if not cfg.enabled or music is None:
        if paths.beats_json.exists():
            paths.beats_json.unlink()   # avoid using a stale analysis
        print("[beats] no music (" + ("disabled" if not cfg.enabled else f"nothing in {Path(input_dir) / 'music'}")
              + "); the montage will use free timing and no audio")
        return None
    y = load_audio(music, cfg.sr)
    info = analyze_signal(y, cfg.sr, cfg)
    paths.beats.mkdir(parents=True, exist_ok=True)
    plot_beats(info, paths.beats / "beats.png")
    out = {"path": str(music), "config": asdict(cfg), **{k: v for k, v in info.items() if not k.startswith("_")}}
    paths.beats_json.write_text(json.dumps(out, indent=2))
    print(f"[beats] {music.name}: {out['tempo']:.1f} BPM (librosa estimate {out['tempo_librosa']}), "
          f"{out['n_beats']} beats, beat spacing jitter {out['ibi_std_ms']} ms, downbeat = beat phase {out['downbeat_offset']} "
          f"(scores {out['downbeat_scores']['combined']})")
    return out


def load_beats(paths: OutputPaths) -> Optional[dict]:
    return json.loads(paths.beats_json.read_text()) if paths.beats_json.exists() else None


# How good a cut feels by position in a 4/4 bar: beat 1 (bar line) > beat 3
# (half bar) > beats 2/4. Onset strength alone can't express this: in most pop
# the snare on 2 and 4 is the loudest hit.
METRIC_WEIGHT = (1.0, 0.25, 0.6, 0.25)


@dataclass
class BeatGrid:
    """Beats from the montage start onward, in montage time."""
    times: np.ndarray       # (M,) seconds, times[0] = 0
    frames: np.ndarray      # (M,) video frame index of each beat
    reward: np.ndarray      # (M,) METRIC_WEIGHT[bar position] * (0.5 + 0.5 * onset strength)
    is_down: np.ndarray     # (M,) bool
    bar_pos: np.ndarray     # (M,) 0..3 = beat 1..4 of the bar
    music_start: float      # music time that plays at montage t = 0
    period: float           # median beat period (s)


def beat_grid(beats: dict, start: float, fps: float) -> BeatGrid:
    bt = np.asarray(beats["beat_times"], float)
    down = np.asarray(beats["is_downbeat"], bool)
    strength = np.asarray(beats["beat_strength"], float)
    bar_pos = (np.arange(len(bt)) - int(beats["downbeat_offset"])) % 4
    cand = np.nonzero(down & (bt >= start - 1e-6))[0]
    if len(cand) == 0:
        raise SystemExit(f"no downbeat after --music-start {start}s (track has {bt[-1]:.1f}s of beats)")
    i0 = int(cand[0])
    times = bt[i0:] - bt[i0]
    reward = np.asarray(METRIC_WEIGHT)[bar_pos[i0:]] * (0.5 + 0.5 * strength[i0:])
    return BeatGrid(times=times, frames=np.round(times * fps).astype(int), reward=reward,
                    is_down=down[i0:], bar_pos=bar_pos[i0:],
                    music_start=float(bt[i0]), period=float(beats["beat_period"]))

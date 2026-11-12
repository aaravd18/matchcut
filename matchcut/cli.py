"""Command-line entry point.

    python -m matchcut run                       # every stage: data/ -> outputs/
    python -m matchcut run --max-clips 4         # montage of the best-connected 4 clips

    python -m matchcut preprocess --input data   # 1  standardize clips
    python -m matchcut motion                    # 2  h4 mean flow + flow time series
    python -m matchcut similarity                # 3  h4 pairwise ranking (+ minimum-motion filter)
    python -m matchcut transitions               # 4  score candidate cut points
    python -m matchcut beats                     # 5  tempo / beats / downbeats of data/music/*
    python -m matchcut sequence --max-clips all  # 6  order clips + place cuts on the beat -> EDL
    python -m matchcut render --click            # 7  montage.mp4 with music (+ metronome check)
    python -m matchcut report                    # 8  HTML report

Each stage reads the previous stages' files from --out, so any stage can be
re-run on its own (e.g. `sequence` then `render` then `report` to try a
different --max-clips without recomputing optical flow).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import FlowConfig, MusicConfig, OutputPaths, PreprocessConfig, RenderConfig, SequenceConfig


def _max_clips(value: str):
    if value.lower() == "all":
        return None
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("--max-clips must be 'all' or a positive integer")
    return n


def _add_preprocess_args(p):
    g = p.add_argument_group("preprocess")
    g.add_argument("--input", type=Path, default=Path("data"), help="folder of raw clips (default: data)")
    g.add_argument("--duration", type=float, default=PreprocessConfig.duration,
                   help="max seconds kept per clip (default: %(default)s)")
    g.add_argument("--window", choices=["center", "start"], default=PreprocessConfig.window,
                   help="keep the centre or the start of long source clips")


def _add_motion_args(p):
    g = p.add_argument_group("motion (h4)")
    g.add_argument("--stride", type=int, default=FlowConfig.stride, help="h4: sample a flow pair every N frames")
    g.add_argument("--gap", type=int, default=FlowConfig.gap, help="h4: flow between frame j and j+GAP (1 = adjacent)")
    g.add_argument("--flow-size", type=int, nargs=2, metavar=("W", "H"),
                   default=(FlowConfig.flow_width, FlowConfig.flow_height), help="resolution for flow")


def _add_min_motion(p):
    p.add_argument("--min-motion", type=float, default=SequenceConfig.min_motion,
                   help="px/frame below which a clip/window counts as static (default: %(default)s)")


def _add_sequence_args(p):
    g = p.add_argument_group("montage")
    g.add_argument("--max-clips", type=_max_clips, default=None, metavar="all|N",
                   help="use every clip (all, default) or only the best-connected N clips")
    g.add_argument("--order", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None,
                   help="fixed clip order, comma-separated ids or short names (skips the search)")
    g.add_argument("--seg-min", type=float, default=SequenceConfig.seg_min, help="min seconds per clip")
    g.add_argument("--seg-max", type=float, default=SequenceConfig.seg_max, help="max seconds per clip")
    g.add_argument("--seg-target", type=float, default=SequenceConfig.seg_target, help="preferred seconds per clip")
    g.add_argument("--weights", type=float, nargs=3, metavar=("MOTION", "COMP", "COLOR"),
                   default=(SequenceConfig.w_motion, SequenceConfig.w_comp, SequenceConfig.w_color),
                   help="cut score weights (default: %(default)s)")


def _add_music_args(p):
    g = p.add_argument_group("music")
    g.add_argument("--no-music", action="store_true", help="ignore data/music: free timing, silent montage")
    g.add_argument("--music", type=str, default=None, help="soundtrack file (default: first file in <input>/music)")
    if "--input" not in p._option_string_actions:
        p.add_argument("--input", type=Path, default=Path("data"), help="folder whose music/ subfolder holds the soundtrack")
    g.add_argument("--bpm", type=float, default=None, help="force the tempo instead of estimating it")
    g.add_argument("--downbeat-offset", type=int, choices=[0, 1, 2, 3], default=None,
                   help="force which detected beat (mod 4) is beat 1 of a bar")
    g.add_argument("--music-start", type=float, default=MusicConfig.start,
                   help="start the montage at the first downbeat at/after this music time (s)")
    g.add_argument("--cut-on", choices=["beats", "bars"], default=MusicConfig.cut_on,
                   help="cut on any beat (lengths vary) or only on bar lines")
    g.add_argument("--w-beat", type=float, default=MusicConfig.w_beat,
                   help="reward for cutting on a strong beat / downbeat (default: %(default)s)")
    g.add_argument("--click", action="store_true", help="also render montage_click.mp4 with a metronome on the beats")


def _add_render_args(p):
    g = p.add_argument_group("render (h4 pair previews)")
    g.add_argument("--top-n", type=int, default=RenderConfig.top_n, help="previews to render")
    g.add_argument("--seconds-a", type=float, default=RenderConfig.seconds_a, help="preview: seconds before the cut")
    g.add_argument("--seconds-b", type=float, default=RenderConfig.seconds_b, help="preview: seconds after the cut")


def _add_report_args(p):
    g = p.add_argument_group("report")
    g.add_argument("--top-k", type=int, default=10, help="h4 pairs shown in the report")


STAGES = {
    "run": ("run every stage in order",
            [_add_preprocess_args, _add_motion_args, _add_min_motion, _add_sequence_args, _add_music_args,
             _add_render_args, _add_report_args]),
    "preprocess": ("standardize raw clips (640x360, 30 fps, <= 20 s)", [_add_preprocess_args]),
    "motion": ("h4 mean flow + pooled flow time series", [_add_motion_args]),
    "similarity": ("h4 cosine matrix and ranked pairs", [_add_min_motion]),
    "transitions": ("score candidate cut points between clips", [_add_min_motion, _add_sequence_args]),
    "beats": ("detect tempo, beats and downbeats of the soundtrack (librosa)", [_add_music_args]),
    "sequence": ("choose clip order and cut points (EDL), on the beat if music was analysed",
                 [_add_min_motion, _add_sequence_args, _add_music_args]),
    "render": ("render montage.mp4 (+ music) and h4 pair previews",
               [_add_min_motion, _add_sequence_args, _add_music_args, _add_render_args]),
    "report": ("HTML report", [_add_report_args]),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m matchcut", description="Motion match-cut montage (Netflix h4 + cut-point scoring).")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, (help_text, adders) in STAGES.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out", type=Path, default=Path("outputs"), help="output root (default: outputs)")
        for add in adders:
            add(p)
    return parser


def _seq_cfg(args) -> SequenceConfig:
    wm, wc, wcol = args.weights
    return SequenceConfig(min_motion=args.min_motion, max_clips=args.max_clips, order=args.order,
                          seg_min=args.seg_min, seg_max=args.seg_max, seg_target=args.seg_target,
                          w_motion=wm, w_comp=wc, w_color=wcol)


def _music_cfg(args) -> MusicConfig:
    return MusicConfig(enabled=not args.no_music, path=args.music, bpm=args.bpm,
                       downbeat_offset=args.downbeat_offset, start=args.music_start,
                       cut_on=args.cut_on, w_beat=args.w_beat)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = OutputPaths(args.out)
    stage = args.stage
    t0 = time.time()

    from .preprocess import load_manifest, run_preprocess
    from .similarity import load_ranked

    if stage in ("run", "preprocess"):
        run_preprocess(args.input, paths, PreprocessConfig(duration=args.duration, window=args.window))
    if stage in ("run", "motion"):
        from .motion import run_motion
        run_motion(load_manifest(paths), paths, FlowConfig(stride=args.stride, gap=args.gap,
                                                           flow_width=args.flow_size[0], flow_height=args.flow_size[1]))
    if stage in ("run", "similarity"):
        from .similarity import run_similarity
        run_similarity(load_manifest(paths), paths, args.min_motion)
    if stage in ("run", "transitions"):
        from .transitions import run_transitions
        run_transitions(load_manifest(paths), paths, _seq_cfg(args))
    if stage in ("run", "beats"):
        from .audio import run_beats
        run_beats(args.input, paths, _music_cfg(args))
    if stage in ("run", "sequence"):
        from .sequence import run_sequence
        run_sequence(load_manifest(paths), paths, _seq_cfg(args), _music_cfg(args))
    if stage in ("run", "render"):
        from .render import run_montage, run_previews
        from .sequence import load_edl
        manifest = load_manifest(paths)
        edl = load_edl(paths)
        if edl is None:
            raise SystemExit(f"{paths.edl} not found — run the 'sequence' stage first")
        run_montage(manifest, edl, paths, click=args.click)
        run_previews(manifest, load_ranked(paths), paths,
                     RenderConfig(top_n=args.top_n, seconds_a=args.seconds_a, seconds_b=args.seconds_b), _seq_cfg(args))
    if stage in ("run", "report"):
        from .report import run_report
        report = run_report(load_manifest(paths), load_ranked(paths), paths, top_k=args.top_k)
        print(f"\nMontage:  {(paths.montage / 'montage.mp4').resolve()}")
        print(f"Report:   {report.resolve()}")
    print(f"[{stage}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Stage 7 — HTML report (outputs/report/index.html).

1. Montage: the rendered video, a timeline, and one card per cut (out-frame,
   in-frame, the motion windows that were compared, score breakdown).
2. h4 analysis: top-K pairs with preview videos, the similarity matrix, clips
   excluded by the minimum-motion filter, and the lowest-scoring pairs.
3. Every clip's middle frame and mean flow.

Open it in a browser; it references images and videos by relative path.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import OutputPaths
from .motion import load_reps
from .render import load_rendered
from .sequence import load_edl
from .similarity import short_names
from .transitions import load_features
from .video import read_frames_at
from .visualize import DIVERGING_MID, DIVERGING_NEG, DIVERGING_POS, color_wheel, flow_panel

THUMB = (480, 270)
JPG = [cv2.IMWRITE_JPEG_QUALITY, 85]
esc = html.escape


# ---------------------------------------------------------------- images

def _clip_images(manifest: dict, reps: dict, img_dir: Path) -> None:
    for clip in manifest["clips"]:
        mid = clip["n_frames"] // 2
        frame = read_frames_at(Path(clip["path"]), [mid], size=THUMB)[mid]
        cv2.imwrite(str(img_dir / f"{clip['id']}_middle.jpg"), frame, JPG)
        cv2.imwrite(str(img_dir / f"{clip['id']}_flow.jpg"),
                    flow_panel(reps[clip["id"]].mean_flow, size=THUMB, arrow_spacing=30), JPG)


def _cut_images(manifest: dict, edl: dict, feats: dict, grid: tuple[int, int], img_dir: Path) -> None:
    paths = {c["id"]: Path(c["path"]) for c in manifest["clips"]}
    wanted: dict[str, list[int]] = {}
    for c in edl["cuts"]:
        wanted.setdefault(c["from"], []).append(c["out_frame"] - 1)
        wanted.setdefault(c["to"], []).append(c["in_frame"])
    frames = {cid: read_frames_at(paths[cid], idx, size=THUMB) for cid, idx in wanted.items()}
    gw, gh = grid
    for c in edl["cuts"]:
        k = c["index"]
        cv2.imwrite(str(img_dir / f"cut{k:02d}_out.jpg"), frames[c["from"]][c["out_frame"] - 1], JPG)
        cv2.imwrite(str(img_dir / f"cut{k:02d}_in.jpg"), frames[c["to"]][c["in_frame"]], JPG)
        out_flow = feats[c["from"]].out_motion[c["out_k"]].reshape(gh, gw, 2)
        in_flow = feats[c["to"]].in_motion[c["in_k"]].reshape(gh, gw, 2)
        cv2.imwrite(str(img_dir / f"cut{k:02d}_out_flow.jpg"), flow_panel(out_flow, size=THUMB, arrow_spacing=30), JPG)
        cv2.imwrite(str(img_dir / f"cut{k:02d}_in_flow.jpg"), flow_panel(in_flow, size=THUMB, arrow_spacing=30), JPG)


# ---------------------------------------------------------------- html pieces

def _hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def _cell_color(s: float) -> tuple[str, str]:
    """Background on the diverging scale, plus a readable ink colour."""
    end = _hex_to_rgb(DIVERGING_POS if s >= 0 else DIVERGING_NEG)
    mid = _hex_to_rgb(DIVERGING_MID)
    rgb = mid + (end - mid) * min(abs(s), 1.0)
    lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb), ("#1f1f1e" if lum > 0.5 else "#ffffff")


def _fmt_motion(rep) -> str:
    return f"|v| {rep.mean_magnitude:.2f} · dx {rep.mean_dx:+.2f} · dy {rep.mean_dy:+.2f}"


def _fig(src: str, alt: str, caption: str) -> str:
    return f'<figure><img loading="lazy" src="{esc(src)}" alt="{esc(alt)}"><figcaption>{caption}</figcaption></figure>'


def _score_row(t: dict, w: dict) -> str:
    motion = (f"{t['motion']:+.2f} <span class='dim'>(cos {t['motion_cos']:+.2f} × speed {t['speed_ratio']:.2f})</span>"
              if t["motion_valid"] else "<span class='dim'>n/a: too little motion, counts as 0</span>")
    return (f"<dl class='scores'>"
            f"<div><dt>motion ×{w['w_motion']}</dt><dd>{motion}</dd></div>"
            f"<div><dt>composition ×{w['w_comp']}</dt><dd>{t['composition']:+.2f}</dd></div>"
            f"<div><dt>colour ×{w['w_color']}</dt><dd>{t['color']:.2f}</dd></div>"
            f"<div><dt>speed out / in</dt><dd>{t['out_speed']:.2f} / {t['in_speed']:.2f} px/f</dd></div></dl>")


def _music_html(edl: dict, paths: OutputPaths) -> str:
    mu = edl.get("music")
    if not mu:
        return '<p class="note">No soundtrack: free timing, silent video (add a file to data/music/ to cut on the beat).</p>'
    total = edl["total_duration"]
    ticks = "".join(
        f'<i class="{"down" if d else ""}" style="left:{100 * t / total:.3f}%"></i>'
        for t, d in zip(mu["beat_times"], mu["is_downbeat"]))
    cfg = mu["config"]
    click = ('<a href="../montage/montage_click.mp4">montage_click.mp4</a> (metronome on the detected beats; '
             'louder tick = bar line)' if (paths.montage / "montage_click.mp4").exists()
             else "render with <code>--click</code> for a metronome version to check the sync by ear")
    return f"""
<div class="beats">{ticks}</div>
<p class="note">Beat ticks under the timeline (tall = bar line). Music: <b>{esc(Path(mu['path']).name)}</b>,
{mu['tempo']:.1f} BPM, playing {mu['start']:.2f}–{mu['end']:.2f}s of the track. Clips last {', '.join(map(str, mu['allowed_beats']))} beats
(cut on: {esc(cfg['cut_on'])}, beat weight {cfg['w_beat']}); {mu['cuts_on_downbeats']}/{len(edl['cuts'])} cuts on bar lines;
ends on a bar line: {'yes' if mu['ends_on_downbeat'] else 'no'}. Check: {click}.</p>
<figure class="wide"><img loading="lazy" src="../beats/beats.png" alt="onset strength with detected beats and downbeats">
<figcaption>Beat analysis of the whole track (librosa onset strength; red = bar lines in the zoom)</figcaption></figure>"""


def _montage_section(edl: dict, paths: OutputPaths) -> str:
    cfg = edl["config"]
    total = edl["total_duration"]
    timeline = "".join(
        f'<div class="seg" style="flex:{s["duration"]}" title="{esc(s["clip_id"])}: {s["in_s"]:.2f}–{s["out_s"]:.2f}s">'
        f'<b>{s["index"]}</b> {esc(s["short_name"])}<span>{s["duration"]:.1f}s'
        + (f' · {s["beats"]}♩' if "beats" in s else "") + '</span></div>'
        for s in edl["segments"])
    cards = []
    for c in edl["cuts"]:
        k = c["index"]
        fs, ts = esc(c["from_short"]), esc(c["to_short"])
        beat = ""
        if "beat" in c:
            beat = (f' · beat {c["beat"]} '
                    + ('<span class="chip">bar line</span>' if c["downbeat"] else f'(beat {c["beat"] % 4 + 1} of the bar)'))
        cards.append(f"""
<article class="pair">
  <header><span class="rank">Cut {k}</span><span class="names">{fs} <span class="arrow">→</span> {ts}
  <span class="dim">at {c['montage_time']:.2f}s{beat}</span></span><span class="score">{c['total']:+.3f}</span></header>
  <div class="grid">
    {_fig(f"img/cut{k:02d}_out.jpg", f"last frame of {fs}", f"{fs} · last frame before the cut")}
    {_fig(f"img/cut{k:02d}_in.jpg", f"first frame of {ts}", f"{ts} · first frame after the cut")}
    {_fig(f"img/cut{k:02d}_out_flow.jpg", f"flow of {fs} before the cut", f"{fs} · flow in the {cfg['window']}s before the cut")}
    {_fig(f"img/cut{k:02d}_in_flow.jpg", f"flow of {ts} after the cut", f"{ts} · flow in the {cfg['window']}s after the cut")}
  </div>
  {_score_row(c, cfg)}
</article>""")
    notes = []
    if edl["unused"]:
        notes.append(f"Not used (best-connected {edl['n_used']} of {edl['n_available']}): {esc(', '.join(edl['unused']))}.")
    if edl["too_short"]:
        notes.append(f"Too short for a {cfg['seg_min']}s segment: {esc(', '.join(edl['too_short']))}.")
    return f"""
<h2>Montage</h2>
<p class="sub">{edl['n_used']} clips · {total:.1f}s · order: {esc(edl['order_method'])} · mean cut score {edl['mean_cut_score']:+.3f}
· segments {cfg['seg_min']}–{cfg['seg_max']}s (target {cfg['seg_target']}s)</p>
<video controls playsinline preload="metadata" src="../montage/montage.mp4"></video>
<div class="timeline">{timeline}</div>
{_music_html(edl, paths)}
{''.join(f'<p class="note">{n}</p>' for n in notes)}
<p class="note">Edit decision list: <a href="../montage/edl.csv">edl.csv</a> · <a href="../montage/edl.json">edl.json</a>.
Best possible cut for every ordered pair: <a href="../transitions/edges.csv">edges.csv</a> · <a href="../transitions/edges_heatmap.png">edges_heatmap.png</a>.</p>
<h3>Cuts</h3>
{''.join(cards)}"""


def _h4_card(pair: dict, reps: dict, names: dict, video: Optional[dict]) -> str:
    a, b = pair["clip_a"], pair["clip_b"]
    na, nb = esc(names[a]), esc(names[b])
    video_html = ""
    if video:
        va, vb = esc(names[video["clip_a"]]), esc(names[video["clip_b"]])
        video_html = (
            f'<video controls loop muted playsinline preload="metadata" src="../comparisons/{esc(video["file"])}"></video>'
            f'<p class="note">Preview: {va} up to {video["a_out_s"]:.2f}s, hard cut at {video["cut_at_seconds"]:.1f}s, '
            f'then {vb} from {video["b_in_s"]:.2f}s. Cut direction and points chosen by the transition score '
            f'({video["transition"]["total"]:+.3f}).</p>')
    return f"""
<article class="pair">
  <header><span class="rank">#{pair['rank']}</span>
  <span class="names" title="{esc(a)} / {esc(b)}">{na} <span class="arrow">↔</span> {nb}</span>
  <span class="score">{pair['score']:+.3f}</span></header>
  <div class="grid">
    {_fig(f"img/{a}_middle.jpg", f"middle frame of {na}", f"{na} · middle frame")}
    {_fig(f"img/{b}_middle.jpg", f"middle frame of {nb}", f"{nb} · middle frame")}
    {_fig(f"img/{a}_flow.jpg", f"mean flow of {na}", f"{na} mean flow · {_fmt_motion(reps[a])}")}
    {_fig(f"img/{b}_flow.jpg", f"mean flow of {nb}", f"{nb} mean flow · {_fmt_motion(reps[b])}")}
  </div>
  {video_html}
</article>"""


def _matrix_table(sim: np.ndarray, ids: list[str], names: dict) -> str:
    head = "".join(f'<th scope="col" title="{esc(c)}">{esc(names[c])}</th>' for c in ids)
    rows = []
    for i, ci in enumerate(ids):
        cells = []
        for j, cj in enumerate(ids):
            bg, ink = _cell_color(float(sim[i, j]))
            tip = f"{names[ci]} vs {names[cj]}: {sim[i, j]:+.4f}"
            cells.append(f'<td style="background:{bg};color:{ink}" title="{esc(tip)}">{sim[i, j]:.2f}</td>')
        rows.append(f'<tr><th scope="row" title="{esc(ci)}">{esc(names[ci])}</th>{"".join(cells)}</tr>')
    return f'<table class="matrix"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


CSS = """
:root { --bg:#fcfcfb; --surface:#ffffff; --ink:#1f1f1e; --muted:#6b6a66; --line:#e4e3df; --accent:#e34948; --seg-a:#e8e7e3; --seg-b:#d9d8d3; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --surface:#232322; --ink:#f0efec; --muted:#a3a29d; --line:#383835; --accent:#e66767; --seg-a:#383835; --seg-b:#4a4a46; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:21px; margin:44px 0 8px; } h3 { font-size:17px; margin:28px 0 10px; }
.sub, .note, figcaption, .dim { color:var(--muted); } .note { font-size:13px; margin:6px 0 0; }
.params { font:12px/1.6 ui-monospace,Menlo,monospace; color:var(--muted); background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:10px 12px; overflow-x:auto; }
.pair { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px; margin:0 0 16px; }
.pair header { display:flex; align-items:baseline; gap:12px; margin-bottom:10px; flex-wrap:wrap; }
.rank { font-weight:700; color:var(--muted); } .names { font-weight:600; flex:1; } .arrow { color:var(--muted); }
.score { font:600 18px ui-monospace,Menlo,monospace; color:var(--accent); }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
figure { margin:0; } img, video { width:100%; display:block; border-radius:6px; background:#000; }
figcaption { font-size:12px; margin-top:3px; }
.pair video { margin-top:10px; }
.timeline { display:flex; gap:2px; margin:10px 0 4px; }
.seg { min-width:0; background:var(--seg-a); border-radius:4px; padding:5px 6px; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.seg:nth-child(even) { background:var(--seg-b); } .seg span { color:var(--muted); margin-left:4px; }
.scores { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:4px 16px; margin:10px 0 0; font-size:13px; }
.scores dt { color:var(--muted); font-size:12px; } .scores dd { margin:0; font-family:ui-monospace,Menlo,monospace; }
.clips { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:12px; }
.clip img + img { margin-top:4px; }
.matrix { border-collapse:separate; border-spacing:2px; font:12px ui-monospace,Menlo,monospace; }
.matrix th { font-weight:600; color:var(--muted); padding:4px 6px; white-space:nowrap; }
.matrix td { width:58px; height:34px; text-align:center; border-radius:4px; }
.scroll { overflow-x:auto; }
.beats { position:relative; height:12px; margin:0 0 6px; border-bottom:1px solid var(--line); }
.beats i { position:absolute; bottom:0; width:1px; height:5px; background:var(--muted); }
.beats i.down { height:12px; width:2px; background:var(--accent); }
.chip { font-size:11px; padding:1px 6px; border-radius:8px; background:var(--accent); color:#fff; }
figure.wide { margin-top:10px; } figure.wide img { background:#fff; }
.legend { display:flex; gap:14px; align-items:center; } .legend img { width:72px; background:transparent; }
ul { padding-left:20px; }
@media (max-width:600px) { .grid { grid-template-columns:1fr; } }
"""


def run_report(manifest: dict, ranked: dict, paths: OutputPaths, top_k: int = 10, n_worst: int = 3) -> Path:
    out = paths.report
    img_dir = out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.jpg"):
        old.unlink()

    reps = {r.clip_id: r for r in load_reps(manifest, paths)}
    ids = [c["id"] for c in manifest["clips"]]
    names = dict(zip(ids, short_names(ids)))
    flow_cfg = json.loads((paths.motion / "config.json").read_text())
    _clip_images(manifest, reps, img_dir)
    cv2.imwrite(str(img_dir / "wheel.png"), color_wheel(96))

    edl = load_edl(paths)
    montage_html = ""
    if edl:
        feats = {f.clip_id: f for f in load_features(manifest, paths)}
        _cut_images(manifest, edl, feats, (flow_cfg["grid_w"], flow_cfg["grid_h"]), img_dir)
        montage_html = _montage_section(edl, paths)

    sim = np.load(paths.similarity / "matrix.npy")
    pairs = ranked["pairs"]
    videos = {frozenset((v["clip_a"], v["clip_b"])): v for v in load_rendered(paths)}
    top = pairs[:top_k]
    worst = pairs[-n_worst:][::-1] if len(pairs) > top_k else []
    excluded = ranked.get("excluded", [])
    cards = "\n".join(_h4_card(p, reps, names, videos.get(frozenset((p["clip_a"], p["clip_b"])))) for p in top)
    worst_cards = "\n".join(_h4_card(p, reps, names, None) for p in worst)
    excluded_html = ""
    if excluded:
        items = "".join(f"<li><b>{esc(names[e['clip_id']])}</b> — mean |v| {e['mean_magnitude']:.3f} px/frame</li>"
                        for e in excluded)
        excluded_html = (f"<h3>Excluded by the minimum-motion filter (&lt; {ranked['min_motion']} px/frame)</h3>"
                         f"<p class='note'>Their flow direction is mostly noise, so their h4 scores are not ranked. "
                         f"They can still appear in the montage, joined on composition and colour only.</p><ul>{items}</ul>")
    clip_rows = "\n".join(
        f"""<figure class="clip"><img loading="lazy" src="img/{esc(c)}_middle.jpg" alt="middle frame of {esc(names[c])}">
<img loading="lazy" src="img/{esc(c)}_flow.jpg" alt="mean flow of {esc(names[c])}">
<figcaption><b title="{esc(c)}">{esc(names[c])}</b> · {reps[c].n_pairs} h4 flow pairs · {_fmt_motion(reps[c])}</figcaption></figure>"""
        for c in ids)
    n_moving = len(ids) - len(excluded)
    coverage = (f"{n_moving} moving clips → {len(pairs)} ranked pairs. Showing the top {len(top)}"
                + (f" ({len(top) / len(pairs):.0%} of ranked pairs)" if pairs else "") + ".")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion Match Cuts</title><style>{CSS}</style></head>
<body><main>
<h1>Motion match-cut montage</h1>
<p class="sub">{len(ids)} input clips. The montage section is the product; the h4 section below is the paper-faithful pairwise analysis.</p>
<div class="params">preprocess: {esc(json.dumps(manifest['config']))}<br>flow: {esc(json.dumps(flow_cfg))}{'<br>sequence: ' + esc(json.dumps(edl['config'])) if edl else ''}</div>
{montage_html}

<h2>Pairwise motion similarity (Netflix h4)</h2>
<p class="sub">{esc(coverage)} Score = cosine similarity between whole-clip time-averaged Farnebäck flow fields (−1 opposite, 0 unrelated, +1 identical).</p>
{cards}
{excluded_html}

<h3>Similarity matrix (all clips, before filtering)</h3>
<p class="note">Hover a cell for the exact value. Also <a href="../similarity/heatmap.png">heatmap.png</a>, <a href="../similarity/matrix.csv">matrix.csv</a>, <a href="../similarity/ranked_pairs.csv">ranked_pairs.csv</a>.</p>
<div class="scroll">{_matrix_table(sim, ids, names)}</div>

{f'<h3>Lowest-scoring ranked pairs</h3><p class="note">Should look like different or opposite motion. If two clips seem to move alike but land here, h4 missed them.</p>{worst_cards}' if worst else ''}

<h2>Clips</h2>
<p class="note">Middle frame and whole-clip mean flow (the h4 representation).</p>
<div class="clips">{clip_rows}</div>

<h2>How to read the flow panels</h2>
<div class="legend"><img src="img/wheel.png" alt="flow colour wheel"><p class="note">Colour = direction pixels move, brightness = speed (each panel scaled to its own range; captions give absolute speed in px/frame at {flow_cfg['flow_width']}×{flow_cfg['flow_height']}). Arrows show local average flow. Flow describes how <i>image content</i> moves: a camera panning right makes content move left.</p></div>
<ul class="note">
<li><b>Beat sync</b>: with a soundtrack, each clip lasts a whole number of beats and every cut lands on a beat (to the nearest frame, ≤17 ms). Cuts earn a bonus by position in the bar (bar line 1.0, beat 3 0.6, beats 2/4 0.25) × beat weight.</li>
<li><b>Cut score</b> = w_motion·motion + w_comp·composition + w_color·colour. Motion = cosine of the pooled flow just before vs just after the cut × speed ratio; it counts as 0 when either side barely moves. Composition = similarity of brightness and edge layout of the two frames at the cut. Colour = Lab histogram overlap.</li>
<li>h4 is position-sensitive and dominated by camera motion (paper, Sec. 6); speed-blind (fixed in the cut score by the speed ratio).</li>
</ul>
</main></body></html>"""
    path = out / "index.html"
    path.write_text(page)
    print(f"[report] wrote {path}")
    return path

# CS180 Final Project — motion match-cut montage

Takes a folder of clips and produces **one montage video** in which each clip
cuts into the next on matching motion, composition and colour. It starts from
heuristic **h4** in Chen et al., *Match Cutting: Finding Cuts with Smooth
Visual Transitions* (WACV 2023, Sec. 3.3), and extends it to choose cut points,
clip order and segment lengths.

Not yet: music / beat sync, transitions other than hard cuts, a frontend.

## Run

```bash
conda activate cs180          # opencv-python, numpy, matplotlib; ffmpeg/ffprobe on PATH
python -m matchcut run        # data/  ->  outputs/montage/montage.mp4 + outputs/report/index.html
```

Useful options (`python -m matchcut run -h` for all):

| Option | Meaning |
|---|---|
| `--max-clips all` / `--max-clips N` | use every clip (default), or only the best-connected N |
| `--order 5601797,7587125,14306783` | fixed order (ids or short names); skips the search |
| `--seg-min 1.5 --seg-max 3.5 --seg-target 2.5` | seconds each clip stays on screen |
| `--weights 0.6 0.25 0.15` | cut score weights: motion, composition, colour |
| `--min-motion 0.05` | px/frame below which a clip/window counts as static |

To try a different montage without recomputing optical flow, re-run only the last stages:

```bash
python -m matchcut sequence --max-clips 4 && python -m matchcut render && python -m matchcut report
```

Tests (synthetic clips, no data needed; run with or without pytest):

```bash
python tests/test_synthetic.py    # h4: pans, tilts, zooms, moving objects, static
python tests/test_sequence.py     # cut scoring, ordering, max-clips, segment limits
```

## Pipeline

| # | Stage | Module | What it does | Output (under `outputs/`) |
|---|---|---|---|---|
| 1 | preprocess | `preprocess.py` | ffmpeg: centre ≤20 s, 30 fps, fit inside 640×360 with the aspect ratio kept (any leftover area padded black), no audio | `preprocessed/*.mp4`, `manifest.json` |
| 2 | motion | `motion.py` | h4 mean flow, plus a flow time series (adjacent-frame flow every 2 frames, pooled to 16×9) | `motion/<id>.npz`, `<id>_flow.jpg`, `summary.csv` |
| 3 | similarity | `similarity.py` | h4 cosine matrix; ranking with the **minimum-motion filter** | `similarity/matrix.*`, `heatmap.png`, `ranked_pairs.*` |
| 4 | transitions | `transitions.py` | features at every candidate cut point (every 0.25 s); best directed cut for every pair | `transitions/<id>.npz`, `edges.{csv,json}`, `edges_heatmap.png` |
| 5 | sequence | `sequence.py` | clip order + exact cut points → edit decision list | `montage/edl.{json,csv}` |
| 6 | render | `render.py` | the montage, plus previews of the top-5 h4 pairs | `montage/montage.mp4`, `comparisons/*.mp4` |
| 7 | report | `report.py` | HTML: montage + per-cut breakdown, h4 analysis, clips | `report/index.html` |

## How a cut is scored (departs from the paper)

h4 compares whole-clip average motion. A real cut joins one moment of clip A
(its out-point) to one moment of clip B (its in-point). For every candidate pair
of moments:

- **Motion** = cosine of the pooled flow in the 0.5 s before the out-point and the 0.5 s
  after the in-point, multiplied by the speed ratio min/max. It is **0 if either side moves
  less than `min_motion`**: no evidence either way.
- **Composition** ("h2-style" frame match for scenery) = mean of the zero-normalized
  cross-correlation (ZNCC) between the two frames at the cut, computed once on their
  brightness layouts and once on their edge layouts (32×18 each).
- **Colour** = Bhattacharyya coefficient (overlap) of the two frames' Lab colour histograms.
  It uses 8×7×7 bins, so neutral grey falls in the middle of a bin instead of on a bin edge.
- **Score** = 0.6·motion + 0.25·composition + 0.15·colour.

**Why not the paper's h2:** it relies on Mask R-CNN person masks, which needs
torch and only detects people. The scenic, city and road footage has no people,
so it would score 0 everywhere. The composition term applies the same idea
(shapes in the same place) to any footage.

**Why missing motion counts as 0 rather than rescaling the weights:** rescaling
would let static clips outscore decent motion matches. With 0, a static clip
mainly serves as a neutral bridge, which is better than cutting between
opposite motions.

## Ordering and cut placement

1. Edge weight E[A,B] = the best cut score from A to B over any cut points that leave
   each clip at least `seg_min` seconds.
2. Candidate orders:
   - ≤ 8 clips: every ordering (exact).
   - More clips: greedy from each starting clip, then local search (reverse a run, move or
     swap clips, swap in an unused clip).
   - `--max-clips N` restricts the search to N-clip paths, so the best-connected N clips win.
3. Dynamic programming places all cut points jointly for a given order. Each segment is
   `seg_min`–`seg_max` long, with a small penalty for straying from `seg_target`.
4. The shortlisted orders are compared by their exact placement score. Edge weights alone
   can mislead: a middle clip can't always use its best in-point and its best out-point
   together.

## The h4 reproduction (pairwise analysis)

For each clip:
1. Flow is computed at 256×144.
2. Farnebäck flow from f_j to f_{j+1}, for j = 0, 4, 8, …
3. Temporal mean (Eq. 5), then flatten.
4. Cosine similarity for every pair.

The official repo (github.com/Netflix/matchcut) only ships precomputed h4 scores,
not the code, so these details are our choices:

| Detail | Paper | Here |
|---|---|---|
| Shots | segmentation + dedup | each clip = one shot |
| Clip length / fps | native | centre ≤20 s, resampled to 30 fps (so px/frame is comparable) |
| Resolution | not stated | flow at 256×144; every clip must be the same size for flattened cosine |
| Aspect ratio | not addressed | fit inside the frame + black bars; flow on the bars set to 0 |
| Sampling | "every four frames" vs Eq. 5 consecutive pairs | adjacent pairs at every 4th frame (`--gap 4` for the other reading) |
| Farnebäck | not stated | OpenCV defaults (0.5, 3, 15, 3, 5, 1.2) |
| Normalization | none | none; **minimum-motion filter**: clips with mean flow < 0.05 px/frame are left out of the ranking |

## Known limitations

- Quality is bounded by the footage: a clip with no partner moving the same way
  can only get weak cuts. See `data/README.md` for what to download.
- Flow is position-sensitive (a 16×9 grid for cuts, per pixel for h4), and
  camera motion dominates over subject motion.
- There are no audio, music or beat constraints yet; segment lengths are free within limits.

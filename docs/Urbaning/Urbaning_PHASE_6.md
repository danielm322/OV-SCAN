# UrbanIng-V2X ↔ OV-SCAN Integration — Phase 6

**Status:** Investigated intermediate (BEV-feature-level) fusion, the "natural next step" both
Phase 4 §7/§9 and Phase 5 §7 flagged. **Negative result:** even in the one case where the
approach is geometrically correct by construction (`i2i`, infra-only — no rotation involved),
naive zero-training BEV-feature pooling underperforms both early fusion and late fusion
substantially, and a follow-up diagnostic shows the root cause is more fundamental than a fixable
implementation bug. Not pursued further; vehicle-involving combos (`v2v`, `v2i_*`, `full`) were
not evaluated, since the evidence from `i2i` alone made that unlikely to be worthwhile (see §6).

---

## 1. Motivation

Phases 1–3 (early fusion) concatenate raw point clouds from multiple agents before the frozen
model runs, in one shared ±54m window — structurally capped: widening the window to reach farther
sources breaks the pretrained head outright (mAP 0.28→0.007, Phase 3 §8), because it pushes input
outside the training distribution. Phases 4–5 (late fusion) run the frozen model independently
per agent, each staying inside its own valid window, then merge final 3D boxes with Weighted Box
Fusion — sidesteps the range ceiling, but trades detection quality for coverage: every late-fusion
combo underperforms its matching early-fusion config (Phase 5 §5), and down-weighting infra
sources in WBF doesn't fix this (`Urbaning_downweight_late_fusion.md`).

Both prior phases flagged **intermediate (BEV-feature) fusion** as the natural next step: keep
each agent's own point cloud inside its own valid window (avoiding early fusion's ceiling) while
fusing *feature maps* — richer than final boxes — before the detection head runs (addressing late
fusion's quality loss). This phase designed, implemented, and tested the first (zero-training)
version of that idea.

## 2. Architecture

Exploration confirmed a favorable starting point: `BaseBEVBackbone` produces a single dense 2D BEV
tensor, `spatial_features_2d` (B, 512, 180, 180 at this project's ±54m/0.075m/stride-8 config),
consumed immediately by `OVScanHead`. It's identical in extent/resolution across every
agent/config (vehicle and infra alike — verified `diff`: vehicle vs infra model/dataset configs
differ only in `_BASE_CONFIG_`'s dataset reference and `DATA_PATH`). `Detector3DTemplate.
build_networks()` makes every stage (`vfe`, `backbone_3d`, `map_to_bev_module`, `backbone_2d`,
`dense_head`) reachable as a named attribute regardless of `module_list`'s contents, so a partial
forward pass (stop before the head, resume on a fused tensor) needs no detector-class changes.

**New script:** `tools/eval_urbaning_v2x_intermediate_fusion.py`. For each unique source: run the
model up through `BACKBONE_2D` only (named-submodule calls, not `module_list` iteration) to get
that source's own BEV tensor in its own local frame. Warp each into a shared window (`torch.nn.
functional.grid_sample`, bilinear, `align_corners=False`) centered on the mean global position of
that combo's own sources at that frame (the "centroid") — chosen because late fusion's WBF
approach never defines a single shared *rasterized* frame to borrow (it operates directly on
unwindowed 3D points), and centroid-centering generalizes uniformly across every combo without an
asymmetric "pick a reference agent" convention. Pool the aligned tensors (mean or max, masked so
only cells with real per-source coverage count), feed the result into the **unmodified, frozen**
`dense_head` via a minimal synthetic `batch_dict`, and translate decoded box centers by
`+centroid` (pure translation — the window shares true global's orientation, no rotation) to get
back to true global coordinates.

One shared model is built once (not once per source): vehicle and infra configs are architecturally
identical and load the same checkpoint, so building 6 separate full model instances (each
carrying its own copy of the ~ViT-H/14 CLIP encoder) would be wasteful and risk GPU memory
pressure. The orchestration loop is frame-outer, unique-source-inner (not late fusion's
"cache every frame per source, fuse after" pattern) — a BEV tensor is ~63MB, so caching one per
source per frame for a whole sequence would be tens of GB; each frame computes every unique
source's tensor once, uses it immediately across every combo/pooling-op needing it, then discards
it before the next frame.

## 3. A real bug, caught by the planned sanity checks: decoded box semantics don't survive a spatial-only warp

`grid_sample` correctly moves *where* a feature vector sits spatially. It does **not** rotate what
that feature vector's channels *mean*. OV-SCAN's head decodes box position-offsets and yaw
relative to whatever frame the input point cloud was expressed in — confirmed independently by
late fusion's own box-transform code, which explicitly composes `R_obj_g = R @ R_obj`
(`source_to_global`) to convert decoded yaw into global. A feature-level warp moves a source's
values into the aligned window but leaves their semantic content still relative to that source's
own original heading.

Caught via the plan's own sanity checks (vehicle1 solo, 20 frames, crossing2 reference sequence):

| Path | mAP_fine |
|---|---|
| Full forward, boxes correctly globalized (ground truth) | **0.573** |
| Identity-centroid warp pipeline, no rotation correction | 0.141 |
| Same, with decoded yaw manually corrected by the source's own rotation | 0.406 |

Manually applying the same yaw-correction late fusion already uses recovers most of the lost
accuracy, confirming the diagnosis. The residual gap is likely the same issue applied to
position-offset-within-cell regression (a smaller effect, since offsets are sub-cell-sized).

**This only affects sources with nonzero rotation relative to global — i.e. vehicles.** Infra
sources have zero rotation by construction (`source_to_global`'s infra branch is translation-only:
`xyz + source['offset'], yaw` — confirmed in code, re-verified empirically below), so `i2i` has no
such bug. Fixing it in general (for `v2v`/`v2i_*`/`full`) would need transforming each source's
*raw points* into the shared frame before its own backbone pass — analogous to how early fusion
already transforms points into a reference frame, but keeping each source's forward pass
independent rather than concatenating — a substantially bigger rewrite than the current
after-the-backbone warp. Not attempted, given §4–§6's findings.

Empirical re-verification on an infra source (`crossing2_11`, 20 frames): full-forward-globalized
vs. module-split + identity-warp + single-source-pool gave mAP_fine 0.3042 vs 0.3042 (mAP_groups
0.3988 vs 0.3987) — exactly matching within floating-point/interpolation rounding, confirming the
pipeline is correct for infra by itself.

## 4. A second, more fundamental issue: the frozen head is brittle to *any* blended feature, not just misaligned ones

Running the real `i2i` combo (4 infra sources, mean *and* max pooling) crashed to near-zero even
though the single-source case is proven correct and infra has no rotation bug. The natural first
hypothesis — pooling across genuinely different viewpoints pushes the frozen head out-of-
distribution — was tested directly and **disproven**:

| Pair pooled (20 frames, crossing2) | mAP_fine |
|---|---|
| Single source (ceiling, from §3's infra check) | ~0.30 |
| `11`+`12` (two sensors mounted together, ~0m apart, nearly identical viewpoint) | 0.165 |
| `31`+`32` (two sensors mounted together, ~0m apart) | 0.089 |
| `11`+`31` (genuinely different viewpoints, ~30m+ apart) | 0.105 |

If "different viewpoints confuse the head" were the dominant effect, far-apart pairs should be
dramatically worse than colocated ones. They aren't — `31`+`32` (colocated) is actually *worse*
than the far-apart pair. **Even blending two nearly-identical feature maps from sensors sitting
next to each other already costs roughly half the solo mAP.** The more accurate diagnosis: OV-SCAN's
frozen head (heatmap conv + transformer cross-attention decoder over the full BEV map) is brittle
to *any* perturbation of its input feature map that didn't come from its own single, direct
backbone pass — not specifically to viewpoint mismatch. This is the same underlying brittleness
Phase 3 §8 already documented for widened point-cloud range, now showing up at the feature level
instead of the input level.

## 5. Full-dataset result: `i2i`, crossing2 reference sequence, 200 frames

| Fusion strategy | mAP (fine) | mAP (groups) |
|---|---|---|
| Early fusion (point concatenation, Phase 3) | 0.149 (crossing2 mean, Phase 3 §10) | 0.193 |
| Late fusion (box-level WBF, Phase 5) | 0.149 (crossing2, Phase 5 §4) | 0.263 |
| **Intermediate fusion, mean pooling (this phase)** | **0.022** | **0.040** |
| **Intermediate fusion, max pooling (this phase)** | **0.028** | **0.053** |

(IoU=0.25. Early/late fusion rows are the existing, already-published per-sequence/aggregate
numbers for the same reference sequence and combo; this phase's rows are the full 200-frame run.)

## 6. Conclusion and recommendation

**Negative result.** Even in the one case where this approach is geometrically correct by
construction (`i2i`), naive zero-training BEV-feature pooling underperforms both early fusion
(point-level) and late fusion (box-level) substantially. The root cause (§4) is not the kind of
thing a better warp, a different reference-frame convention, or a smarter pooling operator would
fix — it's that this specific frozen, never-fine-tuned head was only ever exposed to single-source,
internally-coherent feature maps during training, and has no learned tolerance for *any* blended
input, even a blend of two near-identical ones.

Given this, the vehicle-involving combos (`v2v`, `v2i_vehicle1`, `v2i_vehicle2`, `full`) were
**not evaluated** — fixing their additional rotation-semantics bug (§3) would still leave them
subject to the same fundamental blending-brittleness §4 already demonstrates in the cleanly-correct
case, so the expected payoff didn't justify the substantially larger rewrite (transforming raw
points per source before each one's own backbone pass) that a correct fix would require.

**What would actually need to change for intermediate fusion to work:** the head itself would need
to be trained (or at least fine-tuned) on blended/multi-view features — i.e., this is a training
problem, not a zero-shot inference-time design problem. That reframes "intermediate fusion" from
"the natural next zero-training step" (as Phase 4/5 characterized it) to "the natural next
*training* step" — consistent with the broader finding, now established across three independent
fusion strategies in this project, that this checkpoint's frozen head has a narrow tolerance for
anything outside its exact training-time input distribution.

## 7. File manifest

**New files:**
- `tools/eval_urbaning_v2x_intermediate_fusion.py` — the full pipeline described in §2 (shared
  model, per-source partial forward, centroid-centered warp, masked pooling, frozen-head resume).

**Modified files:**
- `tools/eval_urbaning_v2x_late_fusion.py` — extracted `gt_in_union_range` (previously inlined in
  `fuse_combo_frames`) into its own function, so the new intermediate-fusion script could reuse
  the exact same GT policy without duplicating it.

**MLflow:** new experiment `urbaning_v2x_intermediate_fusion`, runs named
`<combo>__<pool_op>__<sequence>`.

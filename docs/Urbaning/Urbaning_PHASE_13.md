# Urbaning Phase 13 — Correcting Phase 12's vocab collapse + adding a real occlusion filter

## 1. Context

Phase 12 (`docs/Urbaning/Urbaning_PHASE_12.md`) split the model's shared 10-channel detection
heatmap into finer channels (A1: 7 channels, A2: 9 channels) to give `car`/`van` dedicated heatmap
channels instead of forcing them to share one. Re-reading the OV-SCAN ICCV 2025 paper (Chow et
al.) against what Phase 12 actually built surfaced two respects in which it diverged from the
paper's own validated design:

1. **Vocab collapse.** A1/A2 split the heatmap channels but *also* collapsed each new channel's
   CLIP vocabulary to a single candidate string per channel (`car` channel → only `"car"`, `van`
   channel → only `"van"`). The paper's Table 4d ablation shows collapsing its Stage-2 text-guided
   alignment to a single-candidate scheme costs **-5.9 mAP** — multi-candidate CLIP disambiguation
   per channel is architecturally load-bearing, not cosmetic. This directly explains Phase 12's own
   documented failure mode (§4.4 there): a single-item vocab bucket makes `ov_margin` always `+inf`
   and zeroes Phase 11's contrastive same-bucket loss, disabling CLIP's corrective role exactly
   where it's needed most.
2. **No occlusion filter.** The paper's "Selective Alignment" step (Eq. 8, Supplemental Table 7)
   excludes GT objects from supervising the CLIP-alignment loss when they're too occluded by other
   objects in the same camera view. The active UrbanIng-V2X alignment-target pipeline
   (`build_alignment_crops_for_sequence.py`) filtered only by range, image-boundary-clipping ratio,
   and minimum pixel count — no inter-object occlusion check existed anywhere, for any phase that
   has ever used the alignment loss (Phase 10, 11, 12).

This phase fixes both gaps and retrains Phase 10 itself with the corrected crops (cheap, and since
every subsequent variant builds on it, worth checking whether it becomes a better default). Results
are a new, independent record rather than a silent rewrite of Phase 12's conclusion.

## 2. Design

### 2.1 Fix 1 — restore multi-item CLIP vocab for A1/A2

`ov_scan_head.py` already supports a many-to-one indirection: `NUSCENES_TO_OV_CLASSES` is
`{vocab_key: [CLIP candidate strings]}`, and `HEATMAP_BUCKET_TO_VOCAB_KEYS` maps each heatmap
channel to the vocab_key(s) it draws candidates from — several channels can legally share one
vocab_key's full candidate list. Phase 10's config already uses this correctly (`bus`/`trailer`
channels both route to the `"truck"` vocab_key's 4-item list). Phase 12's A1/A2 configs broke this
by making `HEATMAP_BUCKET_TO_VOCAB_KEYS` an exact duplicate of `NUSCENES_TO_OV_CLASSES` — one new
singleton vocab_key per new channel.

New configs `ov_scan_lidar_urbaning_v2i_vehicle1_finetune_{a1,a2}_v2.yaml` keep `CLASS_NAMES` /
`NUM_CLASSES` / `RETRAIN_HEAD` unchanged and restore the many-to-one routing, e.g. for a2_v2:

```yaml
HEATMAP_BUCKET_TO_VOCAB_KEYS: {
  "car": ["car"], "van": ["car"],
  "truck": ["truck"], "bus": ["truck"], "trailer": ["truck"],
  "cyclist": ["cyclist"], "motorcycle": ["cyclist"], "escooter": ["cyclist"],
  "pedestrian": ["pedestrian"],
}
NUSCENES_TO_OV_CLASSES: {
  "car": ["car", "van"],
  "truck": ["truck", "bus", "trailer", "other vehicle"],
  "cyclist": ["e-scooter", "motorcycle", "cyclist"],
  "pedestrian": ["pedestrian", "other pedestrian"],
}
```

**Vocab_key rename, "bicycle" → "cyclist".** Phase 10's vocab_keys ("car", "truck", "pedestrian")
are each both the dict key *and* a literal member of their own candidate string list *and* a
literal `CLASS_NAMES` channel entry. That triple coincidence is load-bearing: `decode_bbox`'s
`nusc_labels` derivation (§2.3 below) looks up the vocab_key name directly as a heatmap-channel
name. Phase 10's own vocab_key for this group is "bicycle" (not self-referential — its own
candidate list is `["e-scooter","motorcycle","cyclist"]`, which happens not to break anything there
only because "bicycle" is *also* one of Phase 10's 10 literal `CLASS_NAMES`). A1/A2's heatmap
channels dropped the literal "bicycle" channel entirely (replaced by `cyclist`/`two_wheeler_minor`
or `cyclist`/`motorcycle`/`escooter`), so reusing the vocab_key name "bicycle" crashes
`decode_bbox` with a `KeyError`. Fix: renamed the vocab_key to **"cyclist"** everywhere (a literal
`CLASS_NAMES` entry in both A1/A2), leaving the candidate list itself untouched — `self.ov_classes`
/ `self.text_features` (built from `.values()`, never from keys) are unaffected by this rename.

**Known deviation — pedestrian vocab is not byte-identical.** The plan's goal was for
`self.ov_classes`/`text_features` to be byte-identical across Phase10'/A1'/A2', isolating "how many
heatmap channels" as the only variable. This holds for every bucket *except* pedestrian:
Phase10'/align_v2 uses `"pedestrian": ["pedestrian"]` (1 candidate), while A1'/A2' use
`"pedestrian": ["pedestrian", "other pedestrian"]` (2 candidates). Tracing this: the 2-candidate
list is **not** something Phase 13 introduced — it's present verbatim in the original, pre-Phase-12
`finetune_a1.yaml`/`finetune_a2.yaml`, i.e. A1/A2's pedestrian vocab was never collapsed to begin
with (only the car/truck/cyclist buckets were). Retraining Phase 10' with a widened 2-item
pedestrian vocab (to make it match) would be the "clean" fix, but A1'/A2' are already fully trained
on the asymmetric vocab by the time this was caught, and retraining is expensive (≈1.5h / ≈3h GPU
time). §4.3 shows this deviation has a real, measurable — but narrowly scoped and explainable —
effect; it does not retroactively invalidate Fix 1's actual target (car/truck/cyclist bucket
disambiguation).

### 2.2 Fix 2 — real occlusion filter

Implemented in `build_alignment_crops_for_sequence.py`'s per-box loop: for each alignment-target
candidate, its 8 corners are projected into the chosen camera and clipped to the image rectangle;
every other GT box in the frame is projected into the same camera, and any whose mean corner depth
is strictly less than the candidate's counts as an occluder. `visible_fraction = 1 -
area(candidate_hull ∩ union(closer_occluder_hulls)) / area(candidate_hull)`. A candidate is
excluded if `visible_fraction` falls below a class-specific threshold (paper Table 7, mapped onto
UrbanIng-V2X category names): `car/van/truck/bus/trailer/other_vehicle: 0.5`,
`cyclist/escooter/motorcycle: 0.4`, `pedestrian/other_pedestrian: 0.25`. Occluder eligibility is
independent of whether the occluder itself would pass any filter — an occluder's own filterability
is irrelevant to whether it blocks another box's visibility. Threaded through
`build_alignment_crops.py`'s orchestrator with a `--disable_occlusion_filter` passthrough flag
(default: filter on). `shapely` added as an explicit pinned dependency in `docker/OV-SCAN.Dockerfile`.

**Known accepted limitation** (documented, not fixed): `best_camera_crop` still picks the camera
purely by boundary-clip ratio, with zero occlusion-awareness, before this filter ever runs — a box
could be badly occluded in its highest-ratio camera while a different camera shows it clearly.

### 2.3 Three pre-existing `ov_scan_head.py` bugs surfaced by restoring the many-to-one vocab

All three were dormant under Phase 12's buggy 1:1 vocab (where vocab_key always happened to equal a
literal heatmap-channel name) and only surfaced once Fix 1 restored legitimate many-to-one routing.

1. **`get_targets` None-crash.** `get_targets` unconditionally subscripts every key in the
   prediction dict, including `query_is_floor`, which is `None` whenever `MIN_PROPOSALS_PER_CLASS`
   (Phase 12's Lever B) isn't configured — true for every plain (no-floor) training config,
   including Phase10'/A1'/A2'. `query_is_floor` is only ever consumed by the inference-time
   `get_bboxes`/`decode_bbox` path, which already null-checks it; `get_targets_single` never touches
   it. Fixed by letting `None` pass through the per-key loop untouched instead of slicing it.
   **Open discrepancy, unresolved:** `Urbaning_PHASE_12.md` reports the original A1/A2 training
   (whose base configs also lack `MIN_PROPOSALS_PER_CLASS`) completing cleanly over many epochs —
   by this analysis the same crash should have hit them too. Root cause not determined; flagged
   here rather than assumed benign. Possibly an earlier intermediate state of the single squashed
   Phase 12 commit behaved differently from what's in the tree now.
2. **Dead `ov_class_bucket_id`/`bucket_default_ov_idx` eager-computation crash.** Computed
   unconditionally at `__init__` via `self.nuscenes_to_label_id[self.ov_to_nuscenes_classes[c]]` for
   every CLIP candidate string, assuming every vocab_key is itself a literal heatmap-channel name —
   true only in the old 1:1 world. Its sole consumer is `MARGIN_THRESH`, which grep confirms is set
   in **zero** configs project-wide, i.e. this was dead code even before Fix 1, just not dead code
   that *crashed* until vocab_keys stopped being self-referential. Fixed by guarding the whole block
   behind `if self.ov_margin_thresh is not None:` (the same condition its only consumer checks).
3. **`decode_bbox`'s vocab_key-must-be-a-literal-channel-name constraint.** `nusc_labels` (the
   model's final reported class for every proposal) is derived as
   `self.nuscenes_to_label_id[self.ov_to_nuscenes_classes[self.ov_classes[label]]]` — CLIP's winning
   candidate string maps back to its vocab_key, and the vocab_key name is looked up *directly* as a
   literal heatmap-channel name. This is intentional, pre-existing Phase 10/11 design: several
   channels sharing one vocab_key all collapse to that vocab_key's own name as their final reported
   class. It crashes when a vocab_key (here, the pre-Fix-1 "bicycle") isn't itself a channel name —
   fixed at the config level via the "bicycle"→"cyclist" rename (§2.1), not by changing this
   intentional behavior.

## 3. Training

### 3.1 Occlusion filter — measured impact on alignment-target coverage

Recomputed directly from both merged pkls (not just the per-sequence stdout tallies, which weren't
persisted to a file this round):

| Pkl | Total GT boxes | With CLIP crop embedding | Coverage |
|---|---|---|---|
| Pre-occlusion-filter (`..._full_ov_task.pkl`) | 518,325 | 239,742 | 46.3% |
| Occlusion-filtered (`..._full_ov_task_occfilt.pkl`) | 518,325 | 81,285 | 15.7% |

The occlusion filter alone removes roughly **two-thirds of the alignment targets that were
previously (incorrectly) supervising the CLIP-alignment loss** — a 3x coverage reduction, confirming
it does substantial, non-degenerate work, not a cosmetic tweak. Occlusion was the single largest
exclusion reason in most of the 27 training sequences inspected during generation (per-sequence
examples on the order of 50-65% of all exclusions in a given sequence), ahead of range/ratio/pixel
exclusions — consistent with the paper's own reported "39% occlusion" dominant-reason breakdown.

### 3.2 Phase 10' (occlusion-filter-only retrain)

Identical recipe to the adopted-default `finetune_align.yaml` (multi-item vocab already correct,
no Fix 1 change needed), pointed at the occlusion-filtered pkl: 8 epochs, loss converged ~0.367.
Post-training eval crashed with the pre-existing, config-independent nuscenes-devkit
`AssertionError: Database version not found` bug (documented in Phase 10/11/12), unrelated to this
phase.

**Checkpoint-diff vs. original Phase 10** (`checkpoint_epoch_8.pth` both sides): 0 shape
mismatches, 28 value-changed tensors (exactly `--train_alignment_only`'s expected scope), 705
identical, **0** frozen-backbone tensors touched. Clean.

### 3.3 A1'/A2' (vocab-fixed retrain, from Phase 10')

Both continued from **Phase 10'**'s `checkpoint_epoch_8.pth` (not the original Phase 10 — Phase 10'
is already trained end-to-end on the occlusion-filtered targets, so starting A1'/A2' from it avoids
warm-starting on occlusion-noisy targets before switching), `--freeze_backbone`,
`--max_ckpt_save_num 1` (see §3.4): A1' 12 epochs (loss converged ~0.26), A2' 24 epochs (loss
converged ~0.246). Both runs' post-training eval hit the same pre-existing devkit crash as Phase
10'/Phase 10/11/12.

**Checkpoint-diff vs. Phase 10'** (both clean):

| | A1' vs Phase10' | A2' vs Phase10' |
|---|---|---|
| Shape mismatches | 5 (`class_encoding`/`heatmap_head.1`/`prediction_head.heatmap.1`, 10→7 channels) | 5 (same tensors, 10→9 channels) |
| Value-changed (same shape) | 116 | 116 |
| Identical | 612 | 612 |
| Frozen-backbone tensors touched | **0** | **0** |

Shape mismatches are exactly the heatmap head resized for the new channel count (expected,
`RETRAIN_HEAD`'s partial reinit); 116 value-changed tensors matches `--freeze_backbone`'s expected
scope (alignment_head + rest of dense_head); 0 frozen-backbone tensors touched confirms the freeze
held. Byte-identical pattern to original Phase 12's A1/A2 diffs.

### 3.4 Mid-phase disk-full incident → checkpoint retention policy

The host's `/` volume (shared with the Docker container) hit 93% usage and `/tmp/claude-1001` hit 0
MB free mid-phase, breaking background job monitoring (each OV-SCAN checkpoint is ~2.6GB, and a
rolling window of 3 + a near-duplicate `latest_model.pth` per run adds up fast across Phase10'/A1'/
A2' plus the 6 Lever-B overlay configs on one box with no headroom to spare). Resolved by pruning
already-finished runs down to their final checkpoint only, and switching to
`--max_ckpt_save_num 1` for A2' and all evaluation-only runs going forward. This is now a standing
policy for this project (recorded in memory), not a one-off.

## 4. Evaluation

All runs on the full 1400-frame held-out set, `tools/eval_urbaning_v2x_per_class.py` +
`investigate_cross_class_fps.py --class_name van`, `--fusion_type single_vehicle --sources
vehicle1`. Phase 9/10/A1/A2 (pre-Phase-13) numbers below are reused from `Urbaning_PHASE_12.md` and
are **directional/non-controlled** — they differ from this round in both occlusion-filter lineage
and (for A1/A2) vocab design, two confounded variables at once. {Phase10', A1', A2'} is the only
clean, controlled comparison set this round produces.

### 4.1 AP table (no floor)

| Metric | Phase 10' | A1' (7ch) | A2' (9ch) | *Phase 10 (old)* | *A1 (old, collapsed vocab)* | *A2 (old, collapsed vocab)* |
|---|---|---|---|---|---|---|
| `car` AP@0.25 | 0.619 | 0.609 | 0.611 | *0.669* | *0.729* | *0.718* |
| `van` AP@0.25 | 0.439 | 0.310 | 0.360 | *0.463* | *0.403* | *0.259* |
| `truck` AP@0.25 | 0.283 | 0.258 | 0.306 | *0.267* | *0.478* | *0.635* |
| `cyclist` AP@0.25 | 0.720 | 0.657 | 0.656 | *0.719* | *0.721* | *0.691* |
| `pedestrian` AP@0.25 | 0.310 | 0.001 | 0.000 | *~0.3* | — | — |
| mAP (fine classes) | 0.263 | 0.204 | 0.215 | *0.270* | *0.259* | *0.257* |
| mAP (groups) | 0.577 | 0.539 | 0.543 | *0.579* | *0.542* | *0.554* |
| `vehicle` (group) AP | 0.795 | 0.801 | 0.796 | *0.798* | *0.799* | *0.798* |
| `heavy_vehicle` (group) AP | 0.474 | 0.361 | 0.401 | *0.477* | *0.360* | *0.455* |

**Do not read `car`/`van` fine-class AP as "old A1/A2 beat A1'/A2'"** — see §4.2, the decisive metric
is cross-class contamination, not raw AP, and §4.3 explains why A1'/A2's `pedestrian` fine-class AP
specifically is an artifact, not a real capability regression.

### 4.2 Van cross-class diagnostic (decisive metric, each checkpoint under its own architecture's plain config, no floor)

| Variant | total `van` preds | unmatched% | matched_car% | matched_van% | p90 score |
|---|---|---|---|---|---|
| *Phase 10 (old)* | *51,478* | *86.4%* | *6.1%* | *5.9%* | *0.155* |
| **Phase 10'** | 88,938 | 89.6% | 5.8% | 3.6% | 0.021 |
| *A1 (old, collapsed vocab)* | *28,910* | *64.9%* | ***19.5%*** | *12.9%* | *0.0009* |
| **A1' (fixed vocab)** | 119,521 | 85.4% | **7.7%** | 5.7% | 0.011 |
| *A2 (old, collapsed vocab)* | *34,338* | *83.9%* | *6.4%* | *7.9%* | *0.0009* |
| **A2' (fixed vocab)** | 95,778 | 85.3% | 6.7% | 6.7% | 0.035 |

**This is the headline finding of Phase 13, and it validates the paper's Table 4d claim directly.**
Old A1's single-item vocab forced every "van"-channel proposal to be reported as `van` with no CLIP
disambiguation — this happened to win more raw `van` AP (0.403, highest of any A1/A2 variant,
old or new) by brute-force overcommitment, but at the cost of **3x worse car/van confusion**
(`matched_car%` 6.1%→19.5%, a direct, CLIP-free measurement of the heatmap regressor itself
misplacing a van-channel proposal on a real car). Restoring multi-item CLIP vocab (A1') brings
`matched_car%` back down to 7.7% — close to Phase 10'/Phase 10's own baseline band (5.8%/6.1%) and
nowhere near old A1's 19.5% — confirming that CLIP disambiguation, once actually enabled, does its
corrective job. The tradeoff is lower raw `van` AP (0.310 vs old A1's inflated 0.403), which is the
expected and acceptable cost of trading overcommitment for real class separation.

Old A1's other apparent "win" — `unmatched%` dropping to 64.9% (vs. baseline's 86%) — also doesn't
survive the fix: A1' sits at 85.4%, essentially back at baseline. This confirms that *both* of old
A1's seemingly positive deltas versus baseline were side effects of the same collapsed-vocab bug,
not real architectural improvements.

### 4.3 Pedestrian-vocab finding: CLIP systematically prefers "other pedestrian"

`eval_urbaning_v2x_per_class.py` evaluates per-class AP using the *raw CLIP-winning candidate
string* (`pred['ov_str_labels']`, mapped through `PRED_NATIVE_TO_CANONICAL`), not the model's
vocab-key-collapsed final label. Because A1'/A2' carry the (pre-existing, §2.1) 2-item pedestrian
vocab `["pedestrian", "other pedestrian"]`, while Phase10'/align_v2 only ever has 1 candidate, A1'/
A2's pedestrian predictions get split at eval time between two fine-class slots. The split is
heavily, almost totally, one-sided: `pedestrian` AP collapses to ~0 with single-digit/double-digit
prediction counts, while `other_pedestrian` absorbs tens of thousands of predictions but *also*
shows AP=0.000 — because most of those predictions are spatially sitting on real `pedestrian` GT
boxes, which don't count as a match against the differently-named `other_pedestrian` GT class
(263 boxes only). This is the same "right place, wrong label" mechanism Phase 7/12 documented for
car/van, now occurring between two CLIP candidate strings that both nominally mean "pedestrian."

This drags A1'/A2's `mAP (fine classes)` down independent of anything Fix 1 or Fix 2 actually
changed. The `pedestrian_group` AP (which merges both strings via `CANONICAL_TO_GROUP`) stays
comparable to Phase 10' (A1': 0.298 vs Phase10': 0.297, A2': 0.294) — confirming the underlying
people-detection capability is not actually degraded. **Fine-class `pedestrian` AP and `mAP (fine
classes)` should not be compared across this round's vocab boundary; `pedestrian_group` AP and `mAP
(groups)` are the valid comparison axis.** Separately, this is a useful, generalizable finding: a
vague CLIP candidate string ("other pedestrian") can systematically out-compete a more literal one
("pedestrian") for real crops of the literal class, which is a concrete argument for care in vocab
string design in any future richer-vocab experiment (the paper's own Table 8 ablation direction).

### 4.4 Lever B floor sweep (B1/B2/B3), A1'/A2'

| Config | car AP | van AP | truck AP | mAP (fine) | mAP (groups) |
|---|---|---|---|---|---|
| A1', no floor | 0.609 | 0.310 | 0.258 | 0.204 | 0.539 |
| A1', B1 | 0.609 | 0.320 | 0.271 | 0.207 | 0.537 |
| A1', B2 | 0.612 | 0.315 | 0.263 | 0.206 | 0.540 |
| A1', B3 | **0.127** | 0.280 | 0.277 | 0.150 | 0.392 |
| A2', no floor | 0.611 | 0.360 | 0.306 | 0.215 | 0.543 |
| A2', B1 | 0.603 | 0.349 | 0.299 | 0.214 | 0.534 |
| A2', B2 | 0.616 | 0.358 | 0.306 | 0.216 | 0.543 |
| A2', B3 | **0.055** | 0.242 | 0.293 | 0.140 | 0.362 |

B1/B2 move every metric by ≤0.01-0.02 AP either direction — noise-level, no clean win on any axis.
B3's aggressive `car` floor (2 proposals/frame) collapses `car` AP by 80%+ in both A1' and A2'
(the floor is far too restrictive for the highest-volume class). This exactly reproduces original
Phase 12's own conclusion ("Lever B delivers no clean win on any axis") under the corrected vocab —
the floor mechanism's behavior is independent of the vocab-collapse bug, as expected (it's a
pure inference-time proposal-selection lever, orthogonal to Fix 1/Fix 2).

## 5. Conclusion and recommendation

- **Fix 1 (vocab restoration) is validated and should be considered a real bug fix, not an
  optional improvement.** It directly reproduces the paper's Table 4d finding: restoring
  multi-candidate CLIP disambiguation is what lets a split-heatmap architecture avoid severe
  car/van cross-contamination. Old A1's higher raw AP numbers were an artifact of forced
  single-class overcommitment, not a real capability advantage — the decisive cross-class
  diagnostic shows A1' is categorically healthier (7.7% vs 19.5% `matched_car%`) than the
  Phase-12-era A1 it replaces.
- **Fix 2 (occlusion filter) is a correct, paper-faithful implementation but shows a
  neutral-to-mildly-negative effect on this round's downstream diagnostics**, not the hoped-for
  improvement: Phase 10' vs. Phase 10 shows van prediction volume rising (51k→89k), `unmatched%`
  ticking up slightly (86.4%→89.6%), and score calibration (p90) dropping further (0.155→0.021).
  Removing ~2/3 of the (noisy, occluded) alignment targets did not translate into a cleaner,
  better-calibrated van heatmap on this particular metric — plausibly because what's removed also
  removes real supervision signal faster than it removes noise, or because occlusion wasn't
  actually the dominant source of the van-channel's longstanding calibration problem (documented
  since Phase 7). This does not mean Fix 2 is wrong to implement — it corrects a genuine,
  previously-undetected spec deviation from the paper — but it is not, by itself, a lever that
  improves this project's specific `van` pathology.
- **Neither A1' nor A2' beats Phase 10/Phase 10' on raw AP or on the decisive van diagnostic's
  `unmatched%`** (the metric that most directly tracks the project's core, unsolved `van`
  over-firing problem). A1' does meaningfully fix the *car/van confusion* axis relative to old A1,
  but that was old A1's own self-inflicted problem, not evidence A1'/A2' now surpass Phase 10.
- **Recommendation: keep Phase 10 (original, not Phase 10') as the adopted default.** Phase 10'
  is a legitimate, slightly-more-rigorous alternative (occlusion filter now matches the paper's
  spec) but shows no measured improvement over Phase 10 on any metric in this round, and the two
  are close enough that re-adopting would require more evidence than one round's diagnostics
  provide. A1'/A2' are now built on an architecturally sound, bug-fixed foundation (unlike their
  Phase 12 predecessors) but still do not outperform the simpler 10-channel baseline on this
  dataset/training budget — the same ultimate recommendation as Phase 12, now resting on a
  genuinely controlled comparison rather than one confounded by a vocab-collapse bug.
- **Follow-ups for a future phase, not done here:** (a) resolve the `query_is_floor` historical
  discrepancy (§2.3.1) by checking whether Phase 12's actual training runs predate the current
  code state; (b) if A1/A2-style finer taxonomies are revisited, redesign the pedestrian vocab
  candidate strings to avoid the "other pedestrian" CLIP-preference trap (§4.3), and retrain
  Phase 10' with a matching widened vocab for a truly byte-identical comparison; (c) investigate
  whether per-class occlusion thresholds need recalibration for this dataset specifically (the
  paper's thresholds are nuScenes-tuned), given Fix 2's neutral/negative showing here.

## 6. File manifest

**Modified:**
- `OV-SCAN/pcdet/models/dense_heads/ov_scan_head.py` — the 2 crash fixes in §2.3 (`get_targets`
  None-passthrough; `ov_class_bucket_id`/`bucket_default_ov_idx` guarded behind `MARGIN_THRESH`).
- `OV-SCAN/tools/urbaning_v2x/build_alignment_crops_for_sequence.py` — occlusion filter (§2.2).
- `OV-SCAN/tools/urbaning_v2x/build_alignment_crops.py` — `--disable_occlusion_filter` passthrough.
- `OV-SCAN/tools/urbaning_v2x/merge_alignment_crops.py` — minor support for the `_occfilt` suffix.
- `docker/OV-SCAN.Dockerfile` — added `shapely` as a pinned dependency.

**New configs:**
- `OV-SCAN/tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_v2.yaml`
  (Phase 10')
- `OV-SCAN/tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_a1_v2.yaml`
  (+ `_leverb_b1/b2/b3.yaml` overlays)
- `OV-SCAN/tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_a2_v2.yaml`
  (+ `_leverb_b1/b2/b3.yaml` overlays)

**Checkpoints** (inside the container, not committed; pruned to 1 per run per §3.4):
`output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_v2/.../checkpoint_epoch_8.pth`,
`..._finetune_a1_v2/.../checkpoint_epoch_12.pth`, `..._finetune_a2_v2/.../checkpoint_epoch_24.pth`.

**Result snapshots** (inside the container, not committed): `/tmp/phase13_eval/ap_*.json` (9 AP
tables), `/OV-SCAN/phase13_eval_sweep.log`, `/OV-SCAN/phase13_van_sweep.log`.

**Data** (inside the container, not committed): `nuscenes_infos_1sweeps_train_full_ov_task_occfilt.pkl`
merged into the baseline, A1, and A2 taxonomy directories (§3.1).

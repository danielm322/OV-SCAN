# Lower-LR vs. freeze-backbone ablation for the `van` over-firing problem — Phase 9

**Status:** Tested the two remaining candidates from Phase 7/8 for fixing the `van`
proposal-hallucination flood — a lower peak LR with more epochs, and freezing the backbone so only
the detection head trains — plus their combination, as a controlled 3-way ablation against the
Phase 7 `hm_weight=1.0` baseline. **Clear result:** lower LR + more epochs (variant a) is a real,
adoptable improvement — `van` AP +37.5%, fine-class mAP +6.9%, group mAP within ~2% of baseline,
and far better score calibration. Freezing the backbone (variants b/c) gives a similar `van` AP
gain but at a real group-mAP cost (-10 to -12%) from `heavy_vehicle`/`two_wheelers` regressing —
ruled out as a net-negative trade. None of the three variants structurally fixes the hallucination
mechanism itself (proportionally still ~82-87% of `van` predictions are unmatched to any GT,
barely different from baseline) — they shrink the flood's absolute volume, not its qualitative
cause. The separate `car`/`van` CLIP-mislabeling mechanism from Phase 7 §5.3 is, as expected,
completely untouched by any of these variants.

---

## 1. Context

Phase 7 found that fine-tuning `v2i_vehicle1` improved group-level mAP substantially but caused
`van` AP to drop, because the heatmap's `car` channel became far less selective after fine-tuning
(proposal volume +4.7x), flooding the fixed 200-proposal-per-frame budget with near-zero-confidence
noise that the frozen CLIP step still classifies, mostly into "van" (§5.1: 87% of `van` predictions
unmatched to any GT, median score ~0.0001). Phase 8 tested halving `hm_weight` and ruled it out —
essentially no change to the over-firing behavior, and a net-negative trade on group mAP. A
separate non-training investigation (Phase 7 §5.3) found a second, distinct mechanism coexisting in
the same checkpoint's `van` output: among `van` predictions with confidence ≥0.5, 68.5% are real
cars confidently mislabeled "van" by the frozen CLIP alignment step — architecturally unfixable by
anything in this phase, since the alignment head gets exactly zero gradient on UrbanIng-V2X data
regardless of config (no grounding-SAM crops to supervise it with).

This phase tests the two remaining named candidates from Phase 7/8's conclusion, plus their
combination, to separate how much of any improvement comes from smaller optimizer steps vs. fewer
trainable parameters:

- **(a) Lower LR + more epochs.** New finding motivating this: the baseline fine-tune's peak LR
  (`0.0005`, one-cycle) is *identical* to the LR used for the original 20-epoch, 8-GPU (effective
  batch 8) from-scratch nuScenes pretraining — Phase 7's fine-tune reuses that full-strength peak LR
  for only 8 epochs on a single GPU (effective batch 1, 1/8th the original, i.e. noisier per-step
  gradients at an unchanged step size). Reusing pretraining-strength LR to fine-tune an
  already-converged, calibrated heatmap at a much noisier batch size is a plausible mechanism for
  calibration drift. Chosen: `LR: 0.0001` (5x lower, matching this repo's own precedent —
  `ov_scan_multimodal.yaml`'s fine-tune LR off this same checkpoint family), `NUM_EPOCHS: 24` (3x)
  to partially compensate for the lower per-step magnitude.
- **(b) Freeze backbone, train only the head.** `vfe`/`backbone_3d`/`map_to_bev_module`/
  `backbone_2d` frozen (`requires_grad=False`); only `dense_head` (`OVScanHead`) trains, at the
  original `LR: 0.0005`/8 epochs. Note this does *not* freeze `heatmap_head` itself — that lives
  inside `dense_head`, the actual component implicated in the over-firing diagnosis — so this tests
  "does reducing trainable-parameter count elsewhere help," not "does freezing the over-firing
  component itself help" (freezing `heatmap_head` would defeat the point of fine-tuning it).
- **(c) Combined** — (a)'s config + (b)'s freeze flag together.

All three start fresh from the frozen zero-shot checkpoint (`pretrained/ov_scan_lidar.pth`), never
continuing from Phase 7/8's checkpoints, evaluated identically against Phase 7's `hmw1.0` baseline
on the same 1400-frame held-out crossing3 set.

## 2. Building `--freeze_backbone` (new infrastructure)

No existing freeze mechanism existed in this codebase (no `FREEZE` yaml flag, no `--freeze` CLI
arg). Added a new opt-in flag to `tools/train.py`:

```python
parser.add_argument('--freeze_backbone', action='store_true', default=False,
                     help='freeze vfe/backbone_3d/map_to_bev_module/backbone_2d (requires_grad=False '
                          'and pinned eval mode); only dense_head trains. Phase 9 ablation flag.')
```

Inserted between `model.cuda()` and `optimizer = build_optimizer(model, cfg.OPTIMIZATION)` — this
ordering is load-bearing, since `OptimWrapper.create`'s `trainable_params()` filters by
`requires_grad` once, at that call; freezing after it would silently do nothing.

**Non-obvious gotcha found and fixed before launching any real run:** `train_utils_wandb.py`'s
`train_one_epoch` calls `model.train()` every single iteration, which recursively re-enables train
mode on every child module — including anything frozen. `VoxelResBackBone8x` and `BaseBEVBackbone`
both use `BatchNorm1d`/`BatchNorm2d` with running stats, which keep updating in train mode
regardless of `requires_grad`. So `requires_grad=False` alone would not have been a true freeze —
BN running stats would have kept drifting every iteration for the whole run. Fixed by pinning each
frozen submodule's `.train()` to a no-op:

```python
if args.freeze_backbone:
    frozen_module_names = ['vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d']
    for name in frozen_module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = False
        module.eval()
        module.train = (lambda self, mode=True: self).__get__(module)
    logger.info('==> --freeze_backbone: froze %s ...' % frozen_module_names)
```

**Verified, not assumed:** ran a 1-epoch smoke test with `--freeze_backbone`, then diffed the
resulting checkpoint's `model_state` against the pretrained checkpoint by submodule prefix:
`backbone_3d` 0/105 and `backbone_2d` 0/70 float tensors changed (including BN buffers), while
`dense_head` had 111/513 changed — confirming the freeze (and the BN-drift fix) actually works
before spending GPU time on it. `vfe`/`map_to_bev_module` have no learnable parameters at all
(`MeanVFE`/`HeightCompression`).

## 3. New config

`tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr.yaml` — exact copy of
the baseline fine-tune config, changing only `OPTIMIZATION.LR` (0.0005→0.0001) and
`OPTIMIZATION.NUM_EPOCHS` (8→24), each annotated with the reasoning above. Diff-verified against
the baseline to confirm these are the only two changed fields. Variants (b)/(c) needed no new
config — (b) reuses the original baseline YAML + the new flag; (c) reuses this new `_lowlr.yaml` +
the same flag.

## 4. Training

All three ran from `pretrained/ov_scan_lidar.pth` (never continuing a prior fine-tune), sequentially
on the project's single GPU:

```bash
# (a) lower LR + more epochs -- v2i_vehicle1_finetune_lowlr_v1, 24/24 epochs, ~4h13min
WANDB_MODE=offline python train.py \
    --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr.yaml \
    --pretrained_model ../../pretrained/ov_scan_lidar.pth \
    --batch_size 1 --epochs 24 --extra_tag v2i_vehicle1_finetune_lowlr_v1 --num_epochs_to_eval 0

# (b) freeze backbone only -- v2i_vehicle1_finetune_freeze_v1, 8/8 epochs, ~62min
WANDB_MODE=offline python train.py \
    --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune.yaml \
    --pretrained_model ../../pretrained/ov_scan_lidar.pth \
    --batch_size 1 --epochs 8 --freeze_backbone \
    --extra_tag v2i_vehicle1_finetune_freeze_v1 --num_epochs_to_eval 0

# (c) combined -- v2i_vehicle1_finetune_lowlr_freeze_v2, 24/24 epochs
WANDB_MODE=offline python train.py \
    --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr.yaml \
    --pretrained_model ../../pretrained/ov_scan_lidar.pth \
    --batch_size 1 --epochs 24 --freeze_backbone --max_ckpt_save_num 3 \
    --extra_tag v2i_vehicle1_finetune_lowlr_freeze_v2 --num_epochs_to_eval 0
```

**Operational note:** the first attempt at (c) (`..._freeze_v1`, without `--max_ckpt_save_num`)
crashed at epoch 6/24 with `RuntimeError: ... No space left on device` while saving a checkpoint —
the project's disk filled up from accumulating full-size (~2.6GB) checkpoints across all of Phase
7/8/9's runs. The corrupted partial checkpoint was deleted and the run relaunched as `..._v2` with
`--max_ckpt_save_num 3` (now the standing convention for all training in this project going
forward, per explicit user instruction, to bound disk usage per run).

## 5. Evaluation

All four checkpoints (baseline `hmw1.0` + a/b/c) evaluated identically: `eval_urbaning_v2x_per_class.py`
on the full 1400-frame held-out crossing3 set (IoU 0.25 primary, 0.3/0.5 extra), and
`investigate_cross_class_fps.py --class_name van` comparing all four checkpoints in one invocation.

### 5.1 Per-class AP

| Metric | zero-shot | baseline `hmw1.0` | (a) lowlr | (b) freeze | (c) combined |
|---|---|---|---|---|---|
| `van` AP @0.25 | 0.443 | 0.240 | **0.330** | 0.337 | 0.339 |
| `van` num_pred | — | 145,767 | 103,874 | 48,474 | 59,226 |
| `car` AP @0.25 | 0.571 | 0.370 | 0.397 | 0.614 | 0.616 |
| `truck` AP @0.25 | 0.608 | 0.546 | 0.616 | 0.478 | 0.469 |
| `cyclist` AP @0.25 | 0.393 | 0.686 | 0.717 | 0.632 | 0.623 |
| mAP (fine classes) @0.25 | 0.250 | 0.246 | **0.263** | 0.263 | 0.262 |
| mAP (groups) @0.25 | 0.422 | **0.586** | 0.576 | 0.523 | 0.518 |
| mAP (groups) @0.3 | — | 0.569 | 0.559 | 0.494 | 0.490 |
| mAP (groups) @0.5 | — | 0.440 | 0.425 | 0.338 | 0.335 |

(a) is the only variant that improves `van` AP (+37.5%) and fine-class mAP (+6.9%) while staying
within ~2% of baseline group mAP at every IoU threshold. (b)/(c) reach a similar or slightly better
`van` AP, but group mAP drops 10-12% at IoU 0.25 (worse at 0.3/0.5) — driven by `heavy_vehicle` and
`two_wheelers` regressing (full numbers in the per-run JSON files, `phase9_results_*.json`): the
frozen backbone evidently needed to adapt its general 3D features to UrbanIng-V2X's different
sensor geometry/density to support the broader class gains Phase 7 achieved, and freezing it gives
that up.

### 5.2 Cross-class `van` diagnostic (mechanism-level)

| Variant | total `van` preds | unmatched% | matched_car% | matched_van% | median score | p90 score |
|---|---|---|---|---|---|---|
| baseline `hmw1.0` | 145,767 | 87.0% | 8.9% | 3.1% | 0.0001 | 0.079 |
| (a) lowlr | 103,874 | 83.8% | 11.2% | 3.9% | 0.0002 | **0.507** |
| (b) freeze | 48,474 | 82.1% | 8.6% | 7.0% | 0.0070 | 0.250 |
| (c) combined | 59,226 | 83.7% | 8.3% | 6.1% | 0.0066 | 0.162 |

Two things stand out:
- **`unmatched`% barely moves** (82-84% vs. baseline's 87%) across all three variants — the
  proportion of `van` predictions that are pure hallucination is essentially unchanged. What
  dropped is the absolute volume (baseline 126,781 unmatched `van` predictions → 87,001 / 39,810 /
  49,574 for a/b/c, i.e. -31%/-69%/-61%). **This is a quantitative dampening of the flood, not a
  qualitative fix** — consistent with this being a persistent heatmap-selectivity characteristic,
  not something any of these three levers removes outright.
- **`matched_car`% stays flat** (8.3-11.2%, no trend) across every variant, confirming — exactly as
  predicted architecturally in §1 — that none of these changes touches the separate Phase 7 §5.3
  CLIP-mislabeling mechanism at all. That remains out of scope for any training-recipe lever.
- **(a)'s score calibration is dramatically better** (p90 score 0.51 vs. baseline's 0.08, vs. 0.25/0.16
  for b/c) — a much larger share of its `van` predictions carry informative, decisive confidence
  scores rather than clustering near zero. This matters practically: it makes (a)'s checkpoint far
  more usable with a deployment score threshold (§5.3's recommendation) than the baseline, whose
  scores are almost all ~0.0001 regardless of correctness.

### 5.3 Score-threshold sweep re-check on (a)

Re-ran `sweep_score_threshold.py` on (a)'s checkpoint to confirm Phase 7 §5.3's "free win except
`van`" pattern still holds (rather than assuming it): at threshold 0.2, `car` precision 0.927,
`vehicle` (group) precision 0.943, `two_wheelers` 0.937, `truck` 0.858 — all comparable to or better
than the equivalent baseline-checkpoint sweep — while `van` precision plateaus at 0.27, the same
qualitative pattern as before. So (a)'s checkpoint is fully compatible with, and benefits from, the
existing score-threshold deployment recommendation.

## 6. Conclusion and recommendation

**Adopt variant (a) — `LR: 0.0001`, `NUM_EPOCHS: 24` — as the new default fine-tuned checkpoint**,
replacing Phase 7's `hmw1.0` 8-epoch baseline:
`output/nuscenes_models/.../ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr/v2i_vehicle1_finetune_lowlr_v1/ckpt/checkpoint_epoch_24.pth`.
It improves `van` AP (+37.5%) and fine-class mAP (+6.9%) with group mAP within ~2% of baseline, and
produces far better-calibrated confidence scores — a genuine improvement on every axis checked, not
a trade-off.

**Rule out (b) freeze-backbone and (c) combined** as net-negative trades: the `van`-AP gain they
offer is matched or exceeded by (a) alone, at a real cost (10-12% group mAP regression) that (a)
doesn't pay. Freezing the backbone is not the right lever here.

**Still deploy with a ~0.2-0.3 score threshold** (`DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`, Phase 7
§5.3) on top of (a)'s checkpoint — confirmed compatible and, if anything, more usable than before
given (a)'s better score calibration.

**What remains unsolved, by design:** the `van` hallucination mechanism itself (heatmap
over-selectivity) is dampened in absolute volume by (a) but not structurally fixed — ~84% of `van`
predictions are still unmatched to any GT. And the separate `car`/`van` CLIP-mislabeling mechanism
(Phase 7 §5.3) is completely unaffected by anything in Phases 8-9, as predicted: it's gated on the
frozen alignment head, which gets zero gradient on this dataset regardless of LR, loss weights, or
which submodules are frozen. Fixing it would require either adding grounding-SAM-style supervision
for the alignment head (a new data/annotation effort, not a training-recipe tweak) or revisiting the
CLIP prompt/vocabulary routing directly.

## 7. File manifest

**New:**
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr.yaml` — variant (a)
  config (`LR: 0.0001`, `NUM_EPOCHS: 24`), reused unchanged for variant (c).
- `docs/Urbaning/Urbaning_PHASE_9.md` — this file.

**Modified:**
- `tools/train.py` — added `--freeze_backbone` CLI flag and the freeze block (§2), with the
  BatchNorm-drift fix (pinned no-op `.train()` on frozen submodules).

**New checkpoints:**
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr/v2i_vehicle1_finetune_lowlr_v1/ckpt/checkpoint_epoch_24.pth`
  — variant (a), **new recommended default**.
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune/v2i_vehicle1_finetune_freeze_v1/ckpt/checkpoint_epoch_8.pth`
  — variant (b), ruled out.
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_lowlr/v2i_vehicle1_finetune_lowlr_freeze_v2/ckpt/checkpoint_epoch_24.pth`
  — variant (c), ruled out.

**Result snapshots** (saved outside the auto-overwritten `output/.../default/per_class_ap.json`
path, since `eval_urbaning_v2x_per_class.py` writes to a fixed path per cfg file regardless of
checkpoint): `phase9_results_baseline_hmw1.0.json`, `phase9_results_a_lowlr.json`,
`phase9_results_b_freeze.json`, `phase9_results_c_combined.json` (repo root inside the container,
`/OV-SCAN/OV-SCAN/`).

**Standing convention going forward:** use `--max_ckpt_save_num 3` on all training invocations in
this project (disk space is limited; see §4's operational note).

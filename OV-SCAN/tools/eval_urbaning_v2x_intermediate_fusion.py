"""
Intermediate (BEV-feature-level) fusion for UrbanIng-V2X -- Phase 6's first evaluation. See
docs/Urbaning/Urbaning_PHASE_6.md for the full design and results.

Early fusion (Phase 1-3) concatenates raw point clouds before the model runs, in one shared +-54m
window -- structurally capped (Phase 3 Sec.8). Late fusion (Phase 4-5) runs the frozen model
independently per agent, then merges final 3D boxes with WBF -- sidesteps the range ceiling but
trades detection quality for coverage (every late-fusion combo underperforms its matching
early-fusion config). This script tests a third strategy, zero training like the other two: run
each agent's own point cloud through the SAME frozen backbone up to (but not including) the
detection head, spatially align the resulting BEV feature maps into one shared frame, pool them
(mean or max), and feed the pooled result into the unmodified, frozen head -- richer than final
boxes (addressing late fusion's quality loss), while each agent's own input never leaves the
distribution the checkpoint was trained on (addressing early fusion's range ceiling).

Architecture (verified against the actual model code before implementing):
  - Detector3DTemplate.build_networks() add_module()'s every module_topology name as an attribute
    (model.vfe, model.backbone_3d, model.map_to_bev_module, model.backbone_2d, model.dense_head)
    regardless of what ends up in module_list -- so each stage can be called directly, independent
    of OVScanLidar.forward()'s own module_list loop.
  - BaseBEVBackbone.forward() produces batch_dict['spatial_features_2d'] (B,512,180,180 at this
    project's +-54m/0.075m/stride-8 config), consumed immediately by OVScanHead.forward(), whose
    eval-mode path reads ONLY that one key from batch_dict (verified: no other upstream state is
    needed to resume at the head with a synthetic carrier dict).
  - Vehicle and infra model/dataset configs are byte-identical except for which dataset config
    they _BASE_CONFIG_ from (vehicle vs infra) and DATA_PATH (verified via `diff`) -- so the model
    built from any one source's config is architecturally and, once this same checkpoint is
    loaded, numerically identical to one built from any other. Building 6 separate full model
    instances (each carrying its own ~ViT-H/14 CLIP encoder) for what's really one frozen set of
    weights would be wasteful and risk GPU memory pressure -- this script builds ONE shared model
    and reuses it for every source's partial forward pass; only each source's own dataset (for its
    own point cloud) differs.
  - Frame-outer, unique-source-inner loop (NOT late fusion's "cache every frame then fuse"
    pattern): a spatial_features_2d tensor is ~63MB, so caching one per unique source per frame
    for a whole sequence would be tens of GB -- infeasible. Each frame computes every unique
    source's BEV tensor once (shared across every combo needing it), discards it at the end of
    that frame's iteration, and keeps only the resulting (tiny) boxes.
  - The fused BEV window's own reference frame is centered on the mean global position of that
    combo's own participating sources at that frame (the "centroid"), with the SAME orientation as
    true global (no rotation) -- late fusion's union-of-sources-own-window GT policy and box
    transforms don't define a single shared frame to borrow (WBF operates directly on unwindowed
    3D points), so this is a new, explicit choice: it generalizes uniformly across every combo
    (no asymmetric "pick a reference agent" convention needed) and makes mapping decoded boxes
    back to true global a pure translation (+centroid, no rotation).

Usage (inside the OV-SCAN container):
    xvfb-run -a python eval_urbaning_v2x_intermediate_fusion.py \
        --root_folder ../datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --data_root ../datasets/urbaning_v2x_late_fusion_batch/20241126_0001_crossing2_00 \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from mlflow_logging import get_or_create_run
from eval_urbaning_v2x_per_class import PRED_NATIVE_TO_CANONICAL
from eval_urbaning_v2x_late_fusion import (
    VEHICLE_CFG, combo_sources_for, unique_sources, load_gt_by_timestamp, load_source_frame_index,
    load_vehicle_poses, load_infra_offset, source_to_local, gt_in_union_range, compute_ap_sweep,
)

POOL_OPS = ['mean', 'max']


def build_shared_model(ckpt, logger):
    """Builds the OV-SCAN model ONCE, from the (arbitrary -- see module docstring) vehicle config,
    shared across every source's partial forward pass. Only each source's own dataset (built
    separately in build_source) differs."""
    cfg_from_yaml_file(VEHICLE_CFG, cfg)
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    return model


def build_source(source, logger, expected_point_cloud_range=None):
    """Builds this source's own dataset (for its own point cloud/DATA_PATH) and loads
    pose/offset/token metadata into `source` in place -- mirrors run_source_inference in
    eval_urbaning_v2x_late_fusion.py, minus building its own model (every source shares
    build_shared_model's single instance -- see module docstring)."""
    cfg_from_yaml_file(source['cfg_file'], cfg)
    cfg.DATA_CONFIG.DATA_PATH = source['data_path']
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False)

    dataset_root = Path(cfg.ROOT_DIR) / 'tools' / cfg.DATA_CONFIG.DATA_PATH
    version = cfg.DATA_CONFIG.VERSION
    if source['mode'] == 'vehicle':
        source['poses'] = load_vehicle_poses(dataset_root, version)
    else:
        source['offset'] = load_infra_offset(dataset_root, version)
    source['ordered_tokens'] = load_source_frame_index(dataset_root, version)

    source['test_set'] = test_set
    source['point_cloud_range'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    if expected_point_cloud_range is not None:
        assert list(source['point_cloud_range']) == list(expected_point_cloud_range), (
            f"source {source['label']}'s POINT_CLOUD_RANGE {source['point_cloud_range']} differs "
            f"from {expected_point_cloud_range} -- every config in this project is supposed to "
            f"share one window; the fixed-size output grid this script builds assumes that.")
    return source


def source_origin_global(source, token):
    """This source's own local-frame origin (0,0,0), in true global coordinates -- vehicle: the
    ego_pose+calibrated_sensor translation; infra: the static recentering offset. Used to place a
    combo's fused BEV window (see combo_centroid)."""
    if source['mode'] == 'vehicle':
        return source['poses'][token][1]
    return source['offset']


def combo_centroid(combo_sources, token_by_label):
    """Mean global position of a combo's own participating sources at the current frame -- where
    the fused BEV window is centered (see module docstring for why)."""
    origins = [source_origin_global(s, token_by_label[s['label']]) for s in combo_sources]
    return np.mean(origins, axis=0)


def bev_features_for_frame(model, source, idx):
    """Runs the shared model up through BACKBONE_2D only (VFE -> BACKBONE_3D -> MAP_TO_BEV ->
    BACKBONE_2D), stopping before DENSE_HEAD, on `source`'s frame `idx`. Named-submodule calls --
    not module_list iteration/counting -- since every module_topology name is reachable as an
    attribute regardless of module_list's contents (see module docstring). Returns
    (spatial_features_2d, token)."""
    data_dict = source['test_set'].collate_batch([source['test_set'][idx]])
    load_data_to_gpu(data_dict)
    token = data_dict['metadata'][0]['token']
    with torch.no_grad():
        data_dict = model.vfe(data_dict)
        data_dict = model.backbone_3d(data_dict)
        data_dict = model.map_to_bev_module(data_dict)
        data_dict = model.backbone_2d(data_dict)
    return data_dict['spatial_features_2d'], token


def output_grid_offsets_m(point_cloud_range, grid_size):
    """(grid_size, grid_size, 2) array of each output BEV cell's physical (x,y) offset in meters
    from the window's own center (0,0) -- pixel-center convention (cell i's center sits at
    x_min + (i+0.5)*cell_size), the same +0.5 pixel-center convention OVScanHead's own
    create_2D_grid uses (ov_scan_head.py:368-379). Axis layout matches the confirmed (H,W) =
    (y-axis, x-axis) orientation running through VoxelResBackBone8x -> HeightCompression ->
    BaseBEVBackbone (no transpose anywhere in that chain)."""
    x_min, y_min, x_max, y_max = point_cloud_range[0], point_cloud_range[1], point_cloud_range[3], point_cloud_range[4]
    cell_x = (x_max - x_min) / grid_size
    cell_y = (y_max - y_min) / grid_size
    i = np.arange(grid_size)
    x = x_min + (i + 0.5) * cell_x
    y = y_min + (i + 0.5) * cell_y
    xx, yy = np.meshgrid(x, y, indexing='xy')  # (H,W): xx varies along columns (W=x), yy along rows (H=y)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def warp_source_to_window(bev_feat, source, token, centroid_xyz, grid_offsets_m, point_cloud_range):
    """Samples `bev_feat` (this source's own local BEV grid, shape (1,C,H,W)) into the shared
    output window centered at `centroid_xyz` (true global). For each output cell, the physical
    global position (centroid + grid_offsets_m) is mapped into this source's own local frame via
    source_to_local -- the exact inverse of source_to_global, already used and tested across every
    late-fusion box transform -- then converted to a normalized grid_sample coordinate
    (align_corners=False: x_min/x_max map to the outermost pixels' OUTER edges, -1/+1, matching
    grid_offsets_m's own pixel-center convention consistently). Returns (warped_feat, valid_mask):
    mask is an exact, analytic in/out-of-window check (not grid_sample's own zero-padding), so
    masked pooling never conflates "really zero" with "outside this source's own window"."""
    H, W = grid_offsets_m.shape[0], grid_offsets_m.shape[1]
    global_xy = centroid_xyz[:2][None, None, :] + grid_offsets_m  # (H,W,2)
    global_xyz = np.concatenate([global_xy, np.zeros((H, W, 1), dtype=np.float32)], axis=-1)
    local_xyz = source_to_local(source, token, global_xyz.reshape(-1, 3))
    local_xy = local_xyz[:, :2].reshape(H, W, 2)

    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    x_norm = 2 * (local_xy[..., 0] - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (local_xy[..., 1] - y_min) / (y_max - y_min) - 1
    valid = (x_norm >= -1) & (x_norm <= 1) & (y_norm >= -1) & (y_norm <= 1)

    grid = torch.from_numpy(np.stack([x_norm, y_norm], axis=-1)).float().to(bev_feat.device)[None]
    warped = F.grid_sample(bev_feat, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
    mask = torch.from_numpy(valid).to(bev_feat.device)[None, None].float()
    return warped, mask


def masked_pool(warped_list, mask_list, op):
    """Cell-wise pooling across sources, counting only cells where that source has real coverage
    (mask==1) -- avoids biasing cells near a source's own window edge toward zero from
    grid_sample's zero-padded out-of-bounds regions."""
    feats = torch.cat(warped_list, dim=0)  # (S,C,H,W)
    masks = torch.cat(mask_list, dim=0)    # (S,1,H,W)
    if op == 'mean':
        denom = masks.sum(dim=0, keepdim=True).clamp(min=1.0)
        fused = (feats * masks).sum(dim=0, keepdim=True) / denom
    elif op == 'max':
        neg_inf = torch.finfo(feats.dtype).min
        masked_feats = torch.where(masks.bool(), feats, torch.full_like(feats, neg_inf))
        fused = masked_feats.max(dim=0, keepdim=True).values
        any_valid = masks.sum(dim=0, keepdim=True) > 0
        fused = torch.where(any_valid.bool(), fused, torch.zeros_like(fused))
    else:
        raise ValueError(f'unknown pool op: {op}')
    return fused  # (1,C,H,W)


def decode_boxes(model, fused_feat):
    """Feeds `fused_feat` into the UNMODIFIED, frozen dense_head (same module, same weights) via a
    minimal synthetic batch_dict -- OVScanHead.forward()'s eval-mode path reads only
    batch_dict['spatial_features_2d'], and OVScanLidar.post_processing()-style consumers need only
    'batch_size' in addition (verified against both files; no other upstream state required).
    Returns the raw pred dict (same keys run_source_inference already extracts elsewhere:
    pred_boxes/pred_scores/ov_str_labels)."""
    batch_dict = {'spatial_features_2d': fused_feat, 'batch_size': 1}
    with torch.no_grad():
        batch_dict = model.dense_head(batch_dict)
    return batch_dict['final_box_dicts'][0]


def build_intermediate_fusion_frames(combos, root_folder, sequence, ckpt, logger, max_samples=None,
                                      pool_ops=POOL_OPS):
    """Runs the full frame-outer, unique-source-inner pipeline (see module docstring) for every
    combo in `combos`, for every pooling op in `pool_ops`. Returns {(combo_name, pool_op): frames}
    in exactly the structure class_ap/compute_ap_sweep expect."""
    sources = unique_sources(combos)
    model = build_shared_model(ckpt, logger)
    point_cloud_range = None
    for source in sources:
        build_source(source, logger, expected_point_cloud_range=point_cloud_range)
        point_cloud_range = source['point_cloud_range']
    logger.info(f'Shared model built once; {len(sources)} unique source(s): '
                f'{[s["label"] for s in sources]} (covering {len(combos)} combo(s))')

    seq_dir = Path(root_folder) / 'dataset' / sequence
    time_sync = pd.read_csv(seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')
    columns = list(time_sync.columns)
    gt_by_ts = load_gt_by_timestamp(Path(root_folder) / 'labels' / f'{sequence}.json')
    num_frames = len(columns) if max_samples is None else min(max_samples, len(columns))

    frames_by_combo_pool = {(combo_name, op): [] for combo_name in combos for op in pool_ops}
    grid_offsets_m = None

    for idx in range(num_frames):
        ts_ms = int(time_sync[columns[idx]]['timestamp_ms'])

        bev_by_label, token_by_label = {}, {}
        for source in sources:
            bev_feat, token = bev_features_for_frame(model, source, idx)
            bev_by_label[source['label']] = bev_feat
            token_by_label[source['label']] = token
            if grid_offsets_m is None:
                grid_size = bev_feat.shape[-2]
                assert bev_feat.shape[-1] == grid_size, 'expected a square BEV grid'
                grid_offsets_m = output_grid_offsets_m(point_cloud_range, grid_size)

        for combo_name, combo_sources_list in combos.items():
            centroid = combo_centroid(combo_sources_list, token_by_label)
            warped, masks = [], []
            for source in combo_sources_list:
                w, m = warp_source_to_window(
                    bev_by_label[source['label']], source, token_by_label[source['label']],
                    centroid, grid_offsets_m, point_cloud_range)
                warped.append(w)
                masks.append(m)

            gt_boxes, gt_class = gt_in_union_range(combo_sources_list, idx, gt_by_ts, ts_ms, point_cloud_range)

            for op in pool_ops:
                fused_feat = masked_pool(warped, masks, op)
                pred = decode_boxes(model, fused_feat)
                pred_boxes = pred['pred_boxes'].cpu().numpy()
                pred_scores = pred['pred_scores'].cpu().numpy()
                pred_class = [PRED_NATIVE_TO_CANONICAL.get(s) for s in pred['ov_str_labels']]
                if len(pred_boxes) > 0:
                    pred_boxes = pred_boxes.copy()
                    pred_boxes[:, :3] += centroid[None, :]  # output window shares true global's
                    # orientation (no rotation) -- pure translation back, see module docstring.
                frames_by_combo_pool[(combo_name, op)].append({
                    'pred_boxes': pred_boxes, 'pred_scores': pred_scores, 'pred_class': pred_class,
                    'gt_boxes': gt_boxes, 'gt_class': gt_class,
                })
        logger.info(f'[{idx + 1}/{num_frames}] processed {len(sources)} unique source(s), '
                    f'{len(combos)} combo(s) x {len(pool_ops)} pooling op(s)')
    return frames_by_combo_pool


def parse_args():
    parser = argparse.ArgumentParser(
        description='Intermediate (BEV-feature-level) fusion for UrbanIng-V2X -- zero-training pooling')
    parser.add_argument('--root_folder', type=str, required=True,
                         help='dir containing dataset/<sequence>/timesync_info.csv and labels/<sequence>.json')
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True,
                         help='dir containing <data_root>/vehicle1_solo, vehicle2_solo, '
                              'infra_solo_<channel> -- converted solo datasets for this sequence '
                              '(the same ones late fusion already uses)')
    parser.add_argument('--intersection', type=str, default=None,
                         help='default: derived from --sequence (crossing<N>)')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--match_iou_thresh', type=float, default=0.25,
                         help='primary AP-matching IoU threshold (unsuffixed metric names)')
    parser.add_argument('--extra_match_iou_threshs', type=str, default='0.3,0.5',
                         help='comma-separated additional AP-matching IoU thresholds, swept from '
                              'the same pooled result. Empty string to disable.')
    parser.add_argument('--pool_ops', type=str, default='mean,max')
    parser.add_argument('--only_combo', type=str, default=None,
                         choices=['i2i_late', 'v2v_late', 'v2i_v1_late', 'v2i_v2_late', 'full_late'],
                         help='restrict to one combo (for quick testing)')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None,
                         help='default: output/urbaning_v2x_intermediate_fusion/<sequence>/')
    parser.add_argument('--mlflow_experiment', type=str, default='urbaning_v2x_intermediate_fusion')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    intersection = args.intersection or re.search(r'crossing\d+', args.sequence).group(0)
    logger.info(f'----------------- UrbanIng-V2X intermediate (BEV) fusion: {args.sequence} ({intersection}) -----------------')

    combos = combo_sources_for(intersection, args.data_root)
    if args.only_combo:
        combos = {args.only_combo: combos[args.only_combo]}
    pool_ops = [p.strip() for p in args.pool_ops.split(',') if p.strip()]

    frames_by_combo_pool = build_intermediate_fusion_frames(
        combos, args.root_folder, args.sequence, args.ckpt, logger, args.max_samples, pool_ops)

    extra_threshs = [float(x) for x in args.extra_match_iou_threshs.split(',') if x.strip()]
    save_dir = Path(args.save_dir) if args.save_dir is not None else \
        Path(cfg.ROOT_DIR) / 'output' / 'urbaning_v2x_intermediate_fusion' / args.sequence

    for (combo_name, pool_op), frames in frames_by_combo_pool.items():
        logger.info(f'--- combo: {combo_name} / pool_op: {pool_op} ---')
        summary, mlflow_metrics = compute_ap_sweep(frames, args.match_iou_thresh, extra_threshs, logger)
        summary['combo'] = combo_name
        summary['pool_op'] = pool_op
        summary['sequence'] = args.sequence

        combo_save_dir = save_dir / combo_name / pool_op
        combo_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = combo_save_dir / 'per_class_ap.json'
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        run_name = f'{combo_name}__{pool_op}__{args.sequence}'
        run = get_or_create_run(cfg.ROOT_DIR, args.mlflow_experiment, run_name, tags={
            'fusion_type': combo_name, 'pool_op': pool_op, 'sequence': args.sequence,
            'intersection': intersection, 'merge_strategy': 'intermediate_bev',
        })
        logger.info(f'MLflow run: {args.mlflow_experiment}/{run_name} ({run.info.run_id})')
        mlflow.log_params({
            'combo': combo_name, 'pool_op': pool_op, 'sequence': args.sequence,
            'intersection': intersection, 'merge_strategy': 'intermediate_bev', 'ckpt': args.ckpt,
            'match_iou_thresh': args.match_iou_thresh, 'extra_match_iou_threshs': args.extra_match_iou_threshs,
        })
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(save_path))
        logger.info(f"[{combo_name}/{pool_op}] mAP_fine={summary['mAP_fine_classes']:.4f} "
                    f"mAP_groups={summary['mAP_groups']:.4f} (wrote {save_path})")
        mlflow.end_run()

    logger.info('Done.')


if __name__ == '__main__':
    main()

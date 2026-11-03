"""Runs INSIDE the OV-SCAN container (needs torch/open_clip/numpy). For one already-extracted
UrbanIng-V2X sequence, projects every train-split GT box (already built into the v2i_vehicle1
pkl, in vehicle1's own raw LiDAR sensor frame) into vehicle1's 6 RGB cameras, picks the
best-visible camera per box, crops it, encodes the crop with the exact same frozen CLIP model
OV-SCAN's alignment head uses for its text embeddings, and saves one L2-normalized embedding
per matched box -- i.e. builds this project's own `gt_grounding_sam` entries from scratch,
skipping the Grounding-SAM detection-matching step the original nuScenes pipeline
(pcdet/models/model_utils/selective_alignment_nuscenes_utils.py) depends on but that was never
run for UrbanIng-V2X and has no generator script in this repo -- we crop directly from each GT
box's own projected corners instead of matching against an external detector's output.

Math/conventions deliberately mirror AlignmentSelector (selective_alignment_nuscenes_utils.py)
wherever there's a 1:1 analog (Box corner convention, view_points-style projection, box-in-image
ratio, margin-then-clip crop expansion) so this produces embeddings in the same convention the
dense head's loss (pcdet/models/dense_heads/ov_scan_head.py) already expects -- just sourced from
UrbanIng-V2X's own raw calibration instead of nuScenes'.

Two load-bearing gotchas, confirmed by inspecting a raw calibration.json directly:
  1. `IntrinsicMatrix` is stored in MATLAB Camera Calibration Toolbox's transposed convention
     [[fx, 0, 0], [0, fy, 0], [cx, cy, 1]] (row-vector-times-matrix projection), NOT the standard
     OpenCV/pinhole column form. It must be TRANSPOSED before use with cv2.projectPoints or any
     standard column-vector pinhole math. Likewise `ImageSize` is [height, width] (MATLAB
     convention), not [width, height].
  2. vehicle1's 6 cameras do NOT all share one lens/distortion model: `front_left`/`front_right`/
     `back_left`/`back_right` report `RadialDistortion`(3)+`TangentialDistortion`(2) (standard
     Brown-Conrady pinhole, handled with cv2.projectPoints), but `left`/`right` report a 4-value
     `DistortionCoefficients` instead -- MATLAB's fisheye calibration model (equidistant/
     Kannala-Brandt-style, same convention as cv2.fisheye), consistent with side cameras using a
     much wider-angle lens. Treating `left`/`right` as pinhole would silently project their boxes
     onto the wrong pixels, especially near the image edges where these two models diverge most.

Usage (inside the container):
    python urbaning_v2x/build_alignment_crops_for_sequence.py \
        --seq_dir /OV-SCAN/datasets/urbaning_v2x_raw/dataset/<sequence> \
        --sequence <sequence> \
        --pkl_path /OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_multiseq/v1.0-custom/nuscenes_infos_1sweeps_train.pkl \
        --save_root /OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_multiseq/ov_alignment/DFN5B-CLIP-ViT-H-14/object_crops/train \
        --out_path /OV-SCAN/datasets/urbaning_v2x_raw/<sequence>_gt_grounding_sam.pkl
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

CAMERAS = ['front_left', 'front_right', 'left', 'right', 'back_left', 'back_right']


def load_calibration(seq_dir):
    with open(seq_dir / 'calibration.json') as f:
        calib = json.load(f)

    vTl = np.asarray(calib['vehicle1_middle_lidar']['extrinsics']['vTl'])

    cams = {}
    for cam in CAMERAS:
        entry = calib[f'vehicle1_{cam}_camera']
        cTv = np.asarray(entry['extrinsics']['cTv'])
        # MATLAB IntrinsicMatrix convention -- see module docstring. Transpose to get the
        # standard column-vector pinhole K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
        K = np.asarray(entry['intrinsics']['IntrinsicMatrix']).T
        height, width = entry['intrinsics']['ImageSize']  # MATLAB convention: [height, width]
        intr = entry['intrinsics']
        if 'DistortionCoefficients' in intr:
            model = 'fisheye'
            dist = np.asarray(intr['DistortionCoefficients'], dtype=np.float64)  # [k1,k2,k3,k4]
        else:
            model = 'pinhole'
            radial = intr['RadialDistortion']  # [k1, k2] or [k1, k2, k3]
            tangential = intr['TangentialDistortion']  # [p1, p2]
            k3 = radial[2] if len(radial) > 2 else 0.0
            dist = np.array([radial[0], radial[1], tangential[0], tangential[1], k3], dtype=np.float64)
        cams[cam] = {
            'cTl': cTv @ vTl,  # L (box/point frame) -> this camera's frame
            'model': model,
            'K': K,
            'dist': dist,
            'width': width,
            'height': height,
        }
    return cams


def box_corners_in_L(box):
    """box: pcdet 7-float gt_boxes row [x, y, z, dx, dy, dz, heading]. Returns (8, 3) corners in
    the L frame, using the same wlh=(dy,dx,dz) reindex and corner layout as AlignmentSelector.Box
    (nuScenes convention: x forward/length, y left/width, z up/height)."""
    x, y, z, dx, dy, dz, heading = box[:7]
    w, l, h = dy, dx, dz
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ corners
    corners += np.array([[x], [y], [z]])
    return corners.T  # (8, 3)


def project_box(corners_L, cam):
    """Projects (8,3) box corners (L frame) into one camera. Returns (px (8,2), depth (8,))."""
    corners_h = np.concatenate([corners_L, np.ones((8, 1))], axis=1)  # (8,4)
    corners_cam = (cam['cTl'] @ corners_h.T).T[:, :3]  # (8,3), camera frame
    depth = corners_cam[:, 2]
    if cam['model'] == 'fisheye':
        # cv2.fisheye is strict about input shape/dtype: (1, N, 3) float64 object points, D as
        # (4,) or (4,1). Points behind the camera aren't meaningful under the equidistant fisheye
        # model either, so depth is still checked by the caller exactly as for pinhole.
        img_pts, _ = cv2.fisheye.projectPoints(
            corners_cam.reshape(1, -1, 3).astype(np.float64), np.zeros(3), np.zeros(3),
            cam['K'], cam['dist'])
    else:
        img_pts, _ = cv2.projectPoints(
            corners_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam['K'], cam['dist'])
    return img_pts.reshape(-1, 2), depth


def box_in_image_ratio(px, cam):
    min_x, min_y = px[:, 0].min(), px[:, 1].min()
    max_x, max_y = px[:, 0].max(), px[:, 1].max()
    box_area = (max_x - min_x) * (max_y - min_y)
    if box_area <= 0:
        return 0.0, (min_x, min_y, max_x, max_y)
    inter_min_x, inter_max_x = max(min_x, 0), min(max_x, cam['width'])
    inter_min_y, inter_max_y = max(min_y, 0), min(max_y, cam['height'])
    if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
        return 0.0, (min_x, min_y, max_x, max_y)
    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    return inter_area / box_area, (min_x, min_y, max_x, max_y)


def is_any_corner_visible(px, depth, cam):
    visible = (px[:, 0] > 0) & (px[:, 0] < cam['width']) & (px[:, 1] > 0) & (px[:, 1] < cam['height']) & (depth > 1)
    in_front = depth > 0.1
    return bool(visible.any()) and bool(in_front.all())


def best_camera_crop(corners_L, cams, min_box_in_image_ratio):
    best = None
    for cam_name, cam in cams.items():
        px, depth = project_box(corners_L, cam)
        if not is_any_corner_visible(px, depth, cam):
            continue
        ratio, (min_x, min_y, max_x, max_y) = box_in_image_ratio(px, cam)
        if best is None or ratio > best['ratio']:
            best = {'cam_name': cam_name, 'ratio': ratio, 'box': (min_x, min_y, max_x, max_y)}
    if best is None or best['ratio'] <= min_box_in_image_ratio:
        return None
    min_x, min_y, max_x, max_y = best['box']
    cam = cams[best['cam_name']]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(cam['width'], max_x), min(cam['height'], max_y)
    if min_x >= max_x or min_y >= max_y:
        return None
    return best['cam_name'], (min_x, min_y, max_x, max_y), best['ratio']


def apply_margin_and_clip(box, margin, width, height):
    min_x, min_y, max_x, max_y = box
    bw, bh = max_x - min_x, max_y - min_y
    min_x -= bw * margin
    max_x += bw * margin
    min_y -= bh * margin
    max_y += bh * margin
    min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
    max_x, max_y = min(width, int(np.ceil(max_x))), min(height, int(np.ceil(max_y)))
    return min_x, min_y, max_x, max_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--pkl_path', required=True)
    parser.add_argument('--save_root', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--clip_model', default='hf-hub:apple/DFN5B-CLIP-ViT-H-14')
    parser.add_argument('--min_box_in_image_ratio', type=float, default=0.3)
    parser.add_argument('--image_crop_margin', type=float, default=0.1)
    parser.add_argument('--min_pixels', type=int, default=4000)
    parser.add_argument('--max_alignment_range', type=float, default=54.0)
    parser.add_argument('--debug_crops_dir', type=str, default=None,
                         help='if set, also save the first --debug_crops_limit matched crops as '
                              'PNGs here, for visual spot-checking the projection math')
    parser.add_argument('--debug_crops_limit', type=int, default=20)
    args = parser.parse_args()
    debug_crops_dir = Path(args.debug_crops_dir) if args.debug_crops_dir else None
    if debug_crops_dir:
        debug_crops_dir.mkdir(parents=True, exist_ok=True)
    num_debug_saved = 0

    seq_dir = Path(args.seq_dir)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    cams = load_calibration(seq_dir)
    time_sync = pd.read_csv(seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')
    ts_to_col = {int(float(time_sync.loc['timestamp_ms', col])): col for col in time_sync.columns}

    with open(args.pkl_path, 'rb') as f:
        infos = pickle.load(f)

    pattern = re.compile(rf'^samples/LIDAR_TOP/{re.escape(args.sequence)}_fused_vehicle1_(\d+)\.pcd\.bin$')
    seq_infos = []
    for info in infos:
        m = pattern.match(info['lidar_path'])
        if m:
            seq_infos.append((info, int(m.group(1))))
    print(f'[{args.sequence}] {len(seq_infos)} frames found in pkl')

    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_model)
    clip_model.eval().cuda()

    results = {}
    num_boxes, num_matched = 0, 0
    for info, ts_ms in seq_infos:
        col = ts_to_col.get(ts_ms)
        if col is None:
            results[info['token']] = [None] * len(info['gt_boxes'])
            continue
        row = time_sync[col]
        entries = []
        for box in info['gt_boxes']:
            num_boxes += 1
            if np.linalg.norm(box[:2]) >= args.max_alignment_range:
                entries.append(None)
                continue
            corners_L = box_corners_in_L(box)
            match = best_camera_crop(corners_L, cams, args.min_box_in_image_ratio)
            if match is None:
                entries.append(None)
                continue
            cam_name, crop_box, ratio = match
            cam = cams[cam_name]
            min_x, min_y, max_x, max_y = apply_margin_and_clip(
                crop_box, args.image_crop_margin, cam['width'], cam['height'])
            if (max_x - min_x) * (max_y - min_y) < args.min_pixels:
                entries.append(None)
                continue

            cam_channel = f'vehicle1_{cam_name}_camera'
            img_filename = row[cam_channel]
            img_path = seq_dir / cam_channel / img_filename
            image = np.array(Image.open(img_path))[min_y:max_y, min_x:max_x]

            if debug_crops_dir and num_debug_saved < args.debug_crops_limit:
                gt_name = info['gt_names'][len(entries)]
                debug_path = debug_crops_dir / f'{args.sequence}_{num_boxes}_{gt_name}_{cam_name}.png'
                Image.fromarray(image).save(debug_path)
                num_debug_saved += 1

            preprocessed = clip_preprocess(Image.fromarray(image)).unsqueeze(0).cuda()
            with torch.no_grad():
                embedding = clip_model.encode_image(preprocessed)
                embedding = embedding / embedding.norm(dim=1, keepdim=True)
                embedding = embedding.squeeze().cpu()
            save_path = save_root / f'object_crop_{args.sequence}_{num_boxes}.pt'
            torch.save(embedding, save_path)

            entries.append({
                'object_embedding_path': str(save_path),
                'pred_corners': ((min_x, min_y), None, (max_x, max_y), None),
                'cam_id': cam_channel,
                'box_in_image_ratio': ratio,
            })
            num_matched += 1
        results[info['token']] = entries

    with open(args.out_path, 'wb') as f:
        pickle.dump(results, f)

    coverage = num_matched / num_boxes if num_boxes else 0.0
    print(f'[{args.sequence}] {num_matched}/{num_boxes} boxes matched ({coverage:.1%} coverage), '
          f'saved to {args.out_path}')


if __name__ == '__main__':
    main()

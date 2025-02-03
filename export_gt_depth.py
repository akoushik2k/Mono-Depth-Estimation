# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map
from skimage.transform import resize


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))
    
    target_shape = (375, 1242)

    gt_depths = []
    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder, "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth", "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        # Resize depth map and ensure consistency
        gt_depth_resized = resize(gt_depth, target_shape, preserve_range=True, anti_aliasing=True)

        print(f"Depth map shape before resize: {gt_depth.shape}, after resize: {gt_depth_resized.shape}")

        gt_depths.append(gt_depth_resized.astype(np.float32))

    # Check all depth maps for consistency
    for depth in gt_depths:
        assert depth.shape == target_shape, f"Inconsistent depth map shape: {depth.shape}"

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))
    print(output_path)

    # Save depth maps as compressed .npz
    np.savez_compressed(output_path, data=np.array(gt_depths))



if __name__ == "__main__":
    export_gt_depths_kitti()

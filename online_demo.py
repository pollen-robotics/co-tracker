# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.utils.visualizer import Visualizer

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

cv2.namedWindow("first_frame")

points = []


def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(x, y)


# callback function
cv2.setMouseCallback("first_frame", on_mouse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    first_frame = cap.read()[1]
    while True:
        im = first_frame.copy()
        for point in points:
            im = cv2.circle(im, point, 3, (0, 0, 255), -1)
        cv2.imshow("first_frame", im)
        key = cv2.waitKey(1)
        if key == 27:
            break

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(
        window_frames, is_first_step, grid_size, grid_query_frame, queries
    ):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            queries=torch.Tensor(queries).to(DEFAULT_DEVICE),
            grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    times = []
    start = time.time()
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        s = time.time()
        if i % model.step == 0 and i != 0:
            queries = np.zeros((1, len(points), 3))
            for point in points:
                queries[0, points.index(point), :] = (i, point[0], point[1])
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=0,
                queries=queries,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
        times.append(time.time() - s)
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    queries = np.zeros((1, 1, 3))
    queries[0, 0, :] = (i, 100, 100)
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=0,
        queries=queries,
        grid_query_frame=args.grid_query_frame,
    )
    print("Time elapsed: ", time.time() - start, " seconds")
    fps = len(times) / (time.time() - start)
    print("FPS: ", fps)
    # plt.plot(np.arange(len(times)), times)
    # plt.show()
    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3, fps=15)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )

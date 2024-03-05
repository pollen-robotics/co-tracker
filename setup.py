# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="cotracker",
    version="2.0",
    install_requires=[
        "torch==2.2.1",
        "torchvision==0.17.1",
        "imageio==2.34.0",
        "imageio-ffmpeg==0.4.9",
        "matplotlib==3.8.3",
        "opencv-python==4.9.0.80",
    ],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib"],
        "dev": ["flake8", "black"],
    },
)

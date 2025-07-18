#!/usr/bin/env python3
"""
Author:
    Lachlan Mares, lachlan.mares@adelaide.edu.au

License:
    GPL-3.0

Description:

"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
import numpy as np
import torch
import torch.jit
import torch.nn.functional as F
from torchvision.transforms import Compose

from src import DepthAnything
from src import Resize, NormalizeImage, PrepareForNet


class DepthAnythingClass:
    def __init__(self, config_settings: dict):
        # Set device
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = torch.device(config_settings["cuda_device"])
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.depth_anything_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(config_settings["encoder"]),
                                                                  local_files_only=True).to(self.device).eval()
        self.height = config_settings["image_height"]
        self.width = config_settings["image_width"]
        self.infer_height = config_settings["infer_height"]
        self.infer_width = int((((self.width / self.height) * self.infer_height) // 14) * 14)

        self.transform = Compose([
            Resize(
                width=self.infer_height,
                height=self.infer_height,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    @torch.inference_mode()
    def normalise(self, depth: np.array) -> np.array:
        return (depth - depth.min()) / (depth.max() - depth.min())

    @torch.inference_mode()
    def infer(self, image: np.array, normalise: bool = True):
        image_transform = self.transform({'image': image / 255})['image']
        image_transform = torch.from_numpy(image_transform).unsqueeze(0).to(self.device)
        depth = self.depth_anything_model(image_transform)
        depth = F.interpolate(depth[None], (self.height, self.width),
                                       mode='bilinear', align_corners=False)[0, 0]
        if normalise:
            depth = self.normalise(depth=depth)

        return depth.cpu().numpy()


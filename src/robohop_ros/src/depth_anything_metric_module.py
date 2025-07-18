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
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(str(Path(__file__).parents[0] / "depth_anything/metric_depth"))

import cv2
import torch
import threading
import numpy as np
import torchvision.transforms as transforms

from pathlib import Path
from src import get_config, build_model


class DepthAnythingMetricClass:
    def __init__(self, config_settings: dict):
        # Set device
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = torch.device(config_settings["cuda_device"])
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        config = get_config(config_settings["model_name"], "infer", None)
        config.pretrained_resource = "local::" + str(Path(__file__).parents[0] / config_settings["pretrained_resource"])

        self.depth_anything_model = build_model(config).to(self.device).eval()

        self.read_lock = threading.Lock()

        self.pointcloud_dtype = [('x', np.float32),
                                 ('y', np.float32),
                                 ('z', np.float32),
                                 ('r', np.float32),
                                 ('g', np.float32),
                                 ('b', np.float32)]

    @torch.inference_mode()
    def infer(self, image: np.array):
        """
        :param image:
        :param info: List containing [Fx, Fy, Cx, Cy]
        :return:
        """
        h, w, c = image.shape
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

        with self.read_lock:
            depth = self.depth_anything_model(image_tensor)

        if isinstance(depth, dict):
            depth = depth.get('metric_depth', depth.get('out'))
        elif isinstance(depth, (list, tuple)):
            depth = depth[-1]

        return cv2.resize(depth.squeeze().detach().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)

    @torch.inference_mode()
    def infer_with_pointcloud(self, image: np.array, info: list):
        """
        :param image:
        :param info: List containing [Fx, Fy, Cx, Cy]
        :return:
        """
        h, w, c = image.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

        with self.read_lock:
            depth = self.depth_anything_model(image_tensor)

        if isinstance(depth, dict):
            depth = depth.get('metric_depth', depth.get('out'))
        elif isinstance(depth, (list, tuple)):
            depth = depth[-1]

        depth = cv2.resize(depth.squeeze().detach().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)

        pointcloud = np.zeros((h, w), dtype=self.pointcloud_dtype)

        image = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        pointcloud["x"] = ((x - info[2]) / info[0]) * depth
        pointcloud["y"] = ((y - info[3]) / info[1]) * depth
        pointcloud["z"] = depth
        pointcloud["r"] = image[:, :, 2]
        pointcloud["g"] = image[:, :, 1]
        pointcloud["b"] = image[:, :, 0]

        return depth, pointcloud
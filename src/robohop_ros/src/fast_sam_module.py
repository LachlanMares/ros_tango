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

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from typing import Union
from pathlib import Path
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.utils.ops import scale_masks
from PIL import Image


class FastSamClass:
    def __init__(self, config_settings: dict):
        root_dir = Path(__file__).resolve().parents[1]
        model_dir = root_dir / 'models'

        # Set device
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = torch.device(config_settings["cuda_device"])
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.image_width = config_settings["width"]
        self.image_height = config_settings["height"]
        self.mask_height = config_settings["mask_height"]
        self.mask_width = config_settings["mask_width"]

        overrides = dict(conf=config_settings["conf"],
                         task="segment",
                         mode="predict",
                         model=str(model_dir / config_settings["model"]),
                         save=False,
                         verbose=False,
                         imgsz=config_settings["imgsz"])

        self.predictor = FastSAMPredictor(overrides=overrides)

        _ = self.predictor(np.zeros((config_settings["height"], config_settings["width"], 3), dtype=np.uint8))

    @torch.inference_mode()
    def infer(self, image: np.array) -> np.array:
        results = self.predictor(image)[0]
        return self.process_results(results=results)

    @torch.inference_mode()
    def segment(self, image: np.array, return_mask_as_dict: bool = True):
        results = self.predictor(image)[0]

        if results.masks is not None:
            # Re-order based upon mask size
            mask_sums = torch.sum(results.masks.data, dim=(1, 2), dtype=torch.int32)
            ordered_sums = torch.argsort(mask_sums, descending=True).to(torch.int32)

            masks = F.interpolate(results.masks.data[ordered_sums].unsqueeze(0),
                                  size=(self.image_height, self.image_width),
                                  mode="nearest-exact").squeeze().to(torch.bool).cpu().numpy()
            mask_sums = mask_sums[ordered_sums].cpu().numpy()

            if return_mask_as_dict:
                return [{'segmentation': masks[i], 'area': mask_sums[i]} for i in range(masks.shape[0])]

            else:
                return masks, mask_sums

        else:
            if return_mask_as_dict:
                return [{'segmentation': np.zeros((self.image_height, self.image_width), dtype=np.uint8),
                         'area': np.zeros(0, dtype=np.uint32)}]

            else:
                return np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros(0, dtype=np.uint32)

    @torch.inference_mode()
    def infer_with_points(self, image: np.array, points: list, add_points_masks: bool) -> (np.array, np.array):
        results = self.predictor(image)[0]

        if results.masks is not None:
            results_mask = self.process_results(results=results)
            point_masks = []

            for point in points:
                point_mask = self.process_results(self.predictor.prompt(results, points=point)[0])
                if add_points_masks:
                    results_mask[point_mask > 0] = np.max(results_mask)+1
                point_masks.append(point_mask)

            return results_mask, np.asarray(point_masks)

        else:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros((1, self.image_height, self.image_width), dtype=np.uint8)

    @torch.inference_mode()
    def infer_with_boxes(self, image: np.array, boxes: list, add_boxes_masks: bool) -> (np.array, np.array):
        results = self.predictor(image)[0]

        if results.masks is not None:
            results_mask = self.process_results(results=results)
            box_masks = []

            for box in boxes:
                box_mask = self.process_results(self.predictor.prompt(results, bboxes=box)[0])
                if add_boxes_masks:
                    results_mask[box_mask > 0] = np.max(results_mask) + 1
                box_masks.append(box_mask)

            return results_mask, np.asarray(box_masks)

        else:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros((1, self.image_height, self.image_width), dtype=np.uint8)

    @torch.inference_mode()
    def infer_robohop_points(self, image: np.array, traversable_point: np.array) -> (np.array, np.array, np.array, np.array, np.array):
        results = self.predictor(image)[0]

        if results.masks is not None:
            if len(traversable_point) > 0:
                _, point_results_indices = self.prompt_points(results, points=traversable_point)

                if point_results_indices is not None:
                    for i, point_results_index in enumerate(point_results_indices):
                        if i == 0:
                            traversable_results = results[point_results_index].masks.data
                            results.masks.data = results.masks.data[~point_results_index]

                        else:
                            traversable_results = torch.stack((traversable_results, results[point_results_index].masks.data), dim=0)

                    if traversable_results.shape[0] > 0:
                        traversable_results = torch.clamp(torch.sum(traversable_results, dim=0, keepdim=True),
                                                          min=0, max=1)
                        traversable_results = F.interpolate(traversable_results.unsqueeze(0) * 255,
                                                            size=(self.image_height, self.image_width),
                                                            mode="nearest-exact").squeeze().to(torch.uint8).cpu().numpy()
                    else:
                        traversable_results = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

                else:
                    traversable_results = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

            else:
                traversable_results = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

            # Re-order based upon mask size
            mask_sums = torch.sum(results.masks.data, dim=(1, 2)).to(torch.int32)
            ordered_masks = torch.argsort(mask_sums, descending=True).to(torch.int32)

            mask_sums = mask_sums[ordered_masks].cpu().numpy()
            results.masks.data = results.masks.data[ordered_masks].to(torch.uint8).unsqueeze(0)
            results.masks.data = F.interpolate(results.masks.data, size=(self.image_height, self.image_width), mode="nearest-exact").squeeze(0)

            # Redo with new size
            mask_sums = torch.sum(results.masks.data, dim=(1, 2)).to(torch.int32).cpu().numpy()
            mask_coordinates = torch.zeros((results.masks.data.shape[0], 2), dtype=torch.long, device=self.device)

            # Can't use One-Hot as some segments overlap, instead use non-zero indices
            non_zeros_indices = torch.nonzero(results.masks.data).cpu().numpy()

            for i in range(results.masks.data.shape[0]):
                non_zeros = torch.nonzero(results.masks.data[i]).float()
                mask_coordinates[i][0] = torch.mean(non_zeros[:, 0])
                mask_coordinates[i][1] = torch.mean(non_zeros[:, 1])

            masks_tensor = results.masks[0].data.squeeze()

            for i in range(1, results.masks.data.shape[0]):
                masks_tensor[results.masks[i].data.squeeze() > 0] = i + 1

            return masks_tensor.cpu().numpy(), mask_sums, mask_coordinates.cpu().numpy(), traversable_results, non_zeros_indices

        else:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros(1, dtype=np.int32), np.zeros(2, dtype=np.uint32), np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros(2, dtype=np.int32)

    @torch.inference_mode()
    def infer_robohop_clip(self, image: np.array, fast_sam_strings: list) -> (np.array, np.array, np.array, np.array, np.array, np.array):
        results = self.predictor(image)[0]

        if results.masks is not None:
            if len(fast_sam_strings) > 0:
                _, text_results_indices = self.prompt_text(results, texts=fast_sam_strings)

                if text_results_indices is not None:
                    for i, text_results_index in enumerate(text_results_indices):
                        if i == 0:
                            traversable_results = results[text_results_index].masks.data
                            results.masks.data = results.masks.data[~text_results_index]

                        else:
                            traversable_results = torch.stack((traversable_results, results[text_results_index].masks.data), dim=0)

                    if traversable_results.shape[0] > 0:
                        traversable_results = torch.clamp(torch.sum(traversable_results, dim=0, keepdim=True), min=0, max=1)
                        traversable_results = F.interpolate(traversable_results.unsqueeze(0) * 255,
                                                            size=(self.image_height, self.image_width),
                                                            mode="nearest-exact").squeeze().to(torch.uint8).cpu().numpy()
                    else:
                        traversable_results = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

                else:
                    traversable_results = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

            else:
                traversable_results = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

            # Re-order based upon mask size
            mask_sums = torch.sum(results.masks.data, dim=(1, 2)).to(torch.int32)
            ordered_masks = torch.argsort(mask_sums, descending=True).to(torch.int32)

            mask_sums = mask_sums[ordered_masks].cpu().numpy()
            results.masks.data = results.masks.data[ordered_masks].to(torch.uint8).unsqueeze(0)
            results.masks.data = F.interpolate(results.masks.data,
                                               size=(self.image_height, self.image_width),
                                               mode="nearest-exact").squeeze(0)
            # Redo with new size
            mask_sums = torch.sum(results.masks.data, dim=(1, 2)).to(torch.int32).cpu().numpy()
            mask_coordinates = torch.zeros((results.masks.data.shape[0], 2), dtype=torch.long, device=self.device)

            # Can't use One-Hot as some segments overlap, instead use non-zero indices
            non_zeros_indices = torch.nonzero(results.masks.data).cpu().numpy()

            for i in range(results.masks.data.shape[0]):
                non_zeros = torch.nonzero(results.masks.data[i]).float()
                mask_coordinates[i][0] = torch.mean(non_zeros[:, 0])
                mask_coordinates[i][1] = torch.mean(non_zeros[:, 1])

            masks_tensor = results.masks[0].data.squeeze()

            for i in range(1, results.masks.data.shape[0]):
                masks_tensor[results.masks[i].data.squeeze() > 0] = i + 1

            return masks_tensor.cpu().numpy(), mask_sums, mask_coordinates.cpu().numpy(), traversable_results, non_zeros_indices

        else:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros(1, dtype=np.int32), np.zeros(2, dtype=np.uint32), np.zeros((self.image_height, self.image_width), dtype=np.uint8), np.zeros(2, dtype=np.int32)

    @torch.inference_mode()
    def prompt_points(self, results, points=None, labels=None) -> (list, list):
        """
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

        Args:
            results (Results | List[Results]): The original inference results from FastSAM models without any prompts.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.

        Returns:
            (List[Results]): The output results determined by prompts.
        """
        if points is None:
            return results, None

        prompt_results = []
        idx_list = []

        if not isinstance(results, list):
            results = [results]

        for result in results:
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]

            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)

            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)

                assert len(labels) == len(points), f"Excepted `labels` got same size as `point`, but got {len(labels)} and {len(points)}"

                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # all negative points
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )

                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = True if label else False

                idx |= point_idx

            prompt_results.append(result[idx])
            idx_list.append(idx)

        return prompt_results, idx_list

    @torch.inference_mode()
    def prompt_text(self, results, texts=None) -> (list, list):
        """
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

        Args:
            results (Results | List[Results]): The original inference results from FastSAM models without any prompts.
            texts (str | List[str], optional): Textual prompts, a list contains string objects.

        Returns:
            (List[Results]): The output results determined by prompts.
        """
        if texts is None:
            return results, None
        prompt_results = []
        idx_list = []

        if not isinstance(results, list):
            results = [results]

        for result in results:
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]

            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)

            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []

                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))

                similarity = self.predictor._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )

                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[:, None] <= text_idx[None, :]).sum(0)

                idx[text_idx] = True

            prompt_results.append(result[idx])
            idx_list.append(idx)

        return prompt_results, idx_list

    @torch.inference_mode()
    def process_results(self, results) -> np.array:
        try:
            mask_shape = results.masks[0].data.shape[1:]
            masks_tensor = torch.zeros(mask_shape, dtype=torch.uint8, device=self.device)
            # Sort the masks by size
            mask_sums = torch.argsort(torch.sum(results.masks.data, dim=(1, 2)), descending=True).to(torch.int32)

            for i, mask in enumerate(results.masks.data[mask_sums]):
                masks_tensor[mask > 0] = i + 1

            return cv2.resize(masks_tensor.cpu().numpy(), (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        except:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)


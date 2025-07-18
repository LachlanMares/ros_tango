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

import time
import rospy
import cv2
import pickle
import torch
import threading
import numpy as np
import networkx as nx

from pathlib import Path
from natsort import natsorted

from src import SuperPoint, LightGlue, numpy_image_to_torch, resize_image, load_image, rbd, rle_to_mask, nodes2key, ModeFilterClass


class RoboHopMatcherLightGlue:
    def __init__(self,
                 resize_width: int = 464,
                 resize_height: int = 400,
                 cuda_device: int = 0,
                 number_of_matchers: int = 5):
        """ """
        if torch.cuda.device_count() > 1:
            self.device = torch.device(cuda_device)
        elif torch.cuda.device_count() == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.resize_width = resize_width
        self.resize_height = resize_height
        self.match_pairs = []
        self.read_lock = threading.Lock()
        self.number_of_matchers = number_of_matchers

        self.lexor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.lightglue_list = []

        for _ in range(self.number_of_matchers):
            self.lightglue_list.append(LightGlue(features='superpoint').eval().to(self.device))

    def get_image_path(self, image_path: str):
        """ """
        return load_image(image_path, resize=(self.resize_height, self.resize_width)).to(self.device)

    def get_image_array(self, img):
        """ """
        return numpy_image_to_torch(resize_image(img, (self.resize_height, self.resize_width))[0]).to(self.device)

    def get_image(self, path_or_array: str):
        """ """
        if isinstance(path_or_array, str):
            return self.get_image_path(path_or_array)
        else:
            return self.get_image_array(path_or_array)

    def mask_2_kp(self, kp, masks):
        """ """
        if masks.shape[1] != self.resize_height or masks.shape[2] != self.resize_width:
            masks = cv2.resize(masks.transpose(1, 2, 0).astype(float),
                               (self.resize_width, self.resize_height),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            masks = masks.transpose(2, 0, 1)

        return masks[:, kp[:, 1].astype(int), kp[:, 0].astype(int)]

    @torch.inference_mode()
    def match_pair_image_with_mask_lightglue(self,
                                             image_features: torch.Tensor,
                                             target_features: torch.Tensor,
                                             image_masks: np.array,
                                             target_masks: np.array,
                                             image_areas: np.array,
                                             target_areas: np.array,
                                             lightglue,
                                             i: int):
        """ """
        light_glue_matches = rbd(
            lightglue({'image0': image_features, 'image1': target_features}))  # remove batch dimension

        source_features, target_features = [rbd(x) for x in [image_features, target_features]]  # remove batch dimension
        kp1, kp2, matches = source_features['keypoints'], target_features['keypoints'], light_glue_matches['matches']
        mkp1, mkp2 = kp1[matches[..., 0]].detach().cpu().numpy(), kp2[matches[..., 1]].detach().cpu().numpy()

        mask2kp1 = self.mask_2_kp(kp=mkp1, masks=image_masks)
        mask2kp2 = self.mask_2_kp(kp=mkp2, masks=target_masks)

        area_diff_ij = image_areas[:, None].astype(float) / target_areas[None, :]
        area_diff_ij[area_diff_ij > 1] = 1 / area_diff_ij[area_diff_ij > 1]
        lmat_ij = (mask2kp1[:, None] * mask2kp2[None,]).sum(-1)
        matches_bool_lfm = lmat_ij.sum(1) != 0
        lmat_ij = lmat_ij * area_diff_ij

        matches_ij = lmat_ij.argmax(1)
        matches_bool_area = area_diff_ij[np.arange(len(matches_ij)), matches_ij] > 0.5
        matches_bool = np.logical_and(matches_bool_lfm, matches_bool_area)
        matched_pair = (np.column_stack([np.argwhere(matches_bool).flatten(), matches_ij[matches_bool]]))

        with self.read_lock:
            self.match_pairs[i] = matched_pair

    @torch.inference_mode()
    def match_pair_image_with_mask_multi_lightglue(self,
                                                   image_features: torch.Tensor,
                                                   target_features_list: list,
                                                   image_masks: np.array,
                                                   target_masks_list: list,
                                                   image_areas: np.array,
                                                   target_areas_list: list,
                                                   ):
        """ """
        number_of_matchers = min(self.number_of_matchers, len(target_areas_list))
        self.match_pairs = [0] * number_of_matchers

        threads = [threading.Thread(target=self.match_pair_image_with_mask_lightglue,
                                    args=(image_features,  # image_features
                                          target_features_list[i],  # source_features
                                          image_masks,  # image_masks
                                          target_masks_list[i],  # target_masks
                                          image_areas,  # image_areas
                                          target_areas_list[i],  # nodes_source
                                          self.lightglue_list[i],
                                          i,  # index
                                          )) for i in range(number_of_matchers)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        return self.match_pairs


class RoboHopLocaliseTopological:
    def __init__(self,
                 image_directory,
                 map_graph,
                 resize_width,
                 resize_height,
                 mode_filter_length: int = 1,
                 max_number_of_matchers: int = 5,
                 max_reloc_level: int = 5,
                 cuda_device: int = 0,
                 map_image_positions=None):
        """ """
        self.image_names = natsorted(os.listdir(f'{image_directory}'))
        self.image_names = [f'{image_directory}/{image_name}' for image_name in self.image_names]
        self.number_of_images = len(self.image_names)
        self.map_graph = map_graph
        self.node_id_to_image_region_idx = np.array([map_graph.nodes[node]['map'] for node in map_graph.nodes()])
        self.map_image_positions = map_image_positions
        self.reloc_level = 1
        self.max_reloc_level = max_reloc_level
        self.number_of_matchers = max_number_of_matchers

        if mode_filter_length > 1:
            self.use_mode_filter = True
            self.mode_filter = ModeFilterClass(filter_length=mode_filter_length)
        else:
            self.use_mode_filter = False

        self.matcher = RoboHopMatcherLightGlue(resize_width=resize_width,
                                               resize_height=resize_height,
                                               cuda_device=cuda_device,
                                               number_of_matchers=self.number_of_matchers)

        # Compute graph image features
        self.target_features_list = self.extract_image_features()

        # Get masks and areas
        self.graph_masks, self.areas, self.coordinates = self.extract_graph_features()

        self.default_reloc_rad = 5  # 5
        self.reloc_rad_add = 2  # 2
        self.reloc_rad = self.default_reloc_rad
        self.localizer_iter_lb = 0
        self.localised_image_index = 0
        self.greedy_propeller = False

    def extract_graph_features(self):
        rospy.loginfo(f"Extracting graph masks and areas")
        masks, areas, coordinates = [], [], []

        for reference_image_index in np.arange(start=0, stop=self.number_of_images):
            reference_node_indices = np.argwhere(
                self.node_id_to_image_region_idx[:, 0] == reference_image_index).flatten()
            graph_data = [self.map_graph.nodes(data=True)[n] for n in reference_node_indices]
            masks.append(nodes2key(node_indices=graph_data, key='segmentation'))
            areas.append(nodes2key(node_indices=graph_data, key='area'))
            coordinates.append(nodes2key(node_indices=graph_data, key='coords'))

        return masks, areas, coordinates

    def extract_image_features(self):
        rospy.loginfo(f"Pre computing SuperPoint features")
        target_features = []
        for image_name in self.image_names:
            image_tensor = self.matcher.get_image(path_or_array=image_name)
            target_features.append(self.matcher.lexor.extract(image_tensor))

        return target_features

    def update_localizer_iter_lb(self):
        """ """
        if self.greedy_propeller:
            if self.localised_image_index > self.localizer_iter_lb:
                self.localizer_iter_lb = self.localised_image_index
        else:
            self.localizer_iter_lb = max(0, self.localised_image_index - 1)  # self.reloc_rad // 2)

    def get_reference_image_indices(self) -> np.array:
        return np.clip(a=np.arange(start=self.localizer_iter_lb,
                                   stop=min(self.localizer_iter_lb + self.reloc_rad, self.number_of_images), step=1),
                       a_min=0, a_max=self.number_of_images)

    def get_closest_map_image_index(self, query_position):
        """ """
        dists = np.linalg.norm(self.map_image_positions - query_position, axis=1)
        return np.argmin(dists)

    def localize_lightglue(self,
                           image_features: torch.Tensor,
                           image_masks: np.array,
                           image_areas: np.array,
                           query_position=None) -> np.array:
        """ """
        self.update_localizer_iter_lb()
        reference_image_indices = self.get_reference_image_indices()

        target_features_list, target_nodes_indices_list = [], []
        target_masks_list, target_areas_list = [], []

        if len(reference_image_indices) > self.number_of_matchers:
            step = min(5, len(reference_image_indices) // self.number_of_matchers)
            reference_image_indices = reference_image_indices[::step][:self.number_of_matchers]

        for reference_indices in reference_image_indices:
            target_features_list.append(self.target_features_list[reference_indices])
            reference_nodes_indices = np.argwhere(self.node_id_to_image_region_idx[:, 0] == reference_indices).flatten()
            target_nodes_indices_list.append(reference_nodes_indices)
            target_masks_list.append(self.graph_masks[reference_indices])
            target_areas_list.append(self.areas[reference_indices])

        match_pairs_list = self.matcher.match_pair_image_with_mask_multi_lightglue(image_features=image_features,
                                                                                   target_features_list=target_features_list,
                                                                                   image_masks=image_masks,
                                                                                   target_masks_list=target_masks_list,
                                                                                   image_areas=image_areas,
                                                                                   target_areas_list=target_areas_list)

        matched_reference_node_indices = np.concatenate(
            [target_nodes_indices_list[i][match_pairs[:, 1]] for i, match_pairs in enumerate(match_pairs_list)])

        match_pairs = np.column_stack([np.vstack(match_pairs_list)[:, 0], matched_reference_node_indices])

        if len(match_pairs) == 0:
            rospy.logwarn("Lost! No matches found. Panic!!!!...")
            #self.reloc_rad += self.reloc_rad_add

        elif len(match_pairs) < 40:
            rospy.logwarn("Low number of matches found. Don't Panic just yet...")
            #self.reloc_rad += self.reloc_rad_add

        else:
            self.reloc_rad = self.default_reloc_rad

        matched_reference_image_indices = self.node_id_to_image_region_idx[matched_reference_node_indices][:, 0]
        bc = np.bincount(matched_reference_image_indices)

        if self.use_mode_filter:
            self.localised_image_index = self.mode_filter.update(new_value=bc.argmax())
        else:
            self.localised_image_index = bc.argmax()

        rospy.loginfo(
            f"reference_image_indices: {reference_image_indices}, Localized to image with index: {self.localised_image_index}")

        if query_position is not None and self.map_image_positions is not None:
            closest_map_image_index = self.get_closest_map_image_index(query_position)
            rospy.loginfo(f"Closest map image index: {closest_map_image_index}")

        return match_pairs


class RoboHopPlanTopological:
    def __init__(self, map_graph, goal_node_indices: list, plan_da_neighbours: bool = False, maps_id: str = ""):
        this_dir = Path(__file__).resolve().parents[0]
        plan_file = this_dir / f'saved_plans/{maps_id}.pkl'

        if plan_file.is_file():
            with open(plan_file, 'rb') as f:
                previous_computed_plan = pickle.load(f)

            self.map_node_weight_string = previous_computed_plan["map_node_weight_string"]
            self.map_graph = previous_computed_plan["map_graph"]
            self.node_id_to_image_region_idx = previous_computed_plan["node_id_to_image_region_idx"]
            self.all_path_lengths = previous_computed_plan["all_path_lengths"]

            # Get goal node neighbours from the same image
            # goal_node_neighbours = list(self.map_graph.neighbors(goal_node_index))
            # goal_image_index = self.node_id_to_image_region_idx[goal_node_index][0]
            # self.goal_node_neighbours =  [n for n in goal_node_neighbours if self.node_id_to_image_region_idx[n][0] == goal_image_index]
            self.goal_node_neighbours = goal_node_indices
            self.goal_node_neighbours_image_index = self.node_id_to_image_region_idx[self.goal_node_neighbours, 0]

            rospy.loginfo(f"{self.goal_node_neighbours_image_index=}, {self.goal_node_neighbours=}")
            self.plan_da_neighbours = plan_da_neighbours

            if self.plan_da_neighbours:
                rospy.loginfo("Precomputing DA neighbours")
                self.da_neighbours = self.precompute_neighbours(graph=self.map_graph, edge_type='da')

        else:
            self.map_node_weight_string = None  # "margin"
            self.map_graph = map_graph
            self.node_id_to_image_region_idx = np.array([map_graph.nodes[node]['map'] for node in map_graph.nodes()])

            rospy.loginfo(f"Precomputing all path lengths...")
            self.all_path_lengths = self.get_path(source=None,
                                                  target=None,
                                                  graph=self.map_graph,
                                                  weight=self.map_node_weight_string,
                                                  all_pairs=True)

            # Get goal node neighbours from the same image
            # goal_node_neighbours = list(self.map_graph.neighbors(goal_node_index))
            # goal_image_index = self.node_id_to_image_region_idx[goal_node_index][0]
            # self.goal_node_neighbours =  [n for n in goal_node_neighbours if self.node_id_to_image_region_idx[n][0] == goal_image_index]
            self.goal_node_neighbours = goal_node_indices
            self.goal_node_neighbours_image_index = self.node_id_to_image_region_idx[self.goal_node_neighbours, 0]

            rospy.loginfo(f"{self.goal_node_neighbours_image_index=}, {self.goal_node_neighbours=}")
            self.plan_da_neighbours = plan_da_neighbours

            if self.plan_da_neighbours:
                rospy.loginfo("Precomputing DA neighbours")
                self.da_neighbours = self.precompute_neighbours(graph=self.map_graph, edge_type='da')

            previous_computed_plan = {"map_node_weight_string": self.map_node_weight_string,
                                      "map_graph": self.map_graph,
                                      "node_id_to_image_region_idx": self.node_id_to_image_region_idx,
                                      "all_path_lengths": self.all_path_lengths,
                                      }

            with open(plan_file, 'wb') as f:
                pickle.dump(previous_computed_plan, f)

    @staticmethod
    def precompute_neighbours(graph, edge_type: str = 'da') -> list:
        """ """
        neighbours = []
        for node in graph.nodes():
            neighbours.append([n for n in graph.neighbors(node) if graph.edges[node, n].get('edgeType') == edge_type])
        return neighbours

    @staticmethod
    def get_path(source, target, graph=None, weight=None, all_pairs=False):
        """ """
        if all_pairs:
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight))
            path_lengths = np.array([[path_lengths[src][tgt] for tgt in graph.nodes()] for src in graph.nodes()])
            return path_lengths  # this returns lengths

        else:
            shortest_path = nx.shortest_path(graph, source=source, target=target, weight=weight)
            return shortest_path  # this returns paths

    def get_path_lengths_matched_nodes(self, matched_reference_node_indices) -> (np.array, list):
        """ """
        mean_path_lengths = []
        pl = []
        nodes_close_to_goal = []

        for s in matched_reference_node_indices:
            path_length_min_per_match_number = []
            s_neighbours = [s]
            # minimize path length over DA neighbours of the matched reference node
            if self.plan_da_neighbours:
                s_neighbours += self.da_neighbours[s]

            for s2 in s_neighbours:
                path_length_min_per_match_number.append(
                    np.min([self.all_path_lengths[s2, g] for g in self.goal_node_neighbours]))

            if len(path_length_min_per_match_number) == 0:  # POSSIBLE BUG
                p = -1
                number_closest_to_goal = s
            else:
                p = np.min(path_length_min_per_match_number)
                number_closest_to_goal = s_neighbours[np.argmin(path_length_min_per_match_number)]

            nodes_close_to_goal.append(number_closest_to_goal)
            pl.append(p)

        pl = np.array(pl)
        pl[pl == -1] = pl.max() + 1
        mean_path_lengths.append(pl.mean())
        rospy.loginfo(
            f"Path length min: {pl.min()}, max: {pl.max()}, matches: {len(matched_reference_node_indices)}, number_min_segments: {np.sum(pl == pl.min())}")

        return pl, nodes_close_to_goal


def main():
    rospy.loginfo("Nothing to see here move on..")


if __name__ == "__main__":
    main()

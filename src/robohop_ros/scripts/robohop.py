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

import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import time
import yaml
import pickle
import queue
import rospy
import threading
import multiprocessing
import numpy as np
from typing import Union

from cv_bridge import CvBridge
from pathlib import Path
from copy import deepcopy

from sensor_msgs.msg import Image, CompressedImage

from robohop_ros.msg import FastSAMDepthGoal
from robohop_ros.srv import RobohopLoadMap, RobohopFastSamStrings

from robohop_ros.srv import FastSamSegmentationRoboHop, FastSamSegmentationRoboHopRequest
from robohop_ros.srv import FastSamSegmentationRoboHopClip, FastSamSegmentationRoboHopClipRequest

from robohop_ros.srv import MonocularMetricDepth, MonocularMetricDepthRequest

from src import RoboHopMatcherLightGlue, RoboHopLocaliseTopological, RoboHopPlanTopological, FastSamClass, DepthAnythingMetricClass


class RoboHopClass:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        self.message_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=1)

        self.multiprocess_lock = multiprocessing.Lock()
        self.read_lock = threading.Lock()

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

        self.fast_sam_infer_with_points = settings_dict["fast_sam_infer_with_points"]

        # Variables
        self.resize_width = settings_dict['resize_width']
        self.resize_height = settings_dict['resize_height']
        self.maps_dir = settings_dict['maps_dir']
        self.mode_filter_length = settings_dict['mode_filter_length']

        self.traversable_point = np.asarray([self.resize_width // 2, self.resize_height - 10])
        self.fast_sam_strings = settings_dict["fast_sam_strings"]

        self.localizer_cuda_device = settings_dict["localizer_cuda_device"]
        self.graph = None
        self.node_id_to_image_region_index = None
        self.localizer = None
        self.max_path_length = 100
        self.planner = None
        self.query_nodes = 0
        self.coordinates = 0
        self.running = False
        self.image_index = 0
        self.goal_mask_default = np.full((self.resize_height, self.resize_width), fill_value=self.max_path_length, dtype=np.float32)

        self.image_and_super_point_results = {}
        self.depth_results = {}
        self.fast_sam_results = {}

        self.robohop_thread = threading.Thread(target=self.robohop_processing_loop)
        self.robohop_thread.start()

        # Publishers
        self.fast_sam_depth_goal_publisher: rospy.Publisher = rospy.Publisher(settings_dict['fast_sam_depth_goal_publisher_topic'], FastSAMDepthGoal, queue_size=1)

        # Services
        self.use_robohop_load_map_service = settings_dict["use_robohop_load_map_service"]
        if self.use_robohop_load_map_service:
            self.robohop_load_map_service = rospy.Service(settings_dict['service_name_robohop_load_map'], RobohopLoadMap, self.load_episode_from_service)

        if "depth_anything" in settings_dict["depth_model"]["model"]:
            self.depth_model = DepthAnythingMetricClass(config_settings=settings_dict['depth_model'])
        else:
            rospy.logerr(f"Invalid monocular metric depth model specified")

        if "FastSAM" in settings_dict["fast_sam_model"]["model"]:
            self.segmentation_model = FastSamClass(config_settings=settings_dict["fast_sam_model"])
        else:
            rospy.logerr(f"Invalid model specified")

        # Subscribers
        rospy.Subscriber(settings_dict['image_message_subscriber_topic'], Image, self.image_callback)
        rospy.Subscriber(settings_dict['compressed_image_message_subscriber_topic'], CompressedImage, self.compressed_image_callback)

        # Timers
        rospy.Timer(rospy.Duration(0.05), self.goal_mask_loop)

    @staticmethod
    def change_edge_attr(graph):
        """ """
        for e in graph.edges(data=True):
            if 'margin' in e[2]:
                e[2]['margin'] = 0.0
        return graph

    def load_episode(self, maps_id: str, graph_id: str, goal_node_indices: Union[list, int] = -1) -> None:
        """ """
        with self.read_lock:
            self.running = False

        maps_path = str(self.maps_dir / maps_id)
        self.graph = pickle.load(open(f"{maps_path}/{graph_id}", 'rb'))
        self.change_edge_attr(graph=self.graph)
        self.node_id_to_image_region_index = np.array([self.graph.nodes[node]['map'] for node in self.graph.nodes()])

        self.localizer = RoboHopLocaliseTopological(image_directory=f"{maps_path}/images",
                                                    map_graph=self.graph,
                                                    resize_width=self.resize_width,
                                                    resize_height=self.resize_height,
                                                    mode_filter_length=self.mode_filter_length,
                                                    cuda_device=self.localizer_cuda_device,
                                                    map_image_positions=None)

        if goal_node_indices == -1:
            goal_node_indices = len(self.graph.nodes) - 1

        else:
            if isinstance(goal_node_indices, int):
                goal_node_indices = np.clip(a=goal_node_indices, a_min=0, a_max=len(self.graph.nodes))

            elif isinstance(goal_node_indices, list):
                goal_node_indices = np.clip(a=np.array(goal_node_indices), a_min=0, a_max=len(self.graph.nodes))

            else:
                rospy.logwarn("Invalid goal_node_index")
                goal_node_indices = len(self.graph.nodes) - 1

        self.planner = RoboHopPlanTopological(map_graph=self.graph,
                                              goal_node_indices=list(goal_node_indices),
                                              maps_id=maps_id)

        with self.read_lock:
            self.running = True

    def load_episode_from_service(self, request) -> int:
        """ """
        with self.read_lock:
            self.running = False

        rospy.loginfo(f"Loading map: {request.maps_id}")

        maps_path = str(self.maps_dir / request.maps_id)
        self.graph = pickle.load(open(f"{maps_path}/{request.graph_id}", 'rb'))
        self.change_edge_attr(graph=self.graph)
        self.node_id_to_image_region_index = np.array([self.graph.nodes[node]['map'] for node in self.graph.nodes()])

        self.localizer = RoboHopLocaliseTopological(image_directory=f"{maps_path}/images",
                                                    map_graph=self.graph,
                                                    resize_width=self.resize_width,
                                                    resize_height=self.resize_height,
                                                    mode_filter_length=self.mode_filter_length,
                                                    cuda_device=self.localizer_cuda_device,
                                                    map_image_positions=None)

        rospy.loginfo(f"{request.goal_node_indices=}")
        if request.goal_node_indices == -1:
            goal_node_indices = len(self.graph.nodes) - 1

        else:
            if isinstance(request.goal_node_indices, int):
                goal_node_indices = np.clip(a=request.goal_node_indices, a_min=0, a_max=len(self.graph.nodes))

            elif isinstance(request.goal_node_indices, tuple) or isinstance(request.goal_node_indices, list):
                goal_node_indices = np.clip(a=np.array(request.goal_node_indices), a_min=0, a_max=len(self.graph.nodes))

            else:
                rospy.logwarn("Invalid goal_node_index")
                goal_node_indices = len(self.graph.nodes) - 1

        self.planner = RoboHopPlanTopological(map_graph=self.graph,
                                              goal_node_indices=list(goal_node_indices),
                                              maps_id=request.maps_id)

        rospy.loginfo(f"Map loaded")

        with self.read_lock:
            self.running = True

        return 1

    def update_fast_sam_strings(self, request) -> int:
        new_fast_sam_strings = []
        for fast_sam_string in request.fast_sam_strings:
            if isinstance(fast_sam_string, str):
                new_fast_sam_strings.append(fast_sam_string)

        if len(new_fast_sam_strings) > 0:
            self.fast_sam_strings = new_fast_sam_strings
            return 1
        else:
            return 0

    def get_goal_mask(self,
                      query_image_features: torch.Tensor,
                      query_masks: np.array,
                      query_areas: np.array,
                      query_position=None) -> (np.array, np.array):
        """ """
        if self.running:
            match_pairs = self.localizer.localize_lightglue(image_features=query_image_features,
                                                            image_masks=query_masks,
                                                            image_areas=query_areas,
                                                            query_position=query_position)

            path_lengths, nodes_close_to_goal = self.planner.get_path_lengths_matched_nodes(match_pairs[:, 1])

            goal_mask = self.goal_mask_default.copy()
            min_path_length = np.min(path_lengths)
            target_points = []

            for i in range(len(path_lengths)):
                goal_mask_indices = query_masks[match_pairs[i, 0]] != 0
                goal_mask[goal_mask_indices] = path_lengths[i]

                if path_lengths[i] == min_path_length:
                    target_point = np.array(np.nonzero(goal_mask_indices))
                    target_points.append([np.mean(target_point[0]).astype(np.uint32),
                                          np.mean(target_point[1]).astype(np.uint32)])

            return goal_mask, np.asarray(target_points, dtype=np.uint32)

        else:
            return self.goal_mask_default.copy(), np.zeros((1, 2), dtype=np.uint32)

    def goal_mask_loop(self, event) -> None:
        """ """
        if self.running:

            try:
                # time_now = time.time()
                queue_dict = self.message_queue.get(timeout=0.01)
                """
                {"message": fast_sam_depth_goal_message,
                 "image_features":self.image_and_super_point_results["image_features"],
                 "masks": self.fast_sam_results["masks"],
                 "areas": self.fast_sam_results["areas"],
                }
                """
                fast_sam_depth_goal_message = queue_dict["message"]

                goal, target_points = self.get_goal_mask(query_image_features=queue_dict["image_features"],
                                                         query_masks=queue_dict["masks"],
                                                         query_areas=queue_dict["areas"])

                fast_sam_depth_goal_message.goal = self.cv_bridge.cv2_to_imgmsg(goal)
                fast_sam_depth_goal_message.target_points = list(target_points.reshape(-1))

                localised_image = cv2.imread(filename=
                                             self.localizer.image_names[self.localizer.localised_image_index],
                                             flags=cv2.IMREAD_UNCHANGED)
                fast_sam_depth_goal_message.localised_image = self.cv_bridge.cv2_to_compressed_imgmsg(localised_image)
                fast_sam_depth_goal_message.localised_image_id = self.localizer.localised_image_index

                self.fast_sam_depth_goal_publisher.publish(fast_sam_depth_goal_message)
                # rospy.loginfo(f"goal_mask_loop: {time.time() - time_now:.3f}")

            except queue.Empty:
                pass

            except Exception as e:
                rospy.logerr(str(type(e)) + " " + str(e))

    def image_and_super_point(self, cv_image: np.array, image_msg: CompressedImage) -> None:
        """ """
        image_tensor = self.localizer.matcher.get_image(path_or_array=cv_image)

        with self.read_lock:
            self.image_and_super_point_results = {"image": deepcopy(image_msg),
                                                  "image_features": self.localizer.matcher.lexor.extract(image_tensor)}

    def get_segmentation_points(self, cv_image: np.array, traversable_point: np.array) -> None:
        """
        :param cv_image: cv2 image [H, W, C]
        :param traversable_point: [[u, v]]
        """

        segmentation, mask_sums, coordinates, traversable_segmentation, non_zeros_indices = self.segmentation_model.infer_robohop_points(
            image=cv_image,
            traversable_point=traversable_point)

        # Create a blank mask
        masks = np.zeros((len(mask_sums), segmentation.shape[0], segmentation.shape[1]), dtype=np.uint8)

        # Insert non-zero indices
        masks[non_zeros_indices[:, 0], non_zeros_indices[:, 1], non_zeros_indices[:, 2]] = 1

        with self.read_lock:
            self.fast_sam_results = {"masks": masks,
                                     "areas": mask_sums,
                                     "coordinates": coordinates,
                                     "segmentation_image": self.cv_bridge.cv2_to_imgmsg(segmentation),
                                     "traversable_segmentation": self.cv_bridge.cv2_to_imgmsg(traversable_segmentation)}

    def get_segmentation_clip(self, cv_image: np.array, fast_sam_strings: list) -> None:
        """
        :param cv_image: cv2 image [H, W, C]
        :param fast_sam_strings: [str, str, str]
        """

        segmentation, mask_sums, coordinates, traversable_segmentation, non_zeros_indices = self.segmentation_model.infer_robohop_clip(
            image=cv_image,
            fast_sam_strings=fast_sam_strings)

        # Create a blank mask
        masks = np.zeros((len(mask_sums), segmentation.shape[0], segmentation.shape[1]), dtype=np.uint8)

        # Insert non-zero indices
        masks[non_zeros_indices[:, 0], non_zeros_indices[:, 1], non_zeros_indices[:, 2]] = 1

        with self.read_lock:
            self.fast_sam_results = {"masks": masks,
                                     "areas": mask_sums,
                                     "coordinates": coordinates,
                                     "segmentation_image": self.cv_bridge.cv2_to_imgmsg(segmentation),
                                     "traversable_segmentation": self.cv_bridge.cv2_to_imgmsg(traversable_segmentation)}

    def compute_depth(self, cv_image: np.array) -> None:
        """
        :param cv_image: cv2 image [H, W, C]
        """
        depth = self.depth_model.infer(image=cv_image)

        with self.read_lock:
            self.depth_results = self.cv_bridge.cv2_to_imgmsg(depth)

    def image_callback(self, image_msg):
        if self.image_queue.full():
            self.image_queue.queue.clear()

        try:
            self.image_queue.put(self.cv_bridge.cv2_to_compressed_imgmsg(self.cv_bridge.imgmsg_to_cv2(image_msg)))

        except Exception as e:
            rospy.logwarn(e)

    def compressed_image_callback(self, image_msg):
        if self.image_queue.full():
            self.image_queue.queue.clear()

        try:
            self.image_queue.put(image_msg)

        except Exception as e:
            rospy.logwarn(e)

    def robohop_processing_loop(self):
        """ """
        while not rospy.is_shutdown():
            if not self.image_queue.empty():
                if self.running:
                    msg = self.image_queue.get()
                    # time_now = time.time()
                    cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

                    depth_process = threading.Thread(target=self.compute_depth, args=(deepcopy(cv_image), ))
                    depth_process.start()

                    if self.fast_sam_infer_with_points:
                        fast_sam_process = threading.Thread(target=self.get_segmentation_points,
                                                            args=(deepcopy(cv_image), self.traversable_point))
                    else:
                        fast_sam_process = threading.Thread(target=self.get_segmentation_clip,
                                                            args=(deepcopy(cv_image), self.fast_sam_strings))
                    fast_sam_process.start()

                    image_and_super_point_process = threading.Thread(target=self.image_and_super_point, args=(deepcopy(cv_image), deepcopy(msg)))
                    image_and_super_point_process.start()

                    depth_process.join()
                    fast_sam_process.join()
                    image_and_super_point_process.join()

                    fast_sam_depth_goal_message = FastSAMDepthGoal()
                    fast_sam_depth_goal_message.header.stamp = rospy.get_rostime()
                    fast_sam_depth_goal_message.image = self.image_and_super_point_results["image"]
                    fast_sam_depth_goal_message.depth = self.depth_results
                    fast_sam_depth_goal_message.segmentation = self.fast_sam_results["segmentation_image"]
                    fast_sam_depth_goal_message.traversable_segmentation = self.fast_sam_results["traversable_segmentation"]

                    try:
                        self.message_queue.put({"message": fast_sam_depth_goal_message,
                                                "image_features":self.image_and_super_point_results["image_features"],
                                                "masks": self.fast_sam_results["masks"],
                                                "areas": self.fast_sam_results["areas"],
                                                })

                    except queue.Full:
                        rospy.logwarn_throttle(1.0, "robohop: queue full")

                    except Exception as e:
                        rospy.logerr(str(type(e)) + " " + str(e))

                    # rospy.loginfo(f"compressed_image_callback: {time.time() - time_now:.3f}")

            else:
                rospy.sleep(0.01)

    def ros_shutdown(self):
        """ """
        self.running = False
        rospy.loginfo(f"Shutting down ROS robohop_no_services node")


class RoboHopServicesClass:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        self.message_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=1)

        self.multiprocess_lock = multiprocessing.Lock()
        self.read_lock = threading.Lock()

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

        # Variables
        self.resize_width = settings_dict['resize_width']
        self.resize_height = settings_dict['resize_height']
        self.maps_dir = settings_dict['maps_dir']
        self.mode_filter_length = settings_dict['mode_filter_length']

        self.traversable_point = [self.resize_width // 2, self.resize_height - 10]
        self.fast_sam_strings = settings_dict["fast_sam_strings"]

        self.localizer_cuda_device = settings_dict["localizer_cuda_device"]
        self.graph = None
        self.node_id_to_image_region_index = None
        self.localizer = None
        self.max_path_length = 100
        self.planner = None
        self.query_nodes = 0
        self.coordinates = 0
        self.running = False
        self.image_index = 0
        self.goal_mask_default = np.full((self.resize_height, self.resize_width), fill_value=self.max_path_length, dtype=np.float32)

        self.image_and_super_point_results = {}
        self.depth_results = {}
        self.fast_sam_results = {}

        # Publishers
        self.fast_sam_depth_goal_publisher: rospy.Publisher = rospy.Publisher(settings_dict['fast_sam_depth_goal_publisher_topic'], FastSAMDepthGoal, queue_size=1)

        # Services
        self.use_robohop_load_map_service = settings_dict["use_robohop_load_map_service"]
        if self.use_robohop_load_map_service:
            self.robohop_load_map_service = rospy.Service(settings_dict['service_name_robohop_load_map'], RobohopLoadMap, self.load_episode_from_service)

        self.robohop_fast_sam_strings_service = rospy.Service(settings_dict['service_name_robohop_fast_sam_strings'], RobohopFastSamStrings, self.update_fast_sam_strings)

        rospy.wait_for_service(settings_dict['metric_depth_service_name'], timeout=15.0)
        self.depth_anything_service = rospy.ServiceProxy(settings_dict['metric_depth_service_name'], MonocularMetricDepth)

        self.fast_sam_infer_with_points = settings_dict["fast_sam_infer_with_points"]
        if self.fast_sam_infer_with_points:
            rospy.wait_for_service(settings_dict['service_name_segmentation_robohop'], timeout=15.0)
            self.fast_sam_robohop_segmentation_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_robohop'], FastSamSegmentationRoboHop)

        else:
            rospy.wait_for_service(settings_dict['service_name_segmentation_robohop_clip'], timeout=15.0)
            self.fast_sam_robohop_segmentation_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_robohop_clip'], FastSamSegmentationRoboHopClip)

        # Threads
        self.robohop_thread = threading.Thread(target=self.robohop_processing_loop)
        self.robohop_thread.start()

        # Subscribers
        rospy.Subscriber(settings_dict['image_message_subscriber_topic'], Image, self.image_callback)
        rospy.Subscriber(settings_dict['compressed_image_message_subscriber_topic'], CompressedImage, self.compressed_image_callback)

        # Timers
        rospy.Timer(rospy.Duration(0.05), self.goal_mask_loop)

    @staticmethod
    def change_edge_attr(graph):
        """ """
        for e in graph.edges(data=True):
            if 'margin' in e[2]:
                e[2]['margin'] = 0.0
        return graph

    def load_episode(self, maps_id: str, graph_id: str, goal_node_indices: Union[list, int] = -1) -> None:
        """ """
        with self.read_lock:
            self.running = False

        maps_path = str(self.maps_dir / maps_id)
        self.graph = pickle.load(open(f"{maps_path}/{graph_id}", 'rb'))
        self.change_edge_attr(graph=self.graph)
        self.node_id_to_image_region_index = np.array([self.graph.nodes[node]['map'] for node in self.graph.nodes()])

        self.localizer = RoboHopLocaliseTopological(image_directory=f"{maps_path}/images",
                                                    map_graph=self.graph,
                                                    resize_width=self.resize_width,
                                                    resize_height=self.resize_height,
                                                    mode_filter_length=self.mode_filter_length,
                                                    cuda_device=self.localizer_cuda_device,
                                                    map_image_positions=None)

        if goal_node_indices == -1:
            goal_node_indices = len(self.graph.nodes) - 1

        else:
            if isinstance(goal_node_indices, int):
                goal_node_indices = np.clip(a=goal_node_indices, a_min=0, a_max=len(self.graph.nodes))

            elif isinstance(goal_node_indices, list):
                goal_node_indices = np.clip(a=np.array(goal_node_indices), a_min=0, a_max=len(self.graph.nodes))

            else:
                rospy.logwarn("Invalid goal_node_index")
                goal_node_indices = len(self.graph.nodes) - 1

        self.planner = RoboHopPlanTopological(map_graph=self.graph,
                                              goal_node_indices=list(goal_node_indices),
                                              maps_id=maps_id)

        with self.read_lock:
            self.running = True

    def load_episode_from_service(self, request) -> int:
        """ """
        with self.read_lock:
            self.running = False

        rospy.loginfo(f"Loading map: {request.maps_id}")

        maps_path = str(self.maps_dir / request.maps_id)
        self.graph = pickle.load(open(f"{maps_path}/{request.graph_id}", 'rb'))
        self.change_edge_attr(graph=self.graph)
        self.node_id_to_image_region_index = np.array([self.graph.nodes[node]['map'] for node in self.graph.nodes()])

        self.localizer = RoboHopLocaliseTopological(image_directory=f"{maps_path}/images",
                                                    map_graph=self.graph,
                                                    resize_width=self.resize_width,
                                                    resize_height=self.resize_height,
                                                    mode_filter_length=self.mode_filter_length,
                                                    cuda_device=self.localizer_cuda_device,
                                                    map_image_positions=None)

        rospy.loginfo(f"{request.goal_node_indices=}")
        if request.goal_node_indices == -1:
            goal_node_indices = len(self.graph.nodes) - 1

        else:
            if isinstance(request.goal_node_indices, int):
                goal_node_indices = np.clip(a=request.goal_node_indices, a_min=0, a_max=len(self.graph.nodes))

            elif isinstance(request.goal_node_indices, tuple) or isinstance(request.goal_node_indices, list):
                goal_node_indices = np.clip(a=np.array(request.goal_node_indices), a_min=0, a_max=len(self.graph.nodes))

            else:
                rospy.logwarn("Invalid goal_node_index")
                goal_node_indices = len(self.graph.nodes) - 1

        self.planner = RoboHopPlanTopological(map_graph=self.graph,
                                              goal_node_indices=list(goal_node_indices),
                                              maps_id=request.maps_id)

        rospy.loginfo(f"Map loaded")

        with self.read_lock:
            self.running = True

        return 1

    def update_fast_sam_strings(self, request) -> int:
        new_fast_sam_strings = []
        for fast_sam_string in request.fast_sam_strings:
            if isinstance(fast_sam_string, str):
                new_fast_sam_strings.append(fast_sam_string)

        if len(new_fast_sam_strings) > 0:
            self.fast_sam_strings = new_fast_sam_strings
            return 1
        else:
            return 0

    def get_goal_mask(self,
                      query_image_features: torch.Tensor,
                      query_masks: np.array,
                      query_areas: np.array,
                      query_position=None) -> (np.array, np.array):
        """ """
        if self.running:
            match_pairs = self.localizer.localize_lightglue(image_features=query_image_features,
                                                            image_masks=query_masks,
                                                            image_areas=query_areas,
                                                            query_position=query_position)

            path_lengths, nodes_close_to_goal = self.planner.get_path_lengths_matched_nodes(match_pairs[:, 1])

            goal_mask = self.goal_mask_default.copy()
            min_path_length = np.min(path_lengths)
            target_points = []

            for i in range(len(path_lengths)):
                goal_mask_indices = query_masks[match_pairs[i, 0]] != 0
                goal_mask[goal_mask_indices] = path_lengths[i]

                if path_lengths[i] == min_path_length:
                    target_point = np.array(np.nonzero(goal_mask_indices))
                    target_points.append([np.mean(target_point[0]).astype(np.uint32),
                                          np.mean(target_point[1]).astype(np.uint32)])

            return goal_mask, np.asarray(target_points, dtype=np.uint32)

        else:
            return self.goal_mask_default.copy(), np.zeros((1, 2), dtype=np.uint32)

    def goal_mask_loop(self, event) -> None:
        """ """
        if self.running:

            try:
                # time_now = time.time()
                queue_dict = self.message_queue.get(timeout=0.01)
                """
                {"message": fast_sam_depth_goal_message,
                 "image_features":self.image_and_super_point_results["image_features"],
                 "masks": self.fast_sam_results["masks"],
                 "areas": self.fast_sam_results["areas"],
                }
                """
                fast_sam_depth_goal_message = queue_dict["message"]

                goal, target_points = self.get_goal_mask(query_image_features=queue_dict["image_features"],
                                                         query_masks=queue_dict["masks"],
                                                         query_areas=queue_dict["areas"])

                fast_sam_depth_goal_message.goal = self.cv_bridge.cv2_to_imgmsg(goal)
                fast_sam_depth_goal_message.target_points = list(target_points.reshape(-1))

                localised_image = cv2.imread(filename=
                                             self.localizer.image_names[self.localizer.localised_image_index],
                                             flags=cv2.IMREAD_UNCHANGED)
                fast_sam_depth_goal_message.localised_image = self.cv_bridge.cv2_to_compressed_imgmsg(localised_image)
                fast_sam_depth_goal_message.localised_image_id = self.localizer.localised_image_index

                self.fast_sam_depth_goal_publisher.publish(fast_sam_depth_goal_message)
                # rospy.loginfo(f"goal_mask_loop: {time.time() - time_now:.3f}")

            except queue.Empty:
                pass

            except Exception as e:
                rospy.logerr(str(type(e)) + " " + str(e))

    def remove_from_dict(self, index: int) -> None:
        """ """
        with self.read_lock:
            if index in self.image_and_super_point_results.keys():
                del self.image_and_super_point_results[index]
            if index in self.depth_results.keys():
                del self.depth_results[index]
            if index in self.fast_sam_results.keys():
                del self.fast_sam_results[index]

    def image_and_super_point(self, image_msg: CompressedImage, index: int) -> None:
        """ """
        image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        image_tensor = self.localizer.matcher.get_image(path_or_array=image)

        with self.read_lock:
            self.image_and_super_point_results[index] = {"image": deepcopy(image_msg),
                                                         "image_features": self.localizer.matcher.lexor.extract(image_tensor)}

    def segmentation_service_points(self, image_msg: CompressedImage, traversable_point: np.array, index: int) -> None:
        """
        :param image_msg: CompressedImage [H, W, C]
        :param traversable_point: [[u, v]]
        """
        service_request = FastSamSegmentationRoboHopRequest()
        service_request.image = image_msg
        service_request.traversable_point = list(traversable_point)
        service_response = self.fast_sam_robohop_segmentation_service(service_request)

        # Create a blank mask
        masks = np.zeros((service_response.segmentation_shape[0], service_response.segmentation_shape[1],
                          service_response.segmentation_shape[2]), dtype=np.uint8)

        # Insert non-zero indices
        non_zero_indices = np.array(service_response.non_zeros_indices).reshape(
            service_response.non_zeros_indices_shape[0], service_response.non_zeros_indices_shape[1])
        masks[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = 1

        # Sum of non-zero elements in each mask
        mask_sums = np.asarray(service_response.mask_sums, dtype=np.uint32)

        # Reshape from 1D to 2D [N, (u, v)]
        coordinates = np.array(service_response.coordinates).reshape(-1, 2)

        with self.read_lock:
            self.fast_sam_results[index] = {"masks": masks,
                                            "areas": mask_sums,
                                            "coordinates": coordinates,
                                            "segmentation_image": service_response.segmentation,
                                            "traversable_segmentation": service_response.traversable_segmentation}

    def segmentation_service_clip(self, image_msg: CompressedImage, fast_sam_strings: list, index: int) -> None:
        """
        :param image_msg: CompressedImage [H, W, C]
        :param fast_sam_strings: [str, str, str]
        """
        service_request = FastSamSegmentationRoboHopClipRequest()
        service_request.image = image_msg
        service_request.fast_sam_strings = fast_sam_strings
        service_response = self.fast_sam_robohop_segmentation_service(service_request)

        # Create a blank mask
        masks = np.zeros((service_response.segmentation_shape[0], service_response.segmentation_shape[1],
                          service_response.segmentation_shape[2]), dtype=np.uint8)

        # Insert non-zero indices
        non_zero_indices = np.array(service_response.non_zeros_indices).reshape(
            service_response.non_zeros_indices_shape[0], service_response.non_zeros_indices_shape[1])

        masks[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = 1

        # Sum of non-zero elements in each mask
        mask_sums = service_response.mask_sums

        # Reshape from 1D to 2D [N, (u, v)]
        coordinates = np.array(service_response.coordinates).reshape(-1, 2)

        with self.read_lock:
            self.fast_sam_results[index] = {"masks": masks,
                                            "areas": mask_sums,
                                            "coordinates": coordinates,
                                            "segmentation_image": service_response.segmentation,
                                            "traversable_segmentation": service_response.traversable_segmentation}

    def depth_service(self, image_msg: CompressedImage, index: int) -> None:
        """ """
        service_request = MonocularMetricDepthRequest()
        service_request.image = image_msg
        service_response = self.depth_anything_service(service_request)

        with self.read_lock:
            self.depth_results[index] = service_response.depth

    def image_callback(self, image_msg):
        if self.image_queue.full():
            self.image_queue.queue.clear()

        try:
            self.image_queue.put(self.cv_bridge.cv2_to_compressed_imgmsg(self.cv_bridge.imgmsg_to_cv2(image_msg)))

        except Exception as e:
            rospy.logwarn(e)

    def compressed_image_callback(self, image_msg):
        if self.image_queue.full():
            self.image_queue.queue.clear()

        try:
            self.image_queue.put(image_msg)

        except Exception as e:
            rospy.logwarn(e)

    def robohop_processing_loop(self):
        """ """
        while not rospy.is_shutdown():
            if not self.image_queue.empty():
                if self.running:
                    msg = self.image_queue.get()

                    # time_now = time.time()
                    depth_process = threading.Thread(target=self.depth_service, args=(deepcopy(msg), self.image_index))
                    depth_process.start()

                    if self.fast_sam_infer_with_points:
                        fast_sam_process = threading.Thread(target=self.segmentation_service_points,
                                                            args=(deepcopy(msg), self.traversable_point, self.image_index))
                    else:
                        fast_sam_process = threading.Thread(target=self.segmentation_service_clip,
                                                            args=(deepcopy(msg), self.fast_sam_strings, self.image_index))
                    fast_sam_process.start()

                    image_and_super_point_process = threading.Thread(target=self.image_and_super_point, args=(deepcopy(msg), self.image_index))
                    image_and_super_point_process.start()

                    depth_process.join()
                    fast_sam_process.join()
                    image_and_super_point_process.join()

                    fast_sam_depth_goal_message = FastSAMDepthGoal()
                    fast_sam_depth_goal_message.header.stamp = rospy.get_rostime()
                    fast_sam_depth_goal_message.image = self.image_and_super_point_results[self.image_index]["image"]
                    fast_sam_depth_goal_message.depth = self.depth_results[self.image_index]
                    fast_sam_depth_goal_message.segmentation = self.fast_sam_results[self.image_index]["segmentation_image"]
                    fast_sam_depth_goal_message.traversable_segmentation = self.fast_sam_results[self.image_index]["traversable_segmentation"]

                    try:
                        self.message_queue.put({"message": fast_sam_depth_goal_message,
                                                "image_features":self.image_and_super_point_results[self.image_index]["image_features"],
                                                "masks": self.fast_sam_results[self.image_index]["masks"],
                                                "areas": self.fast_sam_results[self.image_index]["areas"],
                                                })

                    except queue.Full:
                        rospy.logwarn_throttle(1.0, "robohop: queue full")

                    except Exception as e:
                        rospy.logerr(str(type(e)) + " " + str(e))

                    self.remove_from_dict(index=self.image_index)
                    self.image_index +=1

            else:
                rospy.sleep(0.01)

    def ros_shutdown(self):
        """ """
        self.running = False
        rospy.loginfo(f"Shutting down ROS robohop node")


def main():
    """ """
    rospy.init_node('robohop_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'robohop_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)
    config_settings["maps_dir"] = root_dir / 'maps'

    if rospy.get_param(param_name="~use_model_service"):
        rhc = RoboHopServicesClass(settings_dict=config_settings)

    else:
        rhc = RoboHopClass(settings_dict=config_settings)

    if not config_settings["use_robohop_load_map_service"]:
        rospy.loginfo(f"Loading map: {config_settings['load_map']}")
        rhc.load_episode(maps_id=config_settings[config_settings['load_map']]["maps_id"],
                         graph_id=config_settings[config_settings['load_map']]["graph_id"],
                         goal_node_indices=config_settings[config_settings['load_map']]["goal_node_indices"])
        rospy.loginfo(f"Episode loaded")

    rospy.spin()


if __name__ == "__main__":
    main()

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

import yaml
import rospy

import numpy as np
from cv_bridge import CvBridge
from pathlib import Path

from robohop_ros.srv import FastSamSegmentation, FastSamSegmentationResponse
from robohop_ros.srv import FastSamSegmentationPoints, FastSamSegmentationPointsResponse
from robohop_ros.srv import FastSamSegmentationBoxes, FastSamSegmentationBoxesResponse
from robohop_ros.srv import FastSamSegmentationRoboHop, FastSamSegmentationRoboHopResponse
from robohop_ros.srv import FastSamSegmentationRoboHopClip, FastSamSegmentationRoboHopClipResponse

from src import FastSamClass


class FastSamServiceClass:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        if "FastSAM" in settings_dict["model"]:
            self.model = FastSamClass(config_settings=settings_dict)
        else:
            rospy.logerr(f"Invalid model specified")

        self.segmentation_service = rospy.Service(settings_dict['service_name_segmentation'], FastSamSegmentation, self.handle_segmentation_service_request)
        self.segmentation_points_service = rospy.Service(settings_dict['service_name_segmentation_points'], FastSamSegmentationPoints, self.handle_segmentation_points_service_request)
        self.segmentation_boxes_service = rospy.Service(settings_dict['service_name_segmentation_boxes'], FastSamSegmentationBoxes, self.handle_segmentation_boxes_service_request)
        self.segmentation_robohop_points_service = rospy.Service(settings_dict['service_name_segmentation_robohop'], FastSamSegmentationRoboHop, self.handle_segmentation_robohop_service_request)
        self.segmentation_robohop_clip_service = rospy.Service(settings_dict['service_name_segmentation_robohop_clip'], FastSamSegmentationRoboHopClip, self.handle_segmentation_robohop_clip_service_request)

        self.fast_sam_segmentation_service = rospy.ServiceProxy(settings_dict['service_name_segmentation'], FastSamSegmentation)
        self.fast_sam_segmentation_points_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_points'], FastSamSegmentationPoints)
        self.fast_sam_segmentation_boxes_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_boxes'], FastSamSegmentationBoxes)
        self.fast_sam_segmentation_robohop_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_robohop'], FastSamSegmentationRoboHop)


        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

    def handle_segmentation_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
        ---
        :return response: sensor_msgs/Image segmentation
        """
        segmentation = self.model.infer(image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'))
        response = FastSamSegmentationResponse()
        response.segmentation = self.cv_bridge.cv2_to_imgmsg(segmentation, encoding="mono8")

        return response

    def handle_segmentation_points_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
                        uint32[] points
                        uint8 add_points_result_to_segmentation
        ---
        :return response: sensor_msgs/Image segmentation
                          uint32[] point_segmentation_shape
                          uint8[] point_segmentation
        """
        segmentation, point_segmentation = self.model.infer_with_points(
            image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'),
            points=list(np.asarray(request.points).reshape(-1, 2)),
            add_points_masks=True if request.add_points_result_to_segmentation > 0 else False)

        response = FastSamSegmentationPointsResponse()
        response.segmentation = self.cv_bridge.cv2_to_imgmsg(segmentation, encoding="mono8")
        response.point_segmentation_shape = list(point_segmentation.shape)
        response.point_segmentation = list(point_segmentation.reshape(-1))

        return response

    def handle_segmentation_boxes_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
                        uint32[] boxes
                        uint8 add_boxes_result_to_segmentation
        ---
        :return response: sensor_msgs/Image segmentation
                          uint32[] boxes_segmentation_shape
                          uint8[] boxes_segmentation
        """
        segmentation, boxes_segmentation = self.model.infer_with_boxes(
            image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'),
            boxes=list(np.asarray(request.boxes).reshape(-1, 4)),
            add_boxes_masks=True if request.add_boxes_result_to_segmentation > 0 else False)

        response = FastSamSegmentationBoxesResponse()
        response.segmentation = self.cv_bridge.cv2_to_imgmsg(segmentation, encoding="mono8")
        response.boxes_segmentation_shape = list(boxes_segmentation.shape)
        response.boxes_segmentation = list(boxes_segmentation.reshape(-1))

        return response

    def handle_segmentation_robohop_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
                        uint32[] traversable_point
        ---
        :return response: sensor_msgs/Image segmentation
                          sensor_msgs/Image traversable_segmentation
                          int32[] mask_sums
                          int32[] coordinates
                          uint32[] segmentation_shape
                          uint32[] non_zeros_indices
        """
        segmentation, mask_sums, coordinates, traversable_segmentation, non_zeros_indices = self.model.infer_robohop_points(
            image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'),
            traversable_point=np.asarray(request.traversable_point).reshape(-1, 2))

        response = FastSamSegmentationRoboHopResponse()
        response.segmentation = self.cv_bridge.cv2_to_imgmsg(segmentation, encoding="mono8")
        response.traversable_segmentation = self.cv_bridge.cv2_to_imgmsg(traversable_segmentation, encoding="mono8")
        response.mask_sums = list(mask_sums)
        response.coordinates = list(coordinates.reshape(-1))
        response.segmentation_shape = [len(mask_sums), segmentation.shape[0], segmentation.shape[1]]
        response.non_zeros_indices_shape = list(non_zeros_indices.shape)
        response.non_zeros_indices = list(non_zeros_indices.reshape(-1))

        return response

    def handle_segmentation_robohop_clip_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
                        string[] fast_sam_strings
        ---
        :return response: sensor_msgs/Image segmentation
                          sensor_msgs/Image traversable_segmentation
                          int32[] mask_sums
                          int32[] coordinates
                          uint32[] segmentation_shape
                          uint32[] non_zeros_indices
        """
        segmentation, mask_sums, coordinates, traversable_segmentation, non_zeros_indices = self.model.infer_robohop_clip(
            image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'),
            fast_sam_strings=request.fast_sam_strings)

        response = FastSamSegmentationRoboHopClipResponse()
        response.segmentation = self.cv_bridge.cv2_to_imgmsg(segmentation, encoding="mono8")
        response.traversable_segmentation = self.cv_bridge.cv2_to_imgmsg(traversable_segmentation, encoding="mono8")
        response.mask_sums = list(mask_sums)
        response.coordinates = list(coordinates.reshape(-1))
        response.segmentation_shape = [len(mask_sums), segmentation.shape[0], segmentation.shape[1]]
        response.non_zeros_indices_shape = list(non_zeros_indices.shape)
        response.non_zeros_indices = list(non_zeros_indices.reshape(-1))

        return response

    def ros_shutdown(self):
        rospy.loginfo(f"Shutting down ROS fast_sam_service node")


def main():
    rospy.init_node('fast_sam_service_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'fast_sam_service_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    fssc = FastSamServiceClass(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

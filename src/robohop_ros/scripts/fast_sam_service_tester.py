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
import cv2
import numpy as np

from cv_bridge import CvBridge
from pathlib import Path

from sensor_msgs.msg import Image, PointCloud2, CompressedImage

from robohop_ros.srv import FastSamSegmentation, FastSamSegmentationRequest
from robohop_ros.srv import FastSamSegmentationPoints, FastSamSegmentationPointsRequest
from robohop_ros.srv import FastSamSegmentationBoxes, FastSamSegmentationBoxesRequest
from robohop_ros.srv import FastSamSegmentationRoboHop, FastSamSegmentationRoboHopRequest
from robohop_ros.srv import FastSamSegmentationRoboHopClip, FastSamSegmentationRoboHopClipRequest


class FastSamServiceTester:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        # Publishers
        self.segmentation_pub: rospy.Publisher = rospy.Publisher(settings_dict['segmentation_image_publisher_topic'], Image, queue_size=1)
        self.segmentation_points_pub: rospy.Publisher = rospy.Publisher(settings_dict['segmentation_points_image_publisher_topic'], Image, queue_size=1)
        self.segmentation_boxes_pub: rospy.Publisher = rospy.Publisher(settings_dict['segmentation_boxes_image_publisher_topic'], Image, queue_size=1)
        self.segmentation_traversable_pub: rospy.Publisher = rospy.Publisher(settings_dict['segmentation_traversable_publisher_topic'], Image, queue_size=1)

        self.points = settings_dict['points']
        self.boxes = settings_dict['boxes']

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

        # Service
        rospy.wait_for_service(settings_dict['service_name_segmentation'], timeout=15.0)

        self.fast_sam_segmentation_service = rospy.ServiceProxy(settings_dict['service_name_segmentation'], FastSamSegmentation)
        self.fast_sam_segmentation_points_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_points'], FastSamSegmentationPoints)
        self.fast_sam_segmentation_boxes_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_boxes'], FastSamSegmentationBoxes)
        self.fast_sam_segmentation_robohop_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_robohop'], FastSamSegmentationRoboHop)
        self.fast_sam_segmentation_robohop_clip_service = rospy.ServiceProxy(settings_dict['service_name_segmentation_robohop_clip'], FastSamSegmentationRoboHopClip)

        self.service_counter = 0
        self.service_number = 0
        self.new_service = True

        self.display_segmentation_with_colours = settings_dict['display_segmentation_with_colours']
        self.map_class_colours = np.random.randint(low=0, high=256, size=(256, 3), dtype=np.uint8)

        # Subscribers
        rospy.Subscriber(settings_dict['compressed_image_message_subscriber_topic'], CompressedImage, self.compressed_image_callback)
        rospy.Subscriber(settings_dict['image_message_subscriber_topic'], Image, self.image_callback)

    def image_callback(self, msg):
        self.compressed_image_callback(msg=self.cv_bridge.cv2_to_compressed_imgmsg(self.cv_bridge.imgmsg_to_cv2(msg)))

    def compressed_image_callback(self, msg):
        if self.service_number == 0:
            """
            :param request: sensor_msgs/CompressedImage image
            ---
            :return response: sensor_msgs/Image segmentation
            """
            if self.new_service:
                self.new_service = False
                rospy.loginfo(f"fast_sam_segmentation_service")

            service_request = FastSamSegmentationRequest()
            service_request.image = msg
            service_response = self.fast_sam_segmentation_service(service_request)

            if self.display_segmentation_with_colours:
                segmentation_image = self.map_class_colours[self.cv_bridge.imgmsg_to_cv2(service_response.segmentation, desired_encoding="passthrough")]
                self.segmentation_pub.publish(self.cv_bridge.cv2_to_imgmsg(segmentation_image))

            else:
                self.segmentation_pub.publish(service_response.segmentation)

        elif self.service_number == 1:
            """
            :param request: sensor_msgs/CompressedImage image
                            uint32[] points
                            uint8 add_points_result_to_segmentation
            ---
            :return response: sensor_msgs/Image segmentation
                              uint32[] point_segmentation_shape
                              uint8[] point_segmentation
            """
            if self.new_service:
                self.new_service = False
                rospy.loginfo(f"fast_sam_segmentation_points_service")

            service_request = FastSamSegmentationPointsRequest()
            service_request.image = msg
            service_request.points = list(np.asarray(self.points).reshape(-1))
            service_request.add_points_result_to_segmentation = 1
            service_response = self.fast_sam_segmentation_points_service(service_request)

            point_segmentation = np.array(list(service_response.point_segmentation),
                                          dtype=np.uint8).reshape(tuple(service_response.point_segmentation_shape))

            for j in range(point_segmentation.shape[0]):
                point_segmentation[0] = cv2.circle(point_segmentation[0], (self.points[j][0], self.points[j][1]), 5, (255, 255, 255), 2)
                if j == 0:
                    point_segmentation_stack = point_segmentation[0]
                else:
                    point_segmentation_stack = np.hstack((point_segmentation_stack, point_segmentation[j]))

            if self.display_segmentation_with_colours:
                segmentation_image = self.map_class_colours[self.cv_bridge.imgmsg_to_cv2(service_response.segmentation, desired_encoding="passthrough")]
                self.segmentation_pub.publish(self.cv_bridge.cv2_to_imgmsg(segmentation_image))

            else:
                self.segmentation_pub.publish(service_response.segmentation)

            self.segmentation_points_pub.publish(self.cv_bridge.cv2_to_imgmsg(point_segmentation_stack * 255))

        elif self.service_number == 2:
            """
            :param request: sensor_msgs/CompressedImage image
                            uint32[] boxes
                            uint8 add_boxes_result_to_segmentation
            ---
            :return response: sensor_msgs/Image segmentation
                              uint32[] boxes_segmentation_shape
                              uint8[] boxes_segmentation
            """
            if self.new_service:
                self.new_service = False
                rospy.loginfo(f"fast_sam_segmentation_boxes_service")

            service_request = FastSamSegmentationBoxesRequest()
            service_request.image = msg
            service_request.boxes = list(np.asarray(self.boxes).reshape(-1))
            service_request.add_boxes_result_to_segmentation = 1
            service_response = self.fast_sam_segmentation_boxes_service(service_request)

            boxes_segmentation = np.array(list(service_response.boxes_segmentation),
                                          dtype=np.uint8).reshape(tuple(service_response.boxes_segmentation_shape))

            for j in range(boxes_segmentation.shape[0]):
                boxes_segmentation[0] = cv2.rectangle(boxes_segmentation[0], (self.boxes[j][0], self.boxes[j][1]), (self.boxes[j][2], self.boxes[j][3]), (255, 255, 255), 2)
                if j == 0:
                    boxes_segmentation_stack = boxes_segmentation[0]
                else:
                    boxes_segmentation_stack = np.hstack((boxes_segmentation_stack, boxes_segmentation[j]))

            if self.display_segmentation_with_colours:
                segmentation_image = self.map_class_colours[
                    self.cv_bridge.imgmsg_to_cv2(service_response.segmentation, desired_encoding="passthrough")]
                self.segmentation_pub.publish(self.cv_bridge.cv2_to_imgmsg(segmentation_image))

            else:
                self.segmentation_pub.publish(service_response.segmentation)

            self.segmentation_boxes_pub.publish(self.cv_bridge.cv2_to_imgmsg(boxes_segmentation_stack * 255))

        elif self.service_number == 3:
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
            if self.new_service:
                self.new_service = False
                rospy.loginfo(f"fast_sam_segmentation_robohop_service")

            service_request = FastSamSegmentationRoboHopRequest()
            service_request.image = msg
            service_request.traversable_point = list(np.asarray(self.points[0]).reshape(-1))
            service_response = self.fast_sam_segmentation_robohop_service(service_request)

            # Create a blank mask
            masks = np.zeros((service_response.segmentation_shape[0], service_response.segmentation_shape[1],
                                 service_response.segmentation_shape[2]), dtype=np.uint8)

            # Insert non-zero indices
            non_zero_indices = np.array(service_response.non_zeros_indices).reshape(service_response.non_zeros_indices_shape[0], service_response.non_zeros_indices_shape[1])
            masks[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = 1

            # Sum of non-zero elements in each mask
            mask_sums = service_response.mask_sums

            # Reshape from 1D to 2D [N, (u, v)]
            coordinates = np.array(service_response.coordinates).reshape(-1, 2)

            if self.display_segmentation_with_colours:
                segmentation_image = self.map_class_colours[
                    self.cv_bridge.imgmsg_to_cv2(service_response.segmentation, desired_encoding="passthrough")]
                self.segmentation_pub.publish(self.cv_bridge.cv2_to_imgmsg(segmentation_image))

            else:
                self.segmentation_pub.publish(service_response.segmentation)

            self.segmentation_traversable_pub.publish(service_response.traversable_segmentation)

        elif self.service_number == 4:
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
            if self.new_service:
                self.new_service = False
                rospy.loginfo(f"fast_sam_segmentation_robohop_service_clip")

            service_request = FastSamSegmentationRoboHopClipRequest()
            service_request.image = msg
            service_request.fast_sam_strings = ["floor"]
            service_response = self.fast_sam_segmentation_robohop_clip_service(service_request)

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

            if self.display_segmentation_with_colours:
                segmentation_image = self.map_class_colours[
                    self.cv_bridge.imgmsg_to_cv2(service_response.segmentation, desired_encoding="passthrough")]
                self.segmentation_pub.publish(self.cv_bridge.cv2_to_imgmsg(segmentation_image))

            else:
                self.segmentation_pub.publish(service_response.segmentation)

            self.segmentation_traversable_pub.publish(service_response.traversable_segmentation)

        self.service_counter += 1

        if self.service_counter >= 50:
            self.new_service = True
            self.service_counter = 0
            self.service_number += 1

            if self.service_number == 5:
                self.service_number = 0

    @staticmethod
    def vertical_to_channel_stack(vstack_mask, height):
        vstack_height, _ = vstack_mask.shape
        channels = vstack_height // height
        mask = np.dstack(np.vsplit(vstack_mask, channels)).transpose(2, 0, 1)
        return mask

    def ros_shutdown(self):
        rospy.loginfo(f"Shutting down ROS fast_sam_service_tester node")


def main():
    rospy.init_node('fast_sam_service_tester_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'fast_sam_service_tester_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    fsst = FastSamServiceTester(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

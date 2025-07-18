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

import yaml
import rospy
import numpy as np

from cv_bridge import CvBridge
from pathlib import Path

from sensor_msgs.msg import Image
from robohop_ros.msg import FastSAMDepthGoal


class RoboHopVisualsClass:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        # Publishers
        self.image_pub: rospy.Publisher = rospy.Publisher(settings_dict['image_publisher_topic'], Image, queue_size=1)
        self.segmentation_image_pub: rospy.Publisher = rospy.Publisher(settings_dict['segmentation_image_publisher_topic'], Image, queue_size=1)
        self.traversable_segmentation_image_pub: rospy.Publisher = rospy.Publisher(settings_dict['traversable_segmentation_image_publisher_topic'], Image, queue_size=1)
        self.monocular_depth_image_pub: rospy.Publisher = rospy.Publisher(settings_dict['monocular_depth_image_publisher_topic'], Image, queue_size=1)
        self.goal_image_pub: rospy.Publisher = rospy.Publisher(settings_dict['goal_image_publisher_topic'], Image, queue_size=1)
        self.localised_image_pub: rospy.Publisher = rospy.Publisher(settings_dict['localised_image_publisher_topic'], Image, queue_size=1)
        self.path_length_scaler = settings_dict['path_length_scaler']

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

        # Variables
        self.display_segmentation_with_colours = settings_dict['display_segmentation_with_colours']
        self.map_class_colours = np.random.randint(low=0, high=256, size=(256, 3), dtype=np.uint8)

        # Subscribers
        rospy.Subscriber(settings_dict['fast_sam_depth_goal_subscriber_topic'], FastSAMDepthGoal, self.fast_sam_depth_goal_callback)

    def fast_sam_depth_goal_callback(self, fast_sam_depth_goal_message: FastSAMDepthGoal):
        """ """
        image = self.cv_bridge.compressed_imgmsg_to_cv2(fast_sam_depth_goal_message.image)
        localised_image = self.cv_bridge.compressed_imgmsg_to_cv2(fast_sam_depth_goal_message.localised_image)

        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image))

        h, w, c = image.shape
        lh, lw, lc = localised_image.shape

        if h != lh or w != lw:
            localised_image = cv2.resize(src=localised_image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

        localised_image = cv2.putText(localised_image,
                                      f'{fast_sam_depth_goal_message.localised_image_id}',
                                      (30, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1.5,
                                      (0, 0, 255),
                                      2)

        self.localised_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.vstack((image, localised_image))))

        if self.display_segmentation_with_colours:
            self.segmentation_image_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(
                    self.map_class_colours[self.cv_bridge.imgmsg_to_cv2(fast_sam_depth_goal_message.segmentation)]))

        else:
            self.segmentation_image_pub.publish(fast_sam_depth_goal_message.segmentation)

        self.monocular_depth_image_pub.publish(fast_sam_depth_goal_message.depth)
        self.traversable_segmentation_image_pub.publish(fast_sam_depth_goal_message.traversable_segmentation)

        goal = self.cv_bridge.imgmsg_to_cv2(fast_sam_depth_goal_message.goal)
        goal = np.clip(a=goal * self.path_length_scaler, a_min=0, a_max=255).astype(np.uint8)
        goal = cv2.applyColorMap(src=goal, colormap=cv2.COLORMAP_INFERNO)

        for target_point in np.array(fast_sam_depth_goal_message.target_points).reshape(-1, 2):
            cv2.circle(goal, (target_point[1], target_point[0]), radius=4, color=(255, 255, 255), thickness=2)

        self.goal_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(
            cv2.applyColorMap(src=goal, colormap=cv2.COLORMAP_INFERNO)))

    def ros_shutdown(self):
        """ """
        rospy.loginfo(f"Shutting down ROS robohop_visuals node")


def main():
    """ """
    rospy.init_node('robohop_visuals_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'robohop_visuals_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)
    config_settings["maps_dir"] = root_dir / 'maps'

    rhvc = RoboHopVisualsClass(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

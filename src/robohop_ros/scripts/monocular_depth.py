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

from cv_bridge import CvBridge
from pathlib import Path

from sensor_msgs.msg import Image, CompressedImage
from src import DepthAnythingClass


class MonocularDepth:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        if settings_dict["model_type"] == "depth_anything":
            self.model = DepthAnythingClass(config_settings=settings_dict['depth_anything_settings'])
        else:
            rospy.logerr(f"Invalid monocular depth model specified")

        # Publishers
        self.publish_compressed = settings_dict["publish_compressed"]
        if self.publish_compressed:
            self.monocular_depth_pub: rospy.Publisher = rospy.Publisher(settings_dict['monocular_depth_compressed_publisher_topic'], CompressedImage, queue_size=1)
        else:
            self.monocular_depth_pub: rospy.Publisher = rospy.Publisher(settings_dict['monocular_depth_publisher_topic'], Image, queue_size=1)

        self.timestamp_threshold = settings_dict["timestamp_threshold"]
        self.normalise_image = settings_dict["normalise_image"]

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

        # Subscribers
        rospy.Subscriber(settings_dict['compressed_image_message_subscriber_topic'], CompressedImage, self.compressed_image_callback)
        rospy.Subscriber(settings_dict['image_message_subscriber_topic'], Image, self.image_callback)

    def image_callback(self, msg):
        if abs(rospy.get_rostime() - msg.header.stamp).to_sec() < self.timestamp_threshold:
            depth = self.model.infer(image=self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'),
                                     normalise=self.normalise_image)

            if self.publish_compressed:
                self.monocular_depth_pub.publish(self.cv_bridge.cv2_to_compressed_imgmsg(depth))
            else:
                self.monocular_depth_pub.publish(self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1"))
        else:
            rospy.loginfo_throttle(1, f"Mono Depth: timestamp over threshold")

    def compressed_image_callback(self, msg):
        if abs(rospy.get_rostime() - msg.header.stamp).to_sec() < self.timestamp_threshold:
            depth = self.model.infer(image=self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough'),
                                     normalise=self.normalise_image)

            if self.publish_compressed:
                self.monocular_depth_pub.publish(self.cv_bridge.cv2_to_compressed_imgmsg(depth))
            else:
                self.monocular_depth_pub.publish(self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1"))
        else:
            rospy.loginfo_throttle(1, f"Mono Depth: timestamp over threshold")

    def ros_shutdown(self):
        rospy.loginfo(f"Shutting down ROS monocular_depth node")


def main():
    rospy.init_node('monocular_depth_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'monocular_depth_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    md = MonocularDepth(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

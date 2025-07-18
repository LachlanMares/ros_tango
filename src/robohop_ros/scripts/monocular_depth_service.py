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
import ros_numpy

from cv_bridge import CvBridge
from pathlib import Path

from sensor_msgs.msg import PointCloud2
from src import DepthAnythingClass

from robohop_ros.srv import MonocularDepth, MonocularDepthResponse


class MonocularDepthService:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        if settings_dict["model_type"] == "depth_anything":
            self.model = DepthAnythingClass(config_settings=settings_dict['depth_anything_settings'])
        else:
            rospy.logerr(f"Invalid monocular depth model specified")

        self.monocular_depth_service = rospy.Service(settings_dict['depth_service_name'], MonocularDepth, self.handle_depth_service_request)

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

    def handle_depth_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
        ---
        :return response: sensor_msgs/Image depth
        """
        depth = self.model.infer(image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'))
        response = MonocularDepthResponse()
        response.depth = self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1")

        return response

    def ros_shutdown(self):
        rospy.loginfo(f"Shutting down ROS monocular_depth_service node")


def main():
    rospy.init_node('monocular_depth_service_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'monocular_depth_service_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    mds = MonocularDepthService(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

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

from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from robohop_ros.srv import MonocularDepth, MonocularDepthRequest
from robohop_ros.srv import MonocularMetricDepth, MonocularMetricDepthRequest
from robohop_ros.srv import MonocularMetricPointcloud, MonocularMetricPointcloudRequest
from robohop_ros.srv import MonocularMetricDepthAndPointcloud, MonocularMetricDepthAndPointcloudRequest


class MonocularDepthServiceTester:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        # Publishers
        self.monocular_depth_pub: rospy.Publisher = rospy.Publisher(settings_dict['monocular_depth_image_publisher_topic'], Image, queue_size=1)
        self.monocular_metric_depth_pub: rospy.Publisher = rospy.Publisher(settings_dict['monocular_metric_depth_image_publisher_topic'], Image, queue_size=1)
        self.depth_pointcloud_pub: rospy.Publisher = rospy.Publisher(settings_dict['monocular_depth_pointcloud_topic'], PointCloud2, queue_size=1)

        self.pointcloud_frame_id = settings_dict['monocular_depth_pointcloud_frame_id']
        self.camera_parameters = settings_dict['camera_parameters']
        self.frame_id = settings_dict['frame_id']

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

        # Service
        # rospy.wait_for_service(settings_dict['depth_service_name'], timeout=15.0)
        rospy.wait_for_service(settings_dict['metric_depth_service_name'], timeout=15.0)

        # self.monocular_depth_service = rospy.ServiceProxy(settings_dict['depth_service_name'], MonocularDepth)
        self.monocular_metric_depth_service = rospy.ServiceProxy(settings_dict['metric_depth_service_name'], MonocularMetricDepth)
        self.monocular_metric_pointcloud_service = rospy.ServiceProxy(settings_dict['metric_pointcloud_service_name'], MonocularMetricPointcloud)
        self.monocular_metric_depth_and_pointcloud_service = rospy.ServiceProxy(settings_dict['metric_depth_with_pointcloud_service_name'], MonocularMetricDepthAndPointcloud)

        self.service_counter = 0
        self.service_number = 1
        self.first_service_call = True

        rospy.Subscriber(settings_dict['image_message_subscriber_topic'], Image, self.image_callback)
        rospy.Subscriber(settings_dict['compressed_image_message_subscriber_topic'], CompressedImage, self.compressed_image_callback)

    def image_callback(self, msg):
        self.compressed_image_callback(msg=self.cv_bridge.cv2_to_compressed_imgmsg(self.cv_bridge.imgmsg_to_cv2(msg)))

    def compressed_image_callback(self, msg):
        if self.service_number == 0:
            if self.first_service_call:
                self.first_service_call = False
                rospy.loginfo(f"MonocularDepthRequest() Service")

            service_request = MonocularDepthRequest()
            service_request.image = msg

            service_response = self.monocular_depth_service(service_request)
            self.monocular_depth_pub.publish(service_response.depth)

        if self.service_number == 1:
            if self.first_service_call:
                self.first_service_call = False
                rospy.loginfo(f"MonocularMetricDepthRequest() Service")

            service_request = MonocularMetricDepthRequest()
            service_request.image = msg

            service_response = self.monocular_metric_depth_service(service_request)
            self.monocular_metric_depth_pub.publish(service_response.depth)

        elif self.service_number == 3:
            if self.first_service_call:
                self.first_service_call = False
                rospy.loginfo(f"MonocularMetricPointcloudRequest() Service")

            service_request = MonocularMetricPointcloudRequest()
            service_request.image = msg
            service_request.camera_parameters = self.camera_parameters
            service_request.frame_id = self.frame_id

            service_response = self.monocular_metric_pointcloud_service(service_request)

            self.depth_pointcloud_pub.publish(service_response.pointcloud)

        elif self.service_number == 4:
            if self.first_service_call:
                self.first_service_call = False
                rospy.loginfo(f"MonocularMetricDepthAndPointcloudRequest() Service")

            service_request = MonocularMetricDepthAndPointcloudRequest()
            service_request.image = msg
            service_request.camera_parameters = self.camera_parameters
            service_request.frame_id = self.frame_id

            service_response = self.monocular_metric_depth_and_pointcloud_service(service_request)

            self.monocular_metric_depth_pub.publish(service_response.depth)
            self.depth_pointcloud_pub.publish(service_response.pointcloud)

        self.service_counter += 1

        if self.service_counter == 50:
            self.first_service_call = True
            self.service_counter = 0
            self.service_number += 1

            if self.service_number == 5:
                self.service_number = 1

    def ros_shutdown(self):
        rospy.loginfo(f"Shutting down ROS monocular_depth_service_tester node")


def main():
    rospy.init_node('monocular_depth_service_tester_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'monocular_depth_service_tester_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    mdst = MonocularDepthServiceTester(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

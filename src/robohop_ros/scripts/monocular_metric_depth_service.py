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
from src import DepthAnythingMetricClass

from robohop_ros.srv import MonocularMetricDepth, MonocularMetricDepthResponse
from robohop_ros.srv import MonocularMetricPointcloud, MonocularMetricPointcloudResponse
from robohop_ros.srv import MonocularMetricDepthAndPointcloud, MonocularMetricDepthAndPointcloudResponse

class MonocularMetricDepthService:
    def __init__(self, settings_dict: dict):
        rospy.on_shutdown(self.ros_shutdown)

        if settings_dict["model_type"] == "depth_anything":
            self.model = DepthAnythingMetricClass(config_settings=settings_dict['depth_anything_metric_settings'])
        else:
            rospy.logerr(f"Invalid monocular metric depth model specified")

        self.depth_service = rospy.Service(settings_dict['metric_depth_service_name'], MonocularMetricDepth, self.handle_depth_service_request)
        self.depth_pointcloud_service = rospy.Service(settings_dict['metric_pointcloud_service_name'], MonocularMetricPointcloud, self.handle_depth_pointcloud_service_request)
        self.depth_image_and_pointcloud_service = rospy.Service(settings_dict['metric_depth_with_pointcloud_service_name'], MonocularMetricDepthAndPointcloud, self.handle_depth_and_pointcloud_service_request)

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()

    def handle_depth_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
        ---
        :return response: sensor_msgs/Image depth
        """
        depth = self.model.infer(image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'))
        response = MonocularMetricDepthResponse()
        response.depth = self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1")

        return response

    def handle_depth_pointcloud_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
                        float32[] camera_parameters
                        string frame_id
        ---
        :return response: sensor_msgs/PointCloud2 pointcloud
        """
        _, pointcloud = self.model.infer_with_pointcloud(image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'),
                                                         info=request.camera_parameters)
        pointcloud_msg = ros_numpy.msgify(PointCloud2, pointcloud)
        pointcloud_msg.header.frame_id = request.frame_id

        response = MonocularMetricPointcloudResponse()
        response.pointcloud = pointcloud_msg

        return response

    def handle_depth_and_pointcloud_service_request(self, request):
        """
        :param request: sensor_msgs/CompressedImage image
                        float32[] camera_parameters
                        string frame_id
        ---
        :return response: sensor_msgs/Image depth
                          sensor_msgs/PointCloud2 pointcloud
        """
        depth, pointcloud = self.model.infer_with_pointcloud(image=self.cv_bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding='passthrough'), info=request.camera_parameters)

        response = MonocularMetricDepthAndPointcloudResponse()
        response.depth = self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        response.pointcloud = ros_numpy.msgify(PointCloud2, pointcloud)
        response.pointcloud.header.frame_id = request.frame_id

        return response

    def ros_shutdown(self):
        rospy.loginfo(f"Shutting down ROS monocular_metric_depth_service node")


def main():
    rospy.init_node('monocular_metric_depth_service_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'monocular_metric_depth_service_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    mmds = MonocularMetricDepthService(settings_dict=config_settings)

    del config_settings

    rospy.spin()


if __name__ == "__main__":
    main()

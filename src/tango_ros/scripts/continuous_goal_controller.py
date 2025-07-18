#!/usr/bin/env python3
import os
import sys
"""
Author:
    Stefan Podgorski, stefan.podgorski@adelaide.edu.au

License:
    GPL-3.0

Description:

"""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import rospy
import yaml
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import cv2
import torchvision

from cv_bridge import CvBridge
from robohop_ros.msg import FastSAMDepthGoal
from robohop_ros.srv import RobohopFastSamStrings, RobohopFastSamStringsRequest
from nav_msgs.msg import Odometry
from src import RGBLEDColours, GoalControl, PID
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image

from tf.transformations import quaternion_from_euler


class ROSGoalControl:
    def __init__(self, config_settings: dict):
        # set up camera parameters
        self.original_height: int = 400
        self.original_width: int = 464

        focal_u: float = config_settings['intrinsics']['focal_u']
        focal_v: float = config_settings['intrinsics']['focal_v']
        center_u: float = config_settings['intrinsics']['center_u']
        center_v: float = config_settings['intrinsics']['center_v']

        intrinsics: torch.Tensor = self.build_intrinsics(
            focal_u=focal_u,
            focal_v=focal_v,
            center_u=center_u,
            center_v=center_v
        )

        # set up other bits and pieces for the controller
        self.default_velocity_control: float = config_settings['default_velocity_control']
        self.traversable_classes: list = config_settings['traversable_classes']
        self.grid_size_m: float = config_settings['grid_size']
        self.pid_steer_values: list = config_settings['pid_values']

        pid_steer: PID = PID(
            Kp=self.pid_steer_values[0],
            Ki=self.pid_steer_values[1],
            Kd=self.pid_steer_values[2]
        )

        self.goal_controller: GoalControl = GoalControl(
            traversable_classes=self.traversable_classes,
            pid_steer=pid_steer,
            default_velocity_control=self.default_velocity_control,
            h_image=self.original_height,
            w_image=self.original_width,
            hfov_rads=79 * np.pi / 180,
            intrinsics=intrinsics,
            grid_size=self.grid_size_m,
            device='cpu'
        )

        # CV Bridge
        self.cv_bridge: CvBridge = CvBridge()
        # subscribers
        self.segment_depth_goal_mask_subscriber: rospy.Subscriber = rospy.Subscriber(
            config_settings['fast_sam_depth_goal_subscriber_topic'],
            FastSAMDepthGoal,
            self.control_callback,
            queue_size=1
        )
        # publishers
        self.control_signal_publisher: rospy.Publisher = rospy.Publisher(
            config_settings['twist_control_publisher_topic'],
            Odometry,
            queue_size=1
        )
        self.led_colour_publisher: rospy.Publisher = rospy.Publisher(
            config_settings['rgb_led_topic'],
            UInt8MultiArray,
            queue_size=1
        )
        self.traversability_image_pub: rospy.Publisher = rospy.Publisher(
            config_settings['traversability_image_publisher_topic'], Image, queue_size=1)
        self.cost_map_image_pub: rospy.Publisher = rospy.Publisher(
            config_settings['cost_map_image_publisher_topic'], Image, queue_size=1)
        self.goal_image_pub: rospy.Publisher = rospy.Publisher(
            config_settings['goal_image_publisher_topic'], Image, queue_size=1)

        self.current_led_colour: str = 'black'
        self.controller_watchdog_timeout: bool = True
        self.led_colours = RGBLEDColours()
        self.rgb_led_msg: UInt8MultiArray = UInt8MultiArray()
        self.num_colours = len(list(self.led_colours.colours.keys())[2:6])
        self.colour_counter = 0
        self.theta_buffer = []
        self.buffer_limit = 5
        self.buffer_counter = 0

        rospy.wait_for_service(config_settings['service_name_robohop_fast_sam_strings'], timeout=5.0)
        self.update_fast_sam_strings_service = rospy.ServiceProxy(
            config_settings['service_name_robohop_fast_sam_strings'], RobohopFastSamStrings)

        self.update_fast_sam_strings(fast_sam_strings=["ground", "floor"])

        rospy.Timer(rospy.Duration(config_settings['watchdog_duration']), self.controller_watchdog_callback)

    def update_fast_sam_strings(self, fast_sam_strings: list) -> None:
        service_request = RobohopFastSamStringsRequest()
        service_request.fast_sam_strings = fast_sam_strings
        service_response = self.update_fast_sam_strings_service(service_request)

        if service_response.done == 0:
            rospy.logwarn(f"Update_fast_sam_strings error")
        else:
            rospy.loginfo(f"Update_fast_sam_strings success")

    def split_message(self, fast_SAM_depth_goal_msg: FastSAMDepthGoal) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_mask = torch.from_numpy(
            self.cv_bridge.imgmsg_to_cv2(fast_SAM_depth_goal_msg.traversable_segmentation))
        depth = torch.from_numpy(self.cv_bridge.imgmsg_to_cv2(fast_SAM_depth_goal_msg.depth))
        goal_mask = torch.from_numpy(self.cv_bridge.imgmsg_to_cv2(fast_SAM_depth_goal_msg.goal))
        goal_points = np.array(fast_SAM_depth_goal_msg.target_points).reshape(-1, 2)
        return semantic_mask, depth, goal_mask, goal_points

    @staticmethod
    def build_intrinsics(
            focal_u: float, focal_v: float,
            center_u: float, center_v: float) -> torch.Tensor:
        intrinsics = np.array([
            [focal_u, 0., center_u],
            [0., focal_v, center_v],
            [0., 0., 1]
        ])
        return torch.from_numpy(intrinsics)

    def control_callback(self, fast_SAM_depth_goal_msg: FastSAMDepthGoal) -> None:
        semantic_mask, depth, goal_mask, goal_point = self.split_message(fast_SAM_depth_goal_msg)
        # do the preprocessing and control
        velocity_control, theta_control, traversable_relative_bev_safe, cost_map_relative_bev_safe, point_poses, goal_bev = self.goal_controller.control(
            depth, semantic_mask, goal_mask, goal_point,
            time_delta=0.1)

        mask = (goal_mask == goal_mask.min()).cpu().numpy().astype(np.uint8)
        goal = mask * 255

        max_depth_indices = torch.where(depth == depth[mask].max())
        dd = torch.zeros_like(depth)
        dd[max_depth_indices] = 255
        goal = goal - dd.cpu().numpy().astype(np.uint8)
        # indices_goal_mask = mask[max_depth_indices]
        # pixel_goal = torch.stack(max_depth_indices)[indices_goal_mask.repeat(2, 1)]


        # positive theta rotate towards the left so we want to negate the control
        theta_control = -theta_control
        if len(self.theta_buffer) < self.buffer_limit:
            self.theta_buffer.append(theta_control)
        else:
            self.theta_buffer[self.buffer_counter] = theta_control
            self.buffer_counter += 1
            self.buffer_counter = self.buffer_counter % self.buffer_limit
        theta_control = np.median(np.array(self.theta_buffer))
        h, w = traversable_relative_bev_safe.shape
        point_poses = (point_poses / self.goal_controller.grid_size).astype(int)
        traversable_relative_bev_safe = cv2.cvtColor((traversable_relative_bev_safe * 255).astype(np.uint8),
                                                     cv2.COLOR_BGR2RGB)
        point_poses[:, 0] += w // 2
        traversable_relative_bev_safe[point_poses[:, 1], -point_poses[:, 0]] = (0, 0, 255)
        traversable_relative_bev_safe[goal_bev[1], goal_bev[0]] = (255, 0, 0)
        self.cost_map_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.flipud(cost_map_relative_bev_safe)))
        self.traversability_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.flipud(traversable_relative_bev_safe)))
        self.goal_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(goal))
        # todo: remove
        # todo: we now get a spinny guy.....
        # velocity_control = 0
        # theta_control = 0
        # pack into message
        msg_control = Odometry()
        msg_control.header.stamp = fast_SAM_depth_goal_msg.header.stamp

        q = quaternion_from_euler(0.0, 0.0, 0.0)  # todo: something fancy here

        msg_control.pose.pose.orientation.w = q[3]
        msg_control.pose.pose.orientation.x = q[0]
        msg_control.pose.pose.orientation.y = q[1]
        msg_control.pose.pose.orientation.z = q[2]

        msg_control.twist.twist.linear.x = velocity_control
        msg_control.twist.twist.linear.y = 0.0
        msg_control.twist.twist.linear.z = 0.0

        msg_control.twist.twist.angular.x = 0.0
        msg_control.twist.twist.angular.y = 0.0
        msg_control.twist.twist.angular.z = theta_control

        self.control_signal_publisher.publish(msg_control)

        if self.controller_watchdog_timeout:
            self.controller_watchdog_timeout = False
            if velocity_control > 0:
                if self.current_led_colour != 'green':
                    self.publish_led_message(colour='green')

        if velocity_control == 0:
            # cycle colours when not moving i.e. when at the goal
            colour_index = self.colour_counter % self.num_colours
            self.current_led_colour = list(self.led_colours.colours.keys())[2:6][colour_index]
            self.publish_led_message(colour=self.current_led_colour)
            self.colour_counter += 1

    def controller_watchdog_callback(self, _):
        if self.controller_watchdog_timeout:
            odom_msg = Odometry()
            odom_msg.pose.pose.orientation.w = 1.0
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = 0.0

            self.control_signal_publisher.publish(odom_msg)
            if self.current_led_colour != 'red':
                self.publish_led_message(colour='red')

        else:
            self.controller_watchdog_timeout = True

    def publish_led_message(self, colour: str):
        self.rgb_led_msg.data, self.current_led_colour = self.led_colours.get_colour(colour=colour)
        self.led_colour_publisher.publish(self.rgb_led_msg)


def main():
    rospy.init_node('continuous_goal_controller_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(
        open(str(root_dir / 'config' / 'goal_control_parameters.yaml'), 'r'),
        Loader=yaml.SafeLoader
    )

    goal_control = ROSGoalControl(config_settings)

    rospy.spin()


if __name__ == "__main__":
    main()

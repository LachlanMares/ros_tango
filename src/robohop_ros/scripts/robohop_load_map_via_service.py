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

from pathlib import Path
from robohop_ros.srv import RobohopLoadMap, RobohopLoadMapRequest


def main():
    """ """
    rospy.init_node('robohop_load_map_via_service_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'robohop_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)

    # Services
    rospy.wait_for_service(config_settings['service_name_robohop_load_map'], timeout=15.0)
    load_map_service = rospy.ServiceProxy(config_settings['service_name_robohop_load_map'], RobohopLoadMap)

    rospy.loginfo(f"Calling robohop load map service: {config_settings['load_map']}")

    map_request = RobohopLoadMapRequest()
    map_request.maps_id = config_settings[config_settings['load_map']]['maps_id']
    map_request.graph_id = config_settings[config_settings['load_map']]['graph_id']
    map_request.goal_node_indices = config_settings[config_settings['load_map']]['goal_node_indices']
    map_service_response = load_map_service(map_request)

    if map_service_response.done > 0:
        rospy.loginfo(f"Map loaded")
    else:
        rospy.loginfo(f"Map failed to load")


if __name__ == "__main__":
    main()

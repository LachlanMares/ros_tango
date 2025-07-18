#!/usr/bin/env python3
"""
Author:
    Lachlan Mares, lachlan.mares@adelaide.edu.au

License:
    GPL-3.0

Description:

"""

import rospy
import cv2
import yaml
import numpy as np

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from pathlib import Path


class ImageMsgToVideo:
    def __init__(self, parameters: dict, image_dir: Path, video_dir: Path, camera_id: str):
               
        self.camera_id = camera_id
        self.frame_count = 0
        self.bridge = CvBridge()
        self.save_images = parameters["image_sequence"]
        
        if self.save_images:
            self.image_directory = image_dir / camera_id
            
            if not self.image_directory.exists():
                self.image_directory.mkdir(parents=True, exist_ok=True)
            
            rospy.loginfo(f'{self.camera_id} images will be saved')

        self.video_writer = None
        self.encode_video = parameters["video_sequence"]
        
        if self.encode_video:
            self.fps = parameters["video_encode_fps"]
            self.video_filename = str(video_dir / parameters["filename"]) 
            
            rospy.loginfo(f'{self.camera_id} image to video writer created')

        if parameters["compressed"]:
            self.image_sub = rospy.Subscriber(parameters["compressed_topic"], CompressedImage, self.compressed_image_callback)
        else:
            self.image_sub = rospy.Subscriber(parameters["topic"], Image, self.image_callback)

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        except Exception as e:
            rospy.logerr(e)
            return
        
        if self.save_images:
            cv2.imwrite(str(self.image_directory / f"{self.frame_count}.png"), cv_image)

        if self.encode_video:
            if self.video_writer is None:
                height, width, _ = cv_image.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.fps, (width, height))
            
            self.video_writer.write(cv_image)
        
        self.frame_count += 1

    def compressed_image_callback(self, image_msg):
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        
        except Exception as e:
            rospy.logerr(e)
            return
        
        if self.save_images:
            cv2.imwrite(str(self.image_directory / f"{self.frame_count}.png"), cv_image)

        if self.encode_video:
            if self.video_writer is None:
                height, width, _ = cv_image.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.fps, (width, height))
            
            self.video_writer.write(cv_image)
        
        self.frame_count += 1
        
    def end(self):
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo(f'{self.camera_id} video writer released. Processed {self.frame_count} frames')


if __name__ == '__main__':
    rospy.init_node('robohop_multiple_video_creator_node')

    root_dir = Path(__file__).resolve().parents[1]
    config_settings = yaml.load(open(str(root_dir / 'config' / 'robohop_multiple_video_creator_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)
    creator_directory = root_dir / 'creator'

    if not creator_directory.exists():
        creator_directory.mkdir(parents=True, exist_ok=True)

    target_directory = creator_directory / config_settings["target_directory"]

    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)

    image_directory = target_directory / "images"
    video_directory = target_directory / "videos"

    if not image_directory.exists():
        image_directory.mkdir(parents=True, exist_ok=True)

    if not video_directory.exists():
        video_directory.mkdir(parents=True, exist_ok=True)

    creators = []
    creator_options = config_settings["creator_options"]
    
    for creator_key in creator_options.keys():
        creator_parameters = config_settings["creator_options"][creator_key]

        if creator_parameters["enable"]:
            creators.append(ImageMsgToVideo(parameters=creator_parameters, image_dir=image_directory, video_dir=video_directory, camera_id=creator_key))

    rospy.spin()

    for creator in creators:
        creator.end()

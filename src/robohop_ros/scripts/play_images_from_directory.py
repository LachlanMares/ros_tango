#!/usr/bin/env python3
"""
Author:
    Lachlan Mares, lachlan.mares@adelaide.edu.au

License:
    GPL-3.0

Description:

"""
import yaml
import rospy
import cv2

from cv_bridge import CvBridge
from pathlib import Path

from sensor_msgs.msg import Image, CompressedImage


class ImagesFromDirectoryPlayer:
    def __init__(self, config_settings: dict):
        rospy.on_shutdown(self.ros_shutdown)
        self.images_dir = config_settings["images_dir"]
        self.num_images = len([p for p in self.images_dir.iterdir() if p.suffix==config_settings["file_suffix"]])
        self.done = False
        self.images_index = 0
        self.publish_compressed = config_settings["publish_compressed"]
        self.cv_bridge: CvBridge = CvBridge()

        # Publishers
        if self.publish_compressed:
            self.image_pub: rospy.Publisher = rospy.Publisher(config_settings['compressed_image_publisher_topic'], CompressedImage, queue_size=1)
        else:
            self.viewable_image_pub: rospy.Publisher = rospy.Publisher(config_settings['image_publisher_topic'], Image, queue_size=1)

        self.frame_duration = config_settings["frame_duration"]


    def publisher_loop(self):
        while not self.done:
            frame = cv2.imread(str(self.images_dir / f"{self.images_index}.png"))

            if self.publish_compressed:
                image_message = self.cv_bridge.cv2_to_compressed_imgmsg(frame)
            else:
                image_message = self.cv_bridge.cv2_to_imgmsg(frame)

            self.image_pub.publish(image_message)

            self.images_index += 1

            rospy.sleep(self.frame_duration)

            if self.images_index == self.num_images:
                self.done = True

    @staticmethod
    def ros_shutdown():
        rospy.loginfo(f"Shutting down images_from_directory node")


def main():
    rospy.init_node('play_images_from_directory_node')

    root_dir = Path(__file__).resolve().parents[1]
    images_dir = root_dir / 'images'
    config_settings = yaml.load(open(str(root_dir / 'config' / 'images_from_directory_parameters.yaml'), 'r'), Loader=yaml.SafeLoader)
    config_settings['images_dir'] = images_dir / config_settings["run_id"]

    ifdp = ImagesFromDirectoryPlayer(config_settings=config_settings)

    del config_settings

    ifdp.publisher_loop()


if __name__ == "__main__":
    main()

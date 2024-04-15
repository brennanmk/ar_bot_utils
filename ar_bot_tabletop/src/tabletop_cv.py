#!/usr/bin/env python3

"""

"""

import rospy

import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image


class TabletopCV:
    """
    subscribes to raw camera feed and does some image processing magic
    to create a simulated lidar, also records configurations.
    """

    def __init__(self) -> None:
        rospy.init_node("simulated_lidar")

        self.cv_bridge = CvBridge()

        self.color_sub = rospy.Subscriber(
            "camera/color/image_raw", Image, self.proccess
        )

        rospy.spin()

    def proccess(self, color: Image) -> None:
        """
        generates a lidar image from raw camera image
        """

        # convert images to cv
        cv_image = self.cv_bridge.imgmsg_to_cv2(color, "mono8")


if __name__ == "__main__":
    try:
        TabletopCV()
    except rospy.ROSException:
        pass

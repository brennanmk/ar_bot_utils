#!/usr/bin/env python3

"""
Responsible for simulating a lidar reader from top down camera

Sources:
    https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    https://medium.com/analytics-vidhya/opencv-feature-matching-sift-algorithm-scale-invariant-feature-transform-16672eafb253
"""

import rospy
import message_filters

import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image


class SimulatedLidar:
    """
    subscribes to raw camera feed and does some image processing magic
    to create a simulated lidar measurement.
    """

    def __init__(self) -> None:
        rospy.init_node("simulated_lidar")

        self.sift = cv.SIFT_create()

        self.image_to_match = cv.imread("qr.png", cv.IMREAD_GRAYSCALE)

        self.source_key_points, self.source_description = self.sift.detectAndCompute(
            self.image_to_match, None
        )
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        self.flann = cv.FlannBasedMatcher(index_params, search_params)

        self.cv_bridge = CvBridge()

        self.color_sub = message_filters.Subscriber("camera/color/image_raw", Image)

        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )

        self.image_sync = message_filters.TimeSynchronizer(
            [self.color_sub, self.depth_sub], 10
        )
        self.image_sync.registerCallback(self.generate_lidar_from_image)

        rospy.spin()

    def generate_lidar_from_image(self, color: Image, depth: Image) -> None:
        """
        generates a lidar image from raw camera image
        """

        # convert images to cv representation
        color_image = self.cv_bridge.imgmsg_to_cv2(color, "mono8")
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth)

        # Use ORB to find robot in image
        destination_key_points, destination_description = self.sift.detectAndCompute(
            color_image, None
        )

        matches = self.flann.knnMatch(
            self.source_description, destination_description, k=2
        )

        filtered_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(filtered_matches) > 10:
            src_pts = np.float32(
                [self.source_key_points[m.queryIdx].pt for m in filtered_matches]
            ).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [destination_key_points[m.trainIdx].pt for m in filtered_matches]
            ).reshape(-1, 1, 2)

            homography_matrix, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            pts = np.float32([[0, 0]]).reshape(-1, 1, 2)

            dst = cv.perspectiveTransform(pts, homography_matrix)
            print(dst)
        else:
            print("No matches found")


if __name__ == "__main__":
    try:
        SimulatedLidar()
    except rospy.ROSException:
        pass

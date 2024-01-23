"""
Responsible for simulating a lidar reader from top down camera
"""

import rospy
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image


class SimulatedLidar:
    """
    subscribes to raw camera feed and does some image processing magic
    to create a simulated lidar measurement.
    """

    def __init__(self) -> None:
        rospy.init_node("simulated_lidar")

        self.cv_bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "camera/image/raw", Image, self.generate_lidar_from_image
        )

    def generate_lidar_from_image(self, data: Image):
        """
        generates a lidar image from raw camera image
        """

        # convert ROS image to cv representation
        cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")


if __name__ == "__main__":
    try:
        SimulatedLidar()
    except rospy.ROSException:
        pass

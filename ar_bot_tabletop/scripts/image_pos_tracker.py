#!/usr/bin/env python3

"""
Responsible for tracking robot and obstacle posistions relative to the env.

Sources:
    https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    https://medium.com/analytics-vidhya/opencv-feature-matching-sift-algorithm-scale-invariant-feature-transform-16672eafb253
"""


import cv2 as cv
import numpy as np
from typing import Tuple
from typing import Union


class ImagePosTracker:
    """
    Tracks a marker using SIFT, uses homography matrix to get relative pos

    :param reference_image_path: file path referance image
    :param transform_coordinates: coordinates to get location of in frame
    :param k: k for KNN
    :param minimum_distance: min distance of knn considered as valid match
    :param minimum_matches: how many matches need to be made to be considered as valid
    :param flann_trees: number of flann trees to use
    :param flann_checks: number of flann checks
    """

    def __init__(
        self,
        reference_image_path: str,
        transform_coordinates: Tuple[int, int],
        k: int = 2,
        minimum_distance: int = 0.7,
        minimum_matches: int = 10,
        flann_trees: int = 5,
        flann_checks: int = 50,
    ) -> None:
        self._k = k
        self._minimum_distance = minimum_distance
        self._minimum_matches = minimum_matches
        self._transform_coordinates = np.float32([transform_coordinates]).reshape(
            -1, 1, 2
        )

        self._sift = cv.SIFT_create()

        image_to_match = cv.imread(reference_image_path, cv.IMREAD_GRAYSCALE)

        self._source_key_points, self._source_description = self._sift.detectAndCompute(
            image_to_match, None
        )

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
        search_params = dict(checks=flann_checks)

        self._flann = cv.FlannBasedMatcher(index_params, search_params)

    @property
    def transform_coordinates(self):
        return self._transform_coordinates

    @transform_coordinates.setter
    def transform_coordinates(self, value):
        self._transform_coordinates = value

    def track_image(self, image: np.ndarray) -> Union[Tuple[int, int], None]:
        """
        generates a lidar image from raw camera image, will either return coordinates of
        "transform_coordinates" parameter, or None

        :param image: camera image to perform keypoint matching on
        """
        destination_key_points, destination_description = self._sift.detectAndCompute(
            image, None
        )

        matches = self._flann.knnMatch(
            self._source_description, destination_description, k=self._k
        )

        filtered_matches = [
            m for m, n in matches if m.distance < self._minimum_distance * n.distance
        ]

        if len(filtered_matches) > self._minimum_matches:
            src_pts = np.float32(
                [self._source_key_points[m.queryIdx].pt for m in filtered_matches]
            ).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [destination_key_points[m.trainIdx].pt for m in filtered_matches]
            ).reshape(-1, 1, 2)

            homography_matrix, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            return cv.perspectiveTransform(
                self._transform_coordinates, homography_matrix
            )

        return None

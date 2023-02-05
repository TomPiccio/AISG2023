# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions for drawing pose keypoints and connections
"""

from typing import Any, Iterable, Tuple, Union

import cv2
import numpy as np

from peekingduck.pipeline.nodes.draw.utils.constants import (
    CHAMPAGNE,
    POINT_RADIUS,
    THICK,
    TOMATO,
)
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)


def draw_human_poses(
    image: np.ndarray, keypoints: np.ndarray, keypoint_conns: np.ndarray
) -> None:
    # pylint: disable=too-many-arguments
    """Draw poses onto an image frame.

    Args:
        image (np.array): image of current frame
        keypoints (List[Any]): list of keypoint coordinates
        keypoints_conns (List[Any]): list of keypoint connections
    """
    image_size = get_image_size(image)
    num_persons = keypoints.shape[0]
    if num_persons > 0:
        for i in range(num_persons):
            _draw_connections(image, keypoint_conns[i], image_size, CHAMPAGNE)
            _draw_keypoints(image, keypoints[i], image_size, TOMATO, POINT_RADIUS)

def _draw_connections(
    frame: np.ndarray,
    keypoints: list,
    image_size: Tuple[int, int]
) -> None:
    """Draw connections between detected keypoints"""
    ind=[[5,6],[5,7],[7,9],[6,8],[6,10],[6,12],[11,12],[5,11],[12,14],[11,13],[14,16],[13,15]]
    for connection in ind:
        a,b = connection
        pts = np.array([keypoints[a],keypoints[b]])
        pt1, pt2 = project_points_onto_original_image(pts, image_size)
        cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0,0,0), int(50*scale))


def _draw_keypoints(
    frame: np.ndarray,
    keypoints: list,
    image_size: Tuple[int, int],
    keypoint_dot_color: Tuple[int, int, int],
    keypoint_dot_radius: int 
) -> None:
    # pylint: disable=too-many-arguments
    """Draw detected keypoints"""
    img_keypoints = project_points_onto_original_image(keypoints, image_size)
    n = 0
    for _, keypoint in enumerate(img_keypoints):
        if n == 0:
            _draw_one_keypoint_dot(frame, keypoint, (255,247,136), int(60*scale))
        #elif n>4:
            #_draw_one_keypoint_dot(frame, keypoint, (0,0,0), keypoint_dot_radius)
        n+=1


def _draw_one_keypoint_dot(
    frame: np.ndarray,
    keypoint: np.ndarray,
    keypoint_dot_color: Tuple[int, int, int],
    keypoint_dot_radius: int,
) -> None:
    """Draw single keypoint"""
    cv2.circle(
        frame, (keypoint[0], keypoint[1]), keypoint_dot_radius, keypoint_dot_color, -1
    )

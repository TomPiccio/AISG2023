"""
Draws keypoints on a detected pose.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

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
            _draw_connections(image, keypoint_conns[i], image_size, (0,0,0))
            _draw_keypoints(image, keypoints[i], image_size, TOMATO, POINT_RADIUS)


def _draw_connections(
    frame: np.ndarray,
    connections: Union[None, Iterable[Any]],
    image_size: Tuple[int, int],
    connection_color: Tuple[int, int, int],
) -> None:
    """Draw connections between detected keypoints"""
    if connections is not None:
        for connection in connections:
            pt1, pt2 = project_points_onto_original_image(connection, image_size)
            cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), connection_color, 50)



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
            _draw_one_keypoint_dot(frame, keypoint, (255,247,136), int(60))
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


class Node(AbstractNode):
    """Draws poses onto image.

    The :mod:`draw.poses` node uses the :term:`keypoints`,
    :term:`keypoint_scores`, and :term:`keypoint_conns` predictions from pose
    models to draw the human poses onto the image. For better understanding,
    check out the pose models such as :mod:`HRNet <model.hrnet>` and
    :mod:`PoseNet <model.posenet>`.

    Inputs:
        |img_data|

        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

    Outputs:
        |none_output_data|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.keypoint_dot_color = tuple(self.config["keypoint_dot_color"])
        self.keypoint_connect_color = tuple(self.config["keypoint_connect_color"])
        self.keypoint_text_color = tuple(self.config["keypoint_text_color"])

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws pose details onto input image.

        Args:
            inputs (dict): Dictionary with keys "img", "keypoints", and
                "keypoint_conns".
        """
        draw_human_poses(inputs["img"], inputs["keypoints"], inputs["keypoint_conns"])
        return {}

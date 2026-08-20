from dataclasses import dataclass, field
from typing import Dict, Tuple
import cv2
import numpy as np

# Adikus numbering used by this app:
#
# TOP:     13 14 15 16 17 18 | 19 20 21 22 23 24
# BOTTOM:  12 11 10  9  8  7 |  6  5  4  3  2  1
#
# In the user's Adikus screenshots:
#   top-right = 24
#   bottom-left = 12
#   bottom-right = 1
#
# These normalized x positions are measured from the full-screen
# 709x1536 Adikus layout used by the regression screenshot.
X_NORM = [
    40 / 709, 94 / 709, 148 / 709, 201 / 709, 255 / 709, 309 / 709,
    399 / 709, 453 / 709, 507 / 709, 561 / 709, 615 / 709, 669 / 709,
]

TOP_Y = 390 / 1536
BOTTOM_Y = 1238 / 1536
STACK_STEP = 54.5 / 1536

WHITE_THRESHOLD = 175.0
BLACK_THRESHOLD = 55.0
MAX_CHECKERS_PER_POINT = 15


@dataclass
class Position:
    board_rect: Tuple[int, int, int, int]
    white_points: Dict[int, int] = field(default_factory=dict)
    black_points: Dict[int, int] = field(default_factory=dict)
    white_bar: int = 0
    black_bar: int = 0

    @property
    def white_on_board(self) -> int:
        return sum(self.white_points.values())

    @property
    def black_on_board(self) -> int:
        return sum(self.black_points.values())

    @property
    def white_total(self) -> int:
        return self.white_on_board + self.white_bar

    @property
    def black_total(self) -> int:
        return self.black_on_board + self.black_bar


def _find_board_rect(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Return the playable Adikus screenshot rectangle.

    Adikus screenshots used by this app are full-screen board captures.
    Using the full image avoids the previous HSV contour detector selecting
    the entire wood background and then applying incorrect vertical scaling.
    """
    h, w = image.shape[:2]

    # The supplied Adikus layout is a tall phone screenshot. Use the full
    # image for this layout; point coordinates are normalized below.
    if h / float(w) > 1.7:
        return 0, 0, w, h

    # Conservative fallback for a cropped board image.
    return int(w * 0.04), int(h * 0.20), int(w * 0.92), int(h * 0.70)


def _patch_features(gray: np.ndarray, cx: int, cy: int, radius: int = 13):
    y1 = max(0, cy - radius)
    y2 = min(gray.shape[0], cy + radius + 1)
    x1 = max(0, cx - radius)
    x2 = min(gray.shape[1], cx + radius + 1)
    roi = gray[y1:y2, x1:x2]
    return float(np.mean(roi)), float(np.std(roi))


def _classify(mean_value: float, std_value: float) -> str:
    if mean_value >= WHITE_THRESHOLD:
        return "W"

    # Empty Adikus triangles can be dark. Real black checkers have stronger
    # circular texture/contrast.
    if mean_value <= BLACK_THRESHOLD and std_value >= 12.0:
        return "B"

    return "."


def _point_xy(
    point: int,
    board_rect: Tuple[int, int, int, int],
    stack_index: int = 0,
) -> Tuple[int, int]:
    """Map an Adikus point number to a checker-center sample location."""
    x0, y0, bw, bh = board_rect

    if not 1 <= point <= 24:
        raise ValueError(f"Point must be 1..24, got {point}")

    if point <= 12:
        # Bottom half is numbered right-to-left: 12 ... 1.
        idx = 12 - point
        direction = -1
        base_y = y0 + BOTTOM_Y * bh
    else:
        # Top half is numbered left-to-right: 13 ... 24.
        idx = point - 13
        direction = +1
        base_y = y0 + TOP_Y * bh

    cx = int(round(x0 + X_NORM[idx] * bw))
    cy = int(round(base_y + direction * STACK_STEP * bh * stack_index))
    return cx, cy


def detect_position(image: np.ndarray) -> Position:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Screenshot must be an RGB image.")

    board_rect = _find_board_rect(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    white: Dict[int, int] = {}
    black: Dict[int, int] = {}

    for point in range(1, 25):
        values = []
        for stack_index in range(MAX_CHECKERS_PER_POINT):
            cx, cy = _point_xy(point, board_rect, stack_index)
            mean_value, std_value = _patch_features(gray, cx, cy)
            values.append(_classify(mean_value, std_value))

        # A physical stack is contiguous from the board edge.
        count = 0
        for value in values:
            if value == ".":
                break
            count += 1

        if values[:count].count("W") and values[:count].count("B"):
            raise ValueError(f"Point {point} appears to contain both colors.")

        if count:
            if values[0] == "W":
                white[point] = count
            else:
                black[point] = count

    position = Position(
        board_rect=board_rect,
        white_points=white,
        black_points=black,
    )

    if position.white_total > 15:
        raise ValueError(
            f"White has {position.white_total} checkers; maximum is 15."
        )
    if position.black_total > 15:
        raise ValueError(
            f"Black has {position.black_total} checkers; maximum is 15."
        )

    return position


def validate_position(position: Position) -> None:
    """Reject positions that cannot represent a normal 15-checker game."""
    if position.white_total > 15 or position.black_total > 15:
        raise ValueError("A player cannot have more than 15 checkers.")

    if any(p < 1 or p > 24 for p in position.white_points):
        raise ValueError("White checker detected outside points 1..24.")
    if any(p < 1 or p > 24 for p in position.black_points):
        raise ValueError("Black checker detected outside points 1..24.")

    if set(position.white_points) & set(position.black_points):
        raise ValueError("A point cannot contain both colors.")


def make_debug_image(image: np.ndarray, position: Position) -> np.ndarray:
    out = image.copy()

    for point in range(1, 25):
        count = position.white_points.get(
            point, position.black_points.get(point, 0)
        )

        for k in range(count):
            cx, cy = _point_xy(point, position.board_rect, k)
            cv2.circle(
                out, (cx, cy), max(8, int(0.018 * position.board_rect[2])),
                (0, 220, 0), 2
            )

        cx, cy = _point_xy(point, position.board_rect, 0)
        label_y = cy + (22 if point <= 12 else -22)
        cv2.putText(
            out, str(point), (cx - 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 80, 80), 1, cv2.LINE_AA
        )

    return out

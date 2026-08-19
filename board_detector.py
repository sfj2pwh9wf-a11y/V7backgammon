from dataclasses import dataclass, field
from typing import Dict, Tuple
import cv2
import numpy as np


# Adikus numbering supplied for this app:
# top-right = 1, top-left = 12,
# bottom-left = 13, bottom-right = 24.
#
# x positions are normalized to the board rectangle.
X_NORM = [
    0.0563, 0.1309, 0.2070, 0.2831, 0.3592, 0.4338,
    0.5616, 0.6377, 0.7139, 0.7884, 0.8645, 0.9391,
]
TOP_Y = 0.2053
BOTTOM_Y = 0.8685
STACK_STEP = 0.04155

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


def _find_board_rect(image: np.ndarray) -> Tuple[int, int, int, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # The Adikus board is a large brown region. This deliberately ignores
    # browser chrome and the dark Streamlit page around the screenshot.
    mask = cv2.inRange(
        hsv,
        np.array([5, 30, 30], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((31, 31), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        candidates = []
        h, w = image.shape[:2]
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 0.25 * w * h:
                continue
            ratio = bw / float(bh)
            if 0.35 < ratio < 0.85:
                candidates.append((area, x, y, bw, bh))
        if candidates:
            _, x, y, bw, bh = max(candidates)
            return int(x), int(y), int(bw), int(bh)

    # Safe fallback for an Adikus screenshot.
    h, w = image.shape[:2]
    return int(w * 0.04), int(h * 0.125), int(w * 0.92), int(h * 0.77)


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

    # Empty Adikus triangles can be very dark, especially along the bottom
    # edge. Real black checkers have a much stronger circular texture/contrast.
    if mean_value <= BLACK_THRESHOLD and std_value >= 12.0:
        return "B"

    return "."


def detect_position(image: np.ndarray) -> Position:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Screenshot must be an RGB image.")

    x0, y0, bw, bh = _find_board_rect(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    white: Dict[int, int] = {}
    black: Dict[int, int] = {}

    # We read each point from predetermined geometric locations instead of
    # asking vision to guess the point numbers. This prevents the old
    # 13/14 vs 23/24 numbering failure.
    for point in range(1, 25):
        if point <= 12:
            idx = 12 - point
            cx = int(round(x0 + X_NORM[idx] * bw))
            base_y = int(round(y0 + TOP_Y * bh))
            direction = +1
        else:
            idx = point - 13
            cx = int(round(x0 + X_NORM[idx] * bw))
            base_y = int(round(y0 + BOTTOM_Y * bh))
            direction = -1

        values = []
        for stack_index in range(MAX_CHECKERS_PER_POINT):
            cy = int(round(base_y + direction * STACK_STEP * bh * stack_index))
            mean_value, std_value = _patch_features(gray, cx, cy)
            values.append(_classify(mean_value, std_value))

        # A physical stack starts at the edge of the board and is contiguous.
        # Stop at the first empty location; if a later location looks occupied,
        # the point is marked ambiguous rather than inventing a checker count.
        count = 0
        for value in values:
            if value == ".":
                break
            count += 1

        # A stack is contiguous from the board edge. We intentionally ignore
        # later pixels after the first empty slot because the middle of the
        # board contains dice and other Adikus graphics that can resemble a
        # checker to a simple brightness test.
        if values[:count].count("W") and values[:count].count("B"):
            raise ValueError(f"Point {point} appears to contain both colors.")

        if count:
            if values[0] == "W":
                white[point] = count
            else:
                black[point] = count

    position = Position(
        board_rect=(x0, y0, bw, bh),
        white_points=white,
        black_points=black,
    )

    if position.white_on_board > 15:
        raise ValueError(
            f"White has {position.white_on_board} checkers on the board/bar; maximum is 15."
        )
    if position.black_on_board > 15:
        raise ValueError(
            f"Black has {position.black_on_board} checkers on the board/bar; maximum is 15."
        )

    occupied_points = set(white) | set(black)
    if any(p < 1 or p > 24 for p in occupied_points):
        raise ValueError("Detected a checker outside the 24-point board.")

    return position


def make_debug_image(image: np.ndarray, position: Position) -> np.ndarray:
    out = image.copy()
    x0, y0, bw, bh = position.board_rect

    for point in range(1, 25):
        if point <= 12:
            idx = 12 - point
            cx = int(round(x0 + X_NORM[idx] * bw))
            base_y = int(round(y0 + TOP_Y * bh))
            direction = +1
        else:
            idx = point - 13
            cx = int(round(x0 + X_NORM[idx] * bw))
            base_y = int(round(y0 + BOTTOM_Y * bh))
            direction = -1

        count = position.white_points.get(point, position.black_points.get(point, 0))
        is_white = point in position.white_points

        for k in range(count):
            cy = int(round(base_y + direction * STACK_STEP * bh * k))
            cv2.circle(out, (cx, cy), max(8, int(0.018 * bw)), (0, 220, 0), 2)

        cv2.putText(
            out, str(point), (cx - 10, base_y + (22 if point <= 12 else -22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA
        )

    return out

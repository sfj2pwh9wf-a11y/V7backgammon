from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class DiceResult:
    values: Optional[Tuple[int, int]]
    boxes: Tuple[Tuple[int, int, int, int], ...] = ()


def _count_pips(die: np.ndarray) -> int:
    gray = cv2.cvtColor(die, cv2.COLOR_RGB2GRAY)

    # Ignore the die border and look for the dark pips.
    inner = gray[7:-7, 7:-7]
    mask = cv2.threshold(inner, 95, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    pips = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 35 <= area <= 500 and 4 <= w <= 22 and 4 <= h <= 22:
            pips += 1

    return pips if 1 <= pips <= 6 else 0


def detect_dice(image: np.ndarray, board_rect) -> DiceResult:
    x0, y0, bw, bh = board_rect
    h, w = image.shape[:2]

    # Adikus dice are in the lower-middle/right portion of the board.
    rx1 = int(x0 + 0.50 * bw)
    rx2 = int(x0 + 0.98 * bw)
    ry1 = int(y0 + 0.46 * bh)
    ry2 = int(y0 + 0.68 * bh)

    roi = image[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # White die faces: high value, low-to-moderate saturation.
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 150], dtype=np.uint8),
        np.array([180, 105, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    scale = bw / 657.0
    min_side = max(35, int(45 * scale))
    max_side = int(100 * scale)

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ratio = cw / float(ch) if ch else 0

        if (
            min_side <= cw <= max_side
            and min_side <= ch <= max_side
            and 0.75 <= ratio <= 1.25
            and 1000 * scale * scale <= area <= 9000 * scale * scale
        ):
            candidates.append((rx1 + x, ry1 + y, cw, ch))

    # Remove near-duplicate detections and keep the two most likely dice.
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    selected = []
    for box in candidates:
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        if all((cx - (b[0] + b[2] / 2)) ** 2 + (cy - (b[1] + b[3] / 2)) ** 2 > (20 * scale) ** 2 for b in selected):
            selected.append(box)
        if len(selected) == 2:
            break

    selected.sort(key=lambda b: b[0])

    values = []
    good_boxes = []
    for x, y, cw, ch in selected:
        die = image[y:y + ch, x:x + cw]
        value = _count_pips(die)
        if value:
            values.append(value)
            good_boxes.append((x, y, cw, ch))

    if len(values) == 2:
        return DiceResult(tuple(values), tuple(good_boxes))

    return DiceResult(None, tuple(good_boxes))

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DiceResult:
    values: Optional[Tuple[int, int]]
    boxes: Tuple[Tuple[int, int, int, int], ...] = ()


def _count_pips(die: np.ndarray) -> int:
    """Count the dark pips inside one detected white die."""

    if die.size == 0:
        return 0

    gray = cv2.cvtColor(die, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]

    # Ignore the outer border/shadow of the die.
    margin = max(4, int(min(h, w) * 0.15))

    if h <= 2 * margin or w <= 2 * margin:
        return 0

    inner = gray[margin:h - margin, margin:w - margin]

    # Dark pips.
    mask = cv2.threshold(
        inner,
        120,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]

    # Remove tiny noise while preserving the round pips.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    # Scale the acceptable pip size according to die size.
    scale = min(w, h) / 55.0

    min_area = max(8, int(18 * scale * scale))
    max_area = max(80, int(350 * scale * scale))

    min_side = max(2, int(3 * scale))
    max_side = max(10, int(18 * scale))

    pips = 0

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        pw = stats[i, cv2.CC_STAT_WIDTH]
        ph = stats[i, cv2.CC_STAT_HEIGHT]

        if (
            min_area <= area <= max_area
            and min_side <= pw <= max_side
            and min_side <= ph <= max_side
        ):
            ratio = pw / float(ph) if ph else 0

            if 0.55 <= ratio <= 1.8:
                pips += 1

    return pips if 1 <= pips <= 6 else 0


def _candidate_score(
    gray: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> float:
    """Score how much a region looks like an Adikus die."""

    if w <= 0 or h <= 0:
        return -1.0

    crop = gray[y:y + h, x:x + w]

    if crop.size == 0:
        return -1.0

    mean_value = float(np.mean(crop))

    # Dice are bright.
    if mean_value < 135:
        return -1.0

    ratio = w / float(h)

    # Dice should be close to square.
    if not 0.70 <= ratio <= 1.30:
        return -1.0

    # Look at the central portion to avoid shadows/borders.
    mx = max(2, int(w * 0.15))
    my = max(2, int(h * 0.15))

    inner = crop[my:h - my, mx:w - mx]

    if inner.size == 0:
        return -1.0

    inner_mean = float(np.mean(inner))

    if inner_mean < 145:
        return -1.0

    # Brightness + squareness.
    square_score = 1.0 - abs(1.0 - ratio)

    return inner_mean * square_score


def detect_dice(
    image: np.ndarray,
    board_rect,
) -> DiceResult:
    """
    Detect the two white Adikus dice.

    This version deliberately searches a larger region than the original
    detector because the dice can move slightly between screenshots.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Screenshot must be an RGB image.")

    x0, y0, bw, bh = board_rect

    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ---------------------------------------------------------
    # Search region
    #
    # Adikus dice appear around the middle/lower portion of the
    # board. We intentionally search wider than the old detector.
    # This avoids depending on one exact screenshot position.
    # ---------------------------------------------------------

    rx1 = int(x0 + 0.35 * bw)
    rx2 = int(x0 + 0.98 * bw)

    ry1 = int(y0 + 0.40 * bh)
    ry2 = int(y0 + 0.70 * bh)

    roi_gray = gray[ry1:ry2, rx1:rx2]

    if roi_gray.size == 0:
        return DiceResult(None)

    # ---------------------------------------------------------
    # Bright-object mask
    # ---------------------------------------------------------

    mask = cv2.threshold(
        roi_gray,
        175,
        255,
        cv2.THRESH_BINARY,
    )[1]

    # Connect the bright die face despite the dark pips.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    # Expected die size relative to the 709-pixel reference screenshot.
    scale = bw / 709.0

    min_side = max(30, int(35 * scale))
    max_side = max(min_side + 10, int(90 * scale))

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        if cw < min_side or ch < min_side:
            continue

        if cw > max_side or ch > max_side:
            continue

        ratio = cw / float(ch)

        if not 0.70 <= ratio <= 1.30:
            continue

        box_area = float(cw * ch)

        if box_area <= 0:
            continue

        rectangularity = area / box_area

        # Reject very thin/irregular bright objects.
        if rectangularity < 0.55:
            continue

        gx = rx1 + x
        gy = ry1 + y

        score = _candidate_score(
            gray,
            gx,
            gy,
            cw,
            ch,
        )

        if score < 0:
            continue

        candidates.append(
            (
                score,
                gx,
                gy,
                cw,
                ch,
            )
        )

    # Highest scoring candidates first.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Select two separate dice.
    # ---------------------------------------------------------

    selected = []

    min_separation = max(25, int(25 * scale))

    for candidate in candidates:
        _, x, y, cw, ch = candidate

        cx = x + cw / 2
        cy = y + ch / 2

        too_close = False

        for existing in selected:
            _, ex, ey, ew, eh = existing

            ecx = ex + ew / 2
            ecy = ey + eh / 2

            distance = np.hypot(
                cx - ecx,
                cy - ecy,
            )

            if distance < min_separation:
                too_close = True
                break

        if not too_close:
            selected.append(
                (
                    candidate[0],
                    x,
                    y,
                    cw,
                    ch,
                )
            )

        if len(selected) == 2:
            break

    # We need exactly two dice.
    if len(selected) != 2:
        return DiceResult(
            None,
            tuple(
                (x, y, cw, ch)
                for _, x, y, cw, ch in selected
            ),
        )

    # ---------------------------------------------------------
    # Read the pips.
    # ---------------------------------------------------------

    # Put dice in left-to-right order.
    selected.sort(
        key=lambda item: item[1]
    )

    values = []
    boxes = []

    for _, x, y, cw, ch in selected:
        die = image[
            y:y + ch,
            x:x + cw,
        ]

        value = _count_pips(die)

        if value == 0:
            return DiceResult(
                None,
                tuple(
                    (sx, sy, sw, sh)
                    for _, sx, sy, sw, sh in selected
                ),
            )

        values.append(value)
        boxes.append(
            (x, y, cw, ch)
        )

    if len(values) != 2:
        return DiceResult(
            None,
            tuple(boxes),
        )

    return DiceResult(
        (int(values[0]), int(values[1])),
        tuple(boxes),
    )

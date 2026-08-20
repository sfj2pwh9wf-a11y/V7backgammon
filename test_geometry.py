from pathlib import Path
import numpy as np
from PIL import Image

from board_detector import (
    X_NORM,
    TOP_Y,
    BOTTOM_Y,
    STACK_STEP,
    _point_xy,
    detect_position,
)

# Authoritative Adikus numbering:
# top-right = 24, bottom-left = 12, bottom-right = 1.
assert len(X_NORM) == 12
assert X_NORM[0] < X_NORM[-1]

# The point centers must run left-to-right as:
# top:    13..24
# bottom: 12..1
assert _point_xy(13, (0, 0, 709, 1536))[0] < _point_xy(24, (0, 0, 709, 1536))[0]
assert _point_xy(12, (0, 0, 709, 1536))[0] > _point_xy(1, (0, 0, 709, 1536))[0]

# Regression fixture: the supplied 4-3 Adikus screenshot.
fixture = Path(__file__).with_name("fixtures") / "adikus_4_3.jpg"

if fixture.exists():
    image = np.array(Image.open(fixture).convert("RGB"))
    position = detect_position(image)

    expected_white = {
        6: 5,
        8: 4,
        10: 2,
        13: 2,
        23: 1,
        24: 1,
    }
    expected_black = {
        9: 1,
        12: 4,
        17: 2,
        19: 4,
        20: 2,
        21: 2,
    }

    assert position.white_points == expected_white, position.white_points
    assert position.black_points == expected_black, position.black_points
    assert position.white_bar == 0
    assert position.black_bar == 0
    assert position.white_total == 15
    assert position.black_total == 15

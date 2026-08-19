# Lightweight sanity checks for the fixed Adikus numbering.
from board_detector import X_NORM

assert len(X_NORM) == 12
assert X_NORM[0] < X_NORM[-1]

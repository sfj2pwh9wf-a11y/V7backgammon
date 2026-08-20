from dataclasses import dataclass
from typing import Dict, Tuple, List
import importlib


@dataclass
class EngineResult:
    move: List[Tuple[int, int]]
    position_id: str
    dice: Tuple[int, int]
    depth: int
    side: str


class EngineError(RuntimeError):
    pass


def _load_engine():
    try:
        return importlib.import_module("gnubg_nn")
    except Exception as exc:
        raise EngineError(
            "GNU Backgammon engine could not be loaded. "
            "Make sure requirements.txt uses the package 'gnubg-nn' "
            "and the deployment is using Python 3.11."
        ) from exc


def _to_gnubg_board(
    white_points: Dict[int, int],
    black_points: Dict[int, int],
    white_bar: int,
    black_bar: int,
):
    # GNUBG uses a 2x25 board. The first side is X, the second is O.
    # Index 0 is the bar; points 1..24 are the board.
    board = [[0] * 25 for _ in range(2)]

    for point, count in white_points.items():
        board[0][24-int(point)] = int(count)

    for point, count in black_points.items():
        board[1][24-int(point)] = int(count)

    board[0][0] = int(white_bar)
    board[1][0] = int(black_bar)

    return board


def _call_best_move(engine, board, d1, d2, depth, side):
    # Some older compiled builds require a one-byte side value while newer
    # builds accept a Python string. Try the documented form first, then the
    # byte-compatible form that fixes the previous deployment error.
    try:
        return engine.best_move(board, d1, d2, n=depth, s=side)
    except TypeError as exc:
        if "byte string" not in str(exc):
            raise
        return engine.best_move(board, d1, d2, n=depth, s=side.encode("ascii"))


def analyze_position(
    white_points: Dict[int, int],
    black_points: Dict[int, int],
    white_bar: int,
    black_bar: int,
    dice: Tuple[int, int],
    depth: int = 1,
    side: str = "X",
) -> EngineResult:
    if any(d < 1 or d > 6 for d in dice):
        raise EngineError("Dice must be between 1 and 6.")
    if depth not in (1, 2):
        raise EngineError("Analysis depth must be 1 or 2.")
    if side not in ("X", "O"):
        raise EngineError("Engine side must be X or O.")

    white_total = sum(white_points.values()) + white_bar
    black_total = sum(black_points.values()) + black_bar

    if white_total > 15 or black_total > 15:
        raise EngineError("The reconstructed position contains more than 15 checkers.")
    if any(p < 1 or p > 24 for p in white_points):
        raise EngineError("White contains an invalid point number.")
    if any(p < 1 or p > 24 for p in black_points):
        raise EngineError("Black contains an invalid point number.")
    if set(white_points) & set(black_points):
        raise EngineError("A point contains both colors.")

    board = _to_gnubg_board(
        white_points, black_points, white_bar, black_bar
    )

    engine = _load_engine()

    try:
        move = _call_best_move(
            engine, board, int(dice[0]), int(dice[1]), int(depth), side
        )
        position_id = engine.position_id(board)
    except Exception as exc:
        raise EngineError(f"GNU Backgammon rejected the verified position: {exc}") from exc

    # Normalize a few possible return shapes.
    if isinstance(move, tuple) and len(move) == 2 and all(
        isinstance(x, (tuple, list)) and len(x) == 2 for x in move
    ):
        move = list(move)
    elif isinstance(move, list):
        move = [tuple(x) for x in move]
    else:
        raise EngineError(f"Unexpected engine move format: {move!r}")

    return EngineResult(
        move=[(int(a), int(b)) for a, b in move],
        position_id=str(position_id),
        dice=(int(dice[0]), int(dice[1])),
        depth=int(depth),
        side=side,
    )


def _point_name(point: int) -> str:
    if point == 0:
        return "bar"
    if point == 25:
        return "off"
    return str(point)


def format_move(move: List[Tuple[int, int]]) -> str:
    return ", ".join(f"{_point_name(a)}/{_point_name(b)}" for a, b in move)

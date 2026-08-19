import streamlit as st
from PIL import Image
import numpy as np

from board_detector import detect_position, make_debug_image
from dice_detector import detect_dice
from engine import analyze_position, format_move, EngineError

st.set_page_config(
    page_title="Backgammon Analyzer",
    page_icon="🎲",
    layout="centered",
)

st.title("🎲 Backgammon Analyzer")
st.caption("Adikus screenshot → board reconstruction → automatic dice → GNUBG analysis")

uploaded = st.file_uploader(
    "Upload your latest Adikus screenshot",
    type=["png", "jpg", "jpeg"],
)

if not uploaded:
    st.info("Upload a screenshot of the current board.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img = np.array(image)

try:
    position = detect_position(img)
except Exception as exc:
    st.error(f"Could not read the board: {exc}")
    st.stop()

st.success("Board geometry detected")

with st.expander("Detected board", expanded=True):
    st.write("**WHITE**")
    st.code("  ".join(f"{p}:{n}" for p, n in position.white_points.items() if n))
    st.write("**BLACK**")
    st.code("  ".join(f"{p}:{n}" for p, n in position.black_points.items() if n))

    c1, c2 = st.columns(2)
    c1.metric("White on board", position.white_on_board)
    c2.metric("Black on board", position.black_on_board)

    if position.white_on_board + position.white_bar > 15:
        st.error("White reconstruction is impossible: more than 15 checkers.")
    if position.black_on_board + position.black_bar > 15:
        st.error("Black reconstruction is impossible: more than 15 checkers.")

    st.caption(
        "Point numbering is fixed to the Adikus layout: top-right = 1, "
        "top-left = 12, bottom-left = 13, bottom-right = 24."
    )

    st.image(
        make_debug_image(img, position),
        caption="Detected checker centers",
        use_container_width=True,
    )

st.subheader("Dice")

dice = detect_dice(img, position.board_rect)
if dice.values:
    st.success(f"Dice detected automatically: **{dice.values[0]} – {dice.values[1]}**")
    dice_values = dice.values
else:
    st.warning("Dice could not be read automatically.")
    st.caption("Use the manual fallback below only when the dice are not visible clearly.")

    c1, c2 = st.columns(2)
    d1 = c1.number_input("Die 1", min_value=1, max_value=6, value=1, step=1)
    d2 = c2.number_input("Die 2", min_value=1, max_value=6, value=1, step=1)
    dice_values = (int(d1), int(d2))

with st.expander("Bar / advanced correction"):
    st.caption(
        "The bar matters mathematically because a checker on the bar must enter "
        "before another checker can move. Leave these at zero when the bar is empty."
    )
    c1, c2 = st.columns(2)
    white_bar = c1.number_input(
        "White on bar", min_value=0, max_value=15,
        value=position.white_bar, step=1
    )
    black_bar = c2.number_input(
        "Black on bar", min_value=0, max_value=15,
        value=position.black_bar, step=1
    )

st.subheader("Analysis")
depth = st.selectbox(
    "Analysis depth",
    options=[1, 2],
    index=0,
    format_func=lambda x: "1-ply — strong and fast" if x == 1 else "2-ply — deeper, slower",
)

if st.button("Find strongest move", type="primary", use_container_width=True):
    if not dice_values or len(dice_values) != 2:
        st.error("Two dice are required.")
        st.stop()

    if position.white_on_board + int(white_bar) > 15:
        st.error("White reconstruction is invalid.")
        st.stop()
    if position.black_on_board + int(black_bar) > 15:
        st.error("Black reconstruction is invalid.")
        st.stop()

    try:
        result = analyze_position(
            white_points=position.white_points,
            black_points=position.black_points,
            white_bar=int(white_bar),
            black_bar=int(black_bar),
            dice=(int(dice_values[0]), int(dice_values[1])),
            depth=int(depth),
            side="X",  # White is the player to move.
        )
    except EngineError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"The engine could not analyze this verified position: {exc}")
        st.stop()

    st.success("BEST MOVE")
    st.markdown(f"# {format_move(result.move)}")

    if result.position_id:
        with st.expander("Engine/debug information"):
            st.code(result.position_id)
            st.write(f"Engine side: `{result.side}`")
            st.write(f"Dice: `{result.dice[0]}–{result.dice[1]}`")
            st.write(f"Depth: `{result.depth}-ply`")

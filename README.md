# Backgammon Analyzer v3

This version fixes the two problems that caused the previous deployments to fail:

1. The board detector no longer guesses point numbering from detected checkers.
   The Adikus numbering is hard-coded:
   - top-right = 1
   - top-left = 12
   - bottom-left = 13
   - bottom-right = 24

2. Dice are detected automatically from the screenshot.
   Manual dice entry is only a fallback when the dice cannot be read.

## Important engine change

Use **`gnubg-nn`**, not `gnubg`, in `requirements.txt`.

The app also handles both string and byte-style side arguments so the previous
`argument 5 must be a byte string of length 1, not str` error does not stop analysis.

## Repository layout

Upload these files/folders to the root of your GitHub repository:

- `app.py`
- `board_detector.py`
- `dice_detector.py`
- `engine.py`
- `requirements.txt`
- `runtime.txt`
- `.python-version`
- `.streamlit/config.toml`

Do not upload the ZIP itself as the only file. Streamlit must see `app.py` and
`requirements.txt` in the repository.

## How the screenshot is processed

Screenshot
→ locate Adikus board
→ use fixed 24-point geometry
→ read checker stacks
→ validate checker totals
→ detect dice
→ build a 2×25 GNUBG position
→ generate the strongest legal move

The app refuses to call the engine when the reconstructed board is impossible.

## Analysis depth

- 1-ply is the default and is intended to be fast while considering the
  opponent's next roll.
- 2-ply is available for deeper analysis but will take longer.

## Bar

The bar is retained internally because it changes legal moves. It is kept under
an Advanced/Bar expander rather than cluttering the normal screen. If a checker
is on the bar and automatic detection does not identify it, enter the count
there before running the engine.

## Deployment

On Streamlit Community Cloud, select:

- Repository: your GitHub repository
- Branch: `main`
- Main file: `app.py`

If Streamlit shows an installation error, check the deployment log for the
first package error. Do not change the board code until dependencies install
successfully.

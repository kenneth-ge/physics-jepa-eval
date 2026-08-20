"""Contact sheet of a ladder's clips, written into the case dir at render time.

A continuous case is a sequence of clips ordered by theta, so the natural way
to eyeball it is one still per clip laid out in order. `save_grid` writes that
as `grid.png` beside the clips: COLS per row, each cell labelled with its
ladder index and theta, so a case can be checked without opening any video.

Shared by continuous families; call it from the family's `generate` with the
primary camera's first frame per clip.
"""

import numpy as np
from PIL import Image, ImageDraw

COLS = 8
PAD = 6
LABEL_H = 18
BG = (24, 24, 28)
FG = (215, 215, 220)


def save_grid(frames, labels, path, cols=COLS):
    """frames: list of HxWx3 uint8 arrays in ladder order. labels: one string
    per frame. Writes a (ceil(n/cols) x cols) contact sheet to `path`."""
    if not frames:
        raise ValueError("no frames to lay out")
    frames = [np.asarray(f) for f in frames]
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * PAD,
                              rows * (h + LABEL_H) + (rows + 1) * PAD), BG)
    draw = ImageDraw.Draw(sheet)
    for k, (frame, label) in enumerate(zip(frames, labels)):
        r, c = divmod(k, cols)
        x = PAD + c * (w + PAD)
        y = PAD + r * (h + LABEL_H + PAD)
        sheet.paste(Image.fromarray(frame), (x, y))
        draw.text((x + 2, y + h + 3), label, fill=FG)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return rows

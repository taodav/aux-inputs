import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List

from definitions import ROOT_DIR


def plot_arr(arr: np.ndarray):
    Image.fromarray(np.uint8(arr)).show()


def stringify_actions_q_vals(action_map: List, q_vals: np.ndarray) -> str:
    assert len(action_map) == q_vals.shape[0]
    str = ""
    for a, q in zip(action_map, q_vals):
        str += f"{a}: {q:.3f}\n"
    return str


def create_blank(viz_array: np.ndarray):
    """
    Create a blank slate below the image, of size viz_array.shape[0] // 2, viz_array.shape[1]
    :param viz_array:
    :return:
    """
    h, w, _ = viz_array.shape
    blank_image = Image.new('RGB', (w, h // 2), (255, 255, 255))
    return blank_image


def write_on_image(canvas: Image, text_cols: List[str], start_pos: Tuple[int, int] = (0, 0)):
    """
    Write multiple columns of text onto the given image.
    :param canvas: PIL image to write on
    :param text_cols: list of strings, one for each column
    :param start_pos: starting position
    :return:
    """
    w, h = canvas.size
    font_path = Path(ROOT_DIR, "scripts", "etc", "FreeMono.ttf")
    font = ImageFont.truetype(str(font_path), 18)
    # font = ImageFont.truetype("FreeMono.ttf", 18)

    next_position_to_add = start_pos
    d_actual = ImageDraw.Draw(canvas)
    for text in text_cols:
        img_to_guide = Image.new('RGB', (w, h), (255, 255, 255))

        d = ImageDraw.Draw(img_to_guide)
        d.text((0, 0), text, (0, 0, 0), font=font)

        text_w, text_h = d.textsize(text, font)
        offset_x, offset_y = font.getoffset(text)
        text_w += offset_x
        text_h += offset_y

        d_actual.text(next_position_to_add, text, fill=(0, 0, 0), font=font)
        next_position_to_add = (next_position_to_add[0] + text_w + 20, next_position_to_add[1])

    return canvas


def append_text(viz_array: np.ndarray, to_append: List[str]) -> np.ndarray:
    img_to_append = create_blank(viz_array)
    img_to_append = write_on_image(img_to_append, to_append, start_pos=(0, 0))
    arr_to_append = np.array(img_to_append, dtype=np.uint8)

    final_image = np.concatenate((viz_array, arr_to_append), axis=0)

    return final_image









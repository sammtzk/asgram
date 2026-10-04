# asgram/gui/pyside_components/pil_wrap.py
"""
Converts the PIL Image format to PySide6 compatible QPixmap.
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def blank_pil(img_text="[Image Not Yet Processed]"):
    """Text-only image placeholder for the PySide6 app."""
    default_w, default_h = 512, 256
    out = Image.new('RGB', (default_w, default_h))
    if '[Image Not Yet Processed]' == img_text:
        out_font = ImageFont.load_default(32)
    else:
        out_font = ImageFont.load_default(24)
    out_mod = ImageDraw.Draw(out)

    _, _, w, h = out_mod.textbbox((0, 0), img_text, out_font)
    out_w, out_h = (default_w - w) // 2, (default_h - h) // 2
    out_mod.text((out_w, out_h), img_text, 'white', out_font)

    return out


def pil2pix(_img: Image.Image, _label: QLabel):
    """For displaying PIL outputs from asgram in a PySide6 QLabel."""
    rgb_img = _img.copy().convert('RGB')
    max_w, max_h = _label.width(), _label.height()
    w, h, = rgb_img.size
    if max_w != w:
        rgb_img = rgb_img.resize((max_w, int(round(max_w / w * h))))
        w, h, = rgb_img.size
    if max_h < h:
        rgb_img = rgb_img.resize((int(round(max_h / h * w)), max_h))
        w, h, = rgb_img.size

    bytes_per_line = 3 * w
    return QPixmap.fromImage(QImage(
        rgb_img.tobytes('raw', 'RGB'),
        w, h, bytes_per_line, QImage.Format.Format_RGB888
    ))


def pil_info(_img: Image.Image, _np_arr: np.ndarray):
    """String overview of image and source array dimensions and data types."""
    _w, _h = _img.size
    return f"{_w} x {_h} ; {_img.mode} ; {_np_arr.dtype}"

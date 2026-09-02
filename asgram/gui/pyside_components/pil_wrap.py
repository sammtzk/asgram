# asgram/gui/pyside_components/pil_wrap.py
"""
Converts the PIL Image format to PySide6 compatible QPixmap.
"""

from PIL import Image
from PySide6.QtGui import QImage, QPixmap


DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 768


def pil_to_pixmap(_img: Image.Image, size='s'):
    """For displaying PIL outputs from asgram in a PySide6 window."""
    small = 's' == size
    max_w = DEFAULT_DISPLAY_WIDTH // 2 if small else DEFAULT_DISPLAY_WIDTH
    max_h = DEFAULT_DISPLAY_HEIGHT // 2 if small else DEFAULT_DISPLAY_HEIGHT

    rgb_img = _img.convert('RGB')
    w, h, = rgb_img.size
    if max_w < w:
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


def blank_pil():
    return Image.new('RGB', (DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT))

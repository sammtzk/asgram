# asgram/gui/pyside_components/eyecon.py
"""
The icon for the asgram PySide6 application.
"""

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def make_eyecon(icon=True):
    """
    Method for making a low-resolution icon. Designed by ASGRAM / SAMK.

    Intentionally written opaquely.
    """
    eyecon, yx = np.zeros((64, 64), dtype=np.float32), np.indices((64, 64))
    y, x = -4*(2*yx[0]/63-1), 4*(2*yx[1]/63-1)
    r, theta = np.hypot(x, y), np.mod(np.arctan2(y, x), 2 * np.pi)[:, ::-1]
    t, vl = np.pi >= theta, np.linspace(-50.0, 192.5, 48).tolist()
    pl, ul = np.mod(theta/2, 2*np.pi) > r, -np.square(x)/(2*np.pi)+np.pi/2 < y
    il = (np.mod(theta/2-np.pi, 2*np.pi) > r) & t & ~(pl & ~t)[::-1, ::-1]
    pl, ll = pl & t | (pl & ~t & ~(pl & t)[::-1, ::-1]), ul[::-1, :] & ~ul

    for _s in np.linspace(np.pi, 0, 16):
        eyecon[il & (_s > theta)] = vl.pop(0)
    for _s in np.linspace(2 * np.pi, 0, 32):
        eyecon[pl & (_s > theta)] = vl.pop(0)
    eyecon += eyecon[::-1, ::-1]
    eyecon[ul], eyecon[ll] = 250.0, 182.5
    for _s in np.linspace(1, 2 * np.pi / 3, 16):
        eyecon[ul & (-np.square(x)/(2*np.pi*_s)+np.pi/2*_s > y)] -= 4.21875

    ec = np.round(eyecon).astype(np.uint8).data
    ec = QImage(ec, 64, 64, 64, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(ec)

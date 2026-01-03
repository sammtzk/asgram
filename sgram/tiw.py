# sgram/tiw.py
"""
Constraint Satisfaction Algorithm for Stereogram Generation.

Provides tools to create Single Image Random Dot Stereograms (SIRDS).
Module named after Thimbleby, Inglis, & Witten (TIW).

References
----------
    Thimbleby, Harold & Inglis, Stuart & Witten, Ian. (1994).
    Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms.
    IEEE Computer. 27. 38-48. 10.1109/2.318576.
"""


from PIL import Image
import numpy as np


def _do_work(t, Z_array, x, y, mu, eye_scalar):
    """
    When combined with a while loop, searches nearby pixels in the depth map to
    determine whether Z_array[x, y] is visible or obscured.

    Functional replacement for the do while loop used in the original algorithm
    by Thimbleby, Inglis, & Witten (1994).
    """
    zt = Z_array[x, y] + 2 * (2 - mu * Z_array[x, y]) * t / (mu * eye_scalar)
    try:
        visible_left = Z_array[x - t, y] < zt
    except IndexError:
        visible_left = 1
    try:
        visible_right = Z_array[x + t, y] < zt
    except IndexError:
        visible_right = 1
    visible = visible_left and visible_right
    t += 1
    return t, zt, visible


def _dist(x, y, x0, y0):
    """Calculates Euclidean distance between two points (x, y) and (x0, y0)."""
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def _separation(Z, mu=1/3, dpi=72):
    """Calculates the stereogram pixel separation for a depth map value Z."""
    eye_scalar = round(2.5 * dpi)
    numer = (1 - mu * Z) * eye_scalar
    denom = 2 - mu * Z
    return round(numer / denom)


def SIRDS_from_Z_map(img, mu=1/3, dpi=72, draw_convergence_dots=True):
    """
    Creates a Single Image Random Dot Stereogram from a depth (Z) map.

    Original algorithm described by Thimbleby, Inglis, & Witten (1994) adapted,
    to Python.
    """
    if img.mode != 'L':
        img = img.convert('L')

    Z_array = np.array(img).T
    stereogram = Z_array.copy()
    maxX = Z_array.shape[0]
    maxY = Z_array.shape[1]
    Z_array = np.array(Z_array, dtype=np.float32)
    Z_array = (Z_array - np.min(Z_array)) / (np.max(Z_array) - np.min(Z_array))

    eye_scalar = round(2.5 * dpi)

    for y in range(maxY):
        same = [x for x in range(maxX)]

        for x in range(maxX):
            s = _separation(Z_array[x, y], mu, dpi)
            left = x - round(s / 2)
            right = left + s

            if (0 <= left) and (right < maxX):
                t, zt, visible = _do_work(1, Z_array, x, y, mu, eye_scalar)
                while visible and (zt < 1):
                    t, zt, visible = _do_work(t, Z_array, x, y, mu, eye_scalar)

                if visible:
                    look = same[left]
                    while look not in (left, right):
                        if look < right:
                            left = look
                            look = same[left]
                        else:
                            same[left] = right
                            left = right
                            look = same[left]
                            right = look
                    same[left] = right

        for x in range(maxX - 1, -1, -1):
            if same[x] == x:
                stereogram[x, y] = round(np.random.random()) * 255
            else:
                stereogram[x, y] = stereogram[same[x], y]

    if draw_convergence_dots:
        far = _separation(0, mu, dpi)
        dot_r = np.ceil(_dist(0, 0, maxX, maxY) / 100)
        for y in range(maxY):
            for x in range(maxX):
                if (
                    _dist(x, y, maxX / 2 - far / 2, maxY * 19 / 20) < dot_r
                    or
                    _dist(x, y, maxX / 2 + far / 2, maxY * 19 / 20) < dot_r
                ):
                    stereogram[x, y] = 0

    return Image.fromarray(stereogram.T)

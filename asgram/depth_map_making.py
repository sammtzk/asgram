# sgram/depth_map_making.py
"""
Functions for cleaning up, standardizing, and rescaling depth maps.
"""

from PIL import Image, ImageChops, ImageFilter
import numpy as np
import cv2 as cv
try:
    from asgram.tiw import _separation
except ModuleNotFoundError:
    from tiw import _separation


# PIL Methods Integration
def _resize_img(_img, mult=1.0):
    if 0.0 < mult and 1.0 != mult:
        _img = _img.resize([int(round(n * np.sqrt(mult))) for n in _img.size])
    return _img


# NumPy Methods Integration
def _normalize_img_array(_arr, normalize):
    _arr = np.array(_arr, dtype=np.float32)
    if not normalize:
        return _arr / 255
    return (_arr - np.min(_arr)) / (np.max(_arr) - np.min(_arr))


def _pad_img_array(_arr, mu=1/3, dpi=72):
    far = _separation(0, mu, dpi)
    l_pad = np.repeat(_arr[0:1, :], far, axis=0)
    r_pad = np.repeat(_arr[-1:, :], far, axis=0)
    return np.vstack((l_pad, _arr, r_pad))


# Image Smoothing
def _row_linearization_smoothing(_arr, thresh=0.64, min_win=9, min_lap=3):
    """
    Identifies approximately linear segments within an array and increases the
    granularity of the linear step.
    """

    mxb = _arr.copy()
    length = mxb.shape[0]
    index_arr = np.linspace(0, 1, length)
    mask_arr = np.full(length, np.nan)

    window = min_win  # int(max(np.ceil(length / 24), min_win))
    overlap = min_lap  # int(max(np.ceil(window / 6), min_lap))

    start = 0
    stop = window
    last_loop_flag = False

    while not last_loop_flag:
        if length == stop:
            last_loop_flag = True

        x_samp = index_arr[start:stop]
        y_samp = mxb[start:stop]

        coeffs, sse, _, _, _ = np.polyfit(x_samp, y_samp, deg=1, full=True)
        sst = np.sum(np.square((y_samp - np.mean(y_samp))))
        r_sq = 1 if sse.item() < 1e-5 else np.round(1 - sse / sst, 2).item()

        if thresh <= r_sq:
            slope = round(coeffs[0].item(), 2)
            intercept = round(coeffs[1].item(), 2)
            key = round(abs(intercept) * slope * 10)

            for idx in range(start, stop):
                old = mask_arr[idx]
                if not np.isnan(old):
                    mask_arr[idx] = max(old, key)
                else:
                    mask_arr[idx] = key

        start += window - overlap
        stop = start + window

        if length <= stop:
            stop = length
            start = stop - window

    for k in np.unique(mask_arr):
        if not np.isnan(k):
            lin_indicies = np.where(mask_arr == k)[0]
            first = lin_indicies[0]
            last = lin_indicies[-1]
            _len = lin_indicies.shape[0]
            if (2 < _len) and (16 > _len) and (_len == last - first + 1):
                fval = mxb[first]
                lval = mxb[last]
                mxb[first:last + 1] = np.linspace(fval, lval, _len)

    return mxb


def _linear_segment_smoothing(_arr):
    """
    Identifies approximately linear segments and increases the granularity of
    the linear step within each row of a depth array.
    """

    for y in range(_arr.shape[1]):
        _arr[:, y] = _row_linearization_smoothing(_arr[:, y], 0.7, 5, 4)
    return _arr


def integrated_image_smoothing(_img):
    """
    Smooths an image row-wise using linearization smoothing while preserving
    edges using edge detection and bilateral filtering.
    """

    hblur_arr = cv.GaussianBlur(np.array(_img.convert('L')).T, (1, 3), 0)
    hblur_img = Image.fromarray(hblur_arr.T)

    edge_detection = np.array(hblur_img.filter(ImageFilter.FIND_EDGES)).T
    smoothed_edges = cv.bilateralFilter(edge_detection, 9, 75, 75) > 100
    rshift = np.roll(smoothed_edges, shift=1, axis=0)
    lshift = np.roll(smoothed_edges, shift=-1, axis=0)
    edge_mask = smoothed_edges | rshift | lshift

    b_mask = hblur_arr == 0
    w_mask = hblur_arr == 255
    _mask = b_mask | w_mask | edge_mask

    static_mask = Image.fromarray(_mask.T)

    lss_img = Image.fromarray(_linear_segment_smoothing(hblur_arr).T)

    edges_only = ImageChops.multiply(hblur_img, static_mask)
    lss_only = ImageChops.multiply(lss_img, ImageChops.invert(static_mask))

    return ImageChops.add(edges_only, lss_only)


def _deprecated_smooth_img_array(_arr, mu=1/3, dpi=72, smoothing_amt=0):
    if smoothing_amt > 0:
        far = _separation(0, mu, dpi)
        near = _separation(1, mu, dpi)
        detail = np.ceil(255 / (far - near))

        ls_arr = np.roll(_arr, shift=-1, axis=0)
        rs_arr = np.roll(_arr, shift=1, axis=0)

        rmask = np.abs(_arr - rs_arr) < detail
        lmask = np.abs(_arr - ls_arr) < detail

        mask = rmask | lmask
        ls_amt = -int(round(smoothing_amt / 2))
        rs_amt = ls_amt + smoothing_amt
        for i in range(abs(ls_amt)):
            mask = mask & np.roll(mask, shift=-(i + 1), axis=0)
        for i in range(abs(rs_amt)):
            mask = mask & np.roll(mask, shift=(i + 1), axis=0)

        # pylint: disable=no-member
        hsmooth = cv.GaussianBlur(_arr, (1, smoothing_amt), 0)
        _arr[mask] = hsmooth[mask]

    return _arr


# Object Maker
class ZMap:
    """
    Stores and augments depth maps using PIL and NumPy methods.
    """

    def __init__(
            self, img, mu=1/3, dpi=72,
            scale=1.0, smooth_iis=False, smooth_bilat=False,
            invert=False, normalize=True, pad=False
    ):
        self.og_img = img
        self.mu = mu
        self.dpi = dpi

        self.scale = scale
        self.iis = smooth_iis
        self.bilat = smooth_bilat

        self.invert = invert
        self.normalize = normalize
        self.pad = pad

        self.zarr = np.array([])
        self.zmap = np.array([])
        self.update()

    def update(self):
        """Updates the depth map image according to class parameters."""
        zmap = _resize_img(self.og_img.copy().convert('L'), self.scale)
        zmap = integrated_image_smoothing(zmap) if self.iis else zmap
        zmap = ImageChops.invert(zmap) if self.invert else zmap

        zarr = np.array(zmap).T
        zarr = cv.bilateralFilter(zarr, 9, 75, 75) if self.bilat else zarr
        zarr = _normalize_img_array(zarr, self.normalize)
        zarr = _pad_img_array(zarr, self.mu, self.dpi) if self.pad else zarr

        self.zarr = zarr
        self.zmap = Image.fromarray(zarr.T * 255)

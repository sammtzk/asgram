# asgram/depth_map_making.py
"""
Functions for cleaning up, standardizing, and rescaling depth maps.
"""

import os
import numpy as np
import cv2 as cv
from PIL import Image
import openexr_numpy
try:
    from asgram.utils.utils import _pixel_separation
    from asgram.utils.parallelize import (
            worker_count, run_worker, parallelize_workers
    )
except ModuleNotFoundError:
    from utils.utils import _pixel_separation
    from utils.parallelize import (
            worker_count, run_worker, parallelize_workers
    )


# NumPy and OpenCV Methods Integration
def _normalize_img_array(_arr, normalize=True):
    _arr = np.array(_arr, dtype=np.float32)
    if not normalize:
        return _arr / 255
    max_val = np.max(_arr)
    if 0 < max_val:
        min_val = np.min(_arr)
        return (_arr - min_val) / (max_val - min_val)
    else:
        return _arr / 255


def _resize_img_array(_arr, mult=1.0):
    if 0.0 < mult and 1.0 != mult:
        dims = tuple(int(round(n * np.sqrt(mult))) for n in _arr.shape[::-1])
        interpolation = cv.INTER_AREA if 1.0 > mult else cv.INTER_LANCZOS4
        _arr = cv.resize(_arr, dims, interpolation=interpolation)
    return _arr


def _pad_img_array(_arr, mu=1/3, dpi=72):
    far = _pixel_separation(0, mu, dpi, cross_eyed=False)
    l_pad = np.repeat(_arr[0:1, :], far, axis=0)
    r_pad = np.repeat(_arr[-1:, :], far, axis=0)
    return np.vstack((l_pad, _arr, r_pad))


# Image Smoothing
def _row_linearization_smooth(_arr, thresh=0.64, window=9, overlap=3):
    """
    Identifies approximately linear segments within an array and increases the
    granularity of the linear step.
    """
    mxb = _arr.copy()
    length = mxb.shape[0]
    index_arr = np.linspace(0, 1, length)
    mask_arr = np.full(length, np.nan)

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


def _lss_worker(_args):
    """Worker for lss parallelization. Wraps generic run_worker."""
    ys_to_build, args_dict = _args

    def _row_func_wrapper(y, ad=args_dict):
        """Wrapper for _row_linearization_smooth."""
        return _row_linearization_smooth(_arr=ad['src_mat'][:, y])

    return run_worker(ys_to_build, args_dict, _row_func_wrapper, dim=2)


def linear_segment_smooth(_arr, num_jobs=-1):
    """
    Identifies approximately linear segments and increases the granularity of
    the linear step within each row of a depth array.
    """
    jobs = worker_count(num_jobs)
    if 1 < jobs:
        args_dict = {'src_mat': _arr, 'total_ys': _arr.shape[1]}
        _arr = parallelize_workers(args_dict, _lss_worker, jobs)
    else:
        for y in range(_arr.shape[1]):
            _arr[:, y] = _row_linearization_smooth(_arr[:, y])

    return _arr


def integrated_image_smooth(_arr, num_jobs=-1):
    """
    Smooths a [0, 1] normalized image row-wise using linearization smoothing
    while preserving edges using edge detection and bilateral filtering.
    """
    blur_arr = cv.GaussianBlur(_arr, (1, 3), 0)
    edge_detection = cv.Canny((blur_arr * 255.0).astype(np.uint8), 50, 150)
    smoothed_edges = cv.bilateralFilter(edge_detection, 9, 75, 75) > 100
    rshift = np.roll(smoothed_edges, shift=1, axis=0)
    lshift = np.roll(smoothed_edges, shift=-1, axis=0)

    edge_mask = smoothed_edges | rshift | lshift
    b_mask = blur_arr == 0.0
    w_mask = blur_arr == 1.0
    static_mask = b_mask | w_mask | edge_mask

    lss_arr = linear_segment_smooth(blur_arr, num_jobs)
    composite = np.where(static_mask, blur_arr, lss_arr)

    return composite


# Object Maker
class ZMap:
    """ Stores and augments depth maps using OpenCV and NumPy methods."""

    def __init__(
            self, source, mu=1/3, dpi=72,
            scale=1.0, iis=False, bil=False,
            invert=False, normalize=True, pad=False,
            num_jobs=-1
    ):
        self.src_arr = self._matrix_from_source(source)
        self.mu = mu
        self.dpi = dpi

        self.scale = scale
        self.iis = iis
        self.bil = bil

        self.invert = invert
        self.normalize = normalize
        self.pad = pad

        self.jobs = num_jobs

        self.size = (0, 0)
        self.zm_arr = np.array([])
        self.zm_img = Image.new('1', (0, 0))
        self.update()

    @staticmethod
    def _matrix_from_source(source):
        assert isinstance(source, (np.ndarray, Image.Image, str))
        match source:
            case str():
                assert os.path.isfile(source)
                if '.exr' == source[-4::]:
                    return openexr_numpy.imread(source, channel_names='V').T
                else:
                    # source format should be in Image.registered_extensions()
                    return np.array(Image.open(source).convert('L')).T
            case Image.Image():
                return np.array(source.copy().convert('L')).T
            case _:
                return source.copy()    # assumes array shape is (w, h)

    def update(self):
        """Updates the depth map image according to class parameters."""
        print("Step: Depth Map Making")
        zarr = self.src_arr.copy()

        zarr = _normalize_img_array(zarr, self.normalize)
        zarr = (1.0 - zarr) if self.invert else zarr
        zarr = _resize_img_array(zarr, self.scale)
        zarr = cv.bilateralFilter(zarr, 9, 75, 75) if self.bil else zarr
        zarr = integrated_image_smooth(zarr, self.jobs) if self.iis else zarr
        zarr = _pad_img_array(zarr, self.mu, self.dpi) if self.pad else zarr

        self.size = zarr.shape
        self.zm_arr = zarr
        self.zm_img = Image.fromarray(zarr.T * 255.0)
        print("Complete.")

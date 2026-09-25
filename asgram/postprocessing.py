# asgram/postprocessing.py
"""
Functions for cleaning up autostereograms and adding convergence helpers.
"""

import copy
import numpy as np
import cv2 as cv
from PIL import Image
try:
    from asgram.utils.utils import _pixel_separation as _pix_sep
    from asgram.utils.params import Params
    from asgram.source_pattern_making import SrcPat
    from asgram.pixel_constraint_calculating import PixCon
    from asgram.utils.parallelize import (
            worker_count, run_worker, parallelize_workers
    )
except ModuleNotFoundError:
    from utils.utils import _pixel_separation as _pix_sep
    from utils.params import Params
    from source_pattern_making import SrcPat
    from pixel_constraint_calculating import PixCon
    from utils.parallelize import (
            worker_count, run_worker, parallelize_workers
    )


class Synthesizer:
    """
    Creates an asgram by applying pixel constraints to source pattern.

    If there is a size mismatch between the input arrays, defaults to resizing
    the source pattern to fit the pixel constraints matrix. Alternatively,
    'con_sharp', 'con_smooth', and 'con_balanced' are inputs which handle
    constraint matrix resizing and normalization.

    Returns a tuple of autostereogram matrix (asg_mat) and the resized input
    arrays of sp_arr and con_mat as spat and pcon respectively.
    """

    def __init__(
            self, sp_arr: np.ndarray, con_mat: np.ndarray,
            resize_technique='pattern'
    ):
        self.sp_arr = sp_arr
        self.con_mat = con_mat
        self.resize_technique = resize_technique

        self.asg_mat = np.array([])
        self.sp_arr_resized = np.array([])
        self.con_mat_resized = np.array([])
        self.update()

    @staticmethod
    def synthesize(
        sp_arr: np.ndarray, con_mat: np.ndarray, resize_technique='pattern'
    ):
        """
        Returns a tuple of autostereogram matrix (asg_mat) and the resized
        input arrays of sp_arr and con_mat as spat and pcon respectively.
        """
        spat = sp_arr.copy()
        pcon = con_mat.copy()
        sp_dims = tuple(spat.shape[-2:][::-1])
        pc_dims = tuple(pcon.shape[::-1])

        if sp_dims != pc_dims:
            if 'pattern' == resize_technique:
                _ip = cv.INTER_AREA if sp_dims < pc_dims else cv.INTER_LANCZOS4
                _r = cv.resize(spat[0], pc_dims, interpolation=_ip)[None, :, :]
                _g = cv.resize(spat[1], pc_dims, interpolation=_ip)[None, :, :]
                _b = cv.resize(spat[2], pc_dims, interpolation=_ip)[None, :, :]
                spat = np.vstack([_r, _g, _b])
            else:
                temp = pcon.astype(np.float32)
                if 'con_balanced' == resize_technique:
                    # resize width-wise first
                    temp = cv.resize(
                        src=temp,
                        dsize=(pc_dims[0], sp_dims[1]),
                        interpolation=cv.INTER_LINEAR_EXACT
                    )
                # con_sharp is default interpolation, like in pattern case
                _ip = cv.INTER_AREA if pc_dims < sp_dims else cv.INTER_LANCZOS4
                if 'con_smooth' == resize_technique:
                    _ip = cv.INTER_LINEAR_EXACT
                temp = cv.resize(temp, sp_dims, interpolation=_ip)
                temp *= float(sp_dims[1]) / float(pc_dims[1])  # scale pointers
                pcon = np.round(temp).astype(np.uint16)

        asg_mat = np.take_along_axis(spat, pcon[None, :, :], axis=1)
        return (asg_mat, spat, pcon)

    def update(self):
        """Updates output arrays according to class attributes."""
        asc = self.synthesize(self.sp_arr, self.con_mat, self.resize_technique)
        self.asg_mat, self.sp_arr_resized, self.con_mat_resized = asc


def _redmean_color_diff(color1, color2):
    r1, g1, b1 = color1.astype(float)
    r2, g2, b2 = color2.astype(float)

    rmean = 0.5 * (r1 + r2)
    drs = (r1 - r2) ** 2
    dgs = (g1 - g2) ** 2
    dbs = (b1 - b2) ** 2

    rw = 2 + rmean / 256
    gw = 4
    bw = 2 + (255 - rmean) / 256

    return np.sqrt(rw * drs + gw * dgs + bw * dbs)


def _row_pdvrp(asg_row, thresh=25):
    """Pixel Disparity Visual Rectification Postprocessing"""
    for inner_idx in np.arange(0 + 1, asg_row.shape[1] - 1):
        target_pixel = asg_row[:, inner_idx].astype(float)
        left_pixel = asg_row[:, inner_idx + 1].astype(float)
        right_pixel = asg_row[:, inner_idx - 1].astype(float)

        left_rmcd = _redmean_color_diff(target_pixel, left_pixel)
        right_rmcd = _redmean_color_diff(target_pixel, right_pixel)

        if thresh < left_rmcd and thresh < right_rmcd:
            rbar = int(round((left_pixel[0] + right_pixel[0]) / 2))
            gbar = int(round((left_pixel[1] + right_pixel[1]) / 2))
            bbar = int(round((left_pixel[2] + right_pixel[2]) / 2))
            asg_row[:, inner_idx] = [rbar, gbar, bbar]

    return asg_row


def _pdvrp_worker(_args):
    """Worker for pdvrp parallelization. Wraps generic run_worker."""
    ys_to_build, args_dict = _args

    def _row_func_wrapper(y, ad=args_dict):
        """Wrapper for _row_pdvrpp."""
        return _row_pdvrp(asg_row=ad['src_mat'][:, :, y])

    return run_worker(ys_to_build, args_dict, _row_func_wrapper, dim=3)


def pdvrp(asg, thresh=25, num_jobs=-1):
    """Pixel Disparity Visual Rectification Postprocessing."""
    jobs = worker_count(num_jobs)
    if 1 < jobs:
        args_dict = {'src_mat': asg, 'total_ys': asg.shape[2]}
        asg = parallelize_workers(args_dict, _pdvrp_worker, jobs)
    else:
        for y in range(asg.shape[2]):
            asg[:, :, y] = _row_pdvrp(asg[:, :, y], thresh)

    return asg


def dots(asg, depth, height='bottom', mu=1/3, dpi=72, cross_eyed=False):
    """
    By default draws dots at the far plane. A depth value of 1 will draw at the
    near plane. Values outside of [0, 1] will not draw.
    """
    if 0 <= depth <= 1:
        sampling_arr = copy.deepcopy(asg)
        w, h = sampling_arr.shape[1:]
        all_pixels = sampling_arr.reshape(3, -1).T
        np.random.shuffle(all_pixels)
        s_pixels = all_pixels[:100].T
        colors, counts = np.unique(s_pixels, axis=1, return_counts=True)
        dot_color = colors[:, np.argmin(counts)]

        how_far = _pix_sep(depth, mu, dpi, cross_eyed)
        dot_r2 = (np.hypot(w, h) / 100) ** 2

        x_center_1 = w / 2 - how_far / 2
        x_center_2 = w / 2 + how_far / 2
        match height:
            case 'top':
                y_c_pos = 1
            case 'center':
                y_c_pos = 10
            case _:
                y_c_pos = 19
        y_center = h * y_c_pos / 20

        xs = np.arange(w)[:, None]
        ys = np.arange(h)[None, :]

        dot_1 = (xs - x_center_1) ** 2 + (ys - y_center) ** 2
        dot_2 = (xs - x_center_2) ** 2 + (ys - y_center) ** 2
        mask = (dot_1 < dot_r2) | (dot_2 < dot_r2)

        asg[:, mask] = dot_color[:, None]

    return asg


class Post():
    """Finalizes autostereograms with postprocessing techinques."""

    def __init__(self, sp: SrcPat, pc: PixCon, p: Params):
        self.sp = sp
        self.pc = pc
        self.p = p

        self.final_arr = np.array([])
        self.final_img = Image.new('1', (0, 0))
        self.update()

    @property
    def asg_mat(self):
        synth = Synthesizer(self.sp.sp_arr, self.pc.con_mat)
        return synth.asg_mat

    def update(self):
        """Updates the final autostereogram according to class parameters."""
        print("Step: Postprocessing")
        fin = self.asg_mat
        if self.p.pixel_disparity_smoothing:
            fin = pdvrp(fin, num_jobs=self.p.num_jobs)
        fin = dots(
            fin,
            self.p.convergence_dot_depth, self.p.convergence_dot_placement,
            self.p.mu, self.p.dpi, self.p.cross
        )
        self.final_arr = fin
        self.final_img = Image.fromarray(fin.T).convert('RGB')
        print("Complete.")

    def save(self, file_name='temp', dir_path='', extension='.png'):
        """
        Saves final asgram as an image of a specified format.

        Extension should be one of '.png', '.jpg', or '.jpeg'.
        """
        dir_path = dir_path + '/' if dir_path else dir_path
        file_path = dir_path + file_name + extension

        if extension in ('.png', '.jpg', '.jpeg'):
            with open(file_path, 'wb') as f:
                self.final_img.save(f)

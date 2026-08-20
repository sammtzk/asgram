# asgram/postprocessing.py
"""
Functions for cleaning up autostereograms and adding convergence helpers.
"""

import copy
import numpy as np
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


def _row_pdvrpp(asg_row, thresh=25):
    """Pixel Disparity Visual Rectification Post-Processing"""
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


def _pdvrpp_worker(_args):
    """Worker for pdvrpp parallelization. Wraps generic run_worker."""
    ys_to_build, args_dict = _args

    def _row_func_wrapper(y, ad=args_dict):
        """Wrapper for _row_pdvrpp."""
        return _row_pdvrpp(asg_row=ad['src_mat'][:, :, y])

    return run_worker(ys_to_build, args_dict, _row_func_wrapper, dim=3)


def pdvrpp(asg, thresh=25, num_jobs=-1):
    """Pixel Disparity Visual Rectification Post-Processing."""
    jobs = worker_count(num_jobs)
    if 1 < jobs:
        args_dict = {'src_mat': asg, 'total_ys': asg.shape[2]}
        asg = parallelize_workers(args_dict, _pdvrpp_worker, jobs)
    else:
        for y in range(asg.shape[2]):
            asg[:, :, y] = _row_pdvrpp(asg[:, :, y], thresh)

    return asg


def conv_dots(asg, depth, height='bottom', mu=1/3, dpi=72, cross_eyed=False):
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

        how_far = _pixel_separation(depth, mu, dpi, cross_eyed)
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


def finish(
        asg,
        depth, height='bottom', mu=1/3, dpi=72, cross=False,
        pdvrs=False, num_jobs=-1
):
    """Combines module postprocessing helpers into one function."""
    print("Step: Postprocessing")
    asg = pdvrpp(asg, num_jobs=num_jobs) if pdvrs else asg
    asg = conv_dots(asg, depth, height, mu, dpi, cross)
    print("Complete.")
    return asg

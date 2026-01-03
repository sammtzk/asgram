# sgram/tiw.py
"""
Makes single image stereograms.
"""

from PIL import Image
import numpy as np
try:
    from sgram.utils import (
        _prepare_z_arr, _pixel_separation, _sirds_init, _asg_row, _conv_dots
    )
    from sgram.parallelize.local_process import (
        UPDATE_COUNTER, COUNTER, LOCK, STOP_EARLY,
        determine_processes, pool_runs
    )
    from sgram.parallelize.gradio_process import set_processes, executor_runs
except ModuleNotFoundError:
    from utils import (
        _prepare_z_arr, _pixel_separation, _sirds_init, _asg_row, _conv_dots
    )
    from parallelize.local_process import (
        UPDATE_COUNTER, COUNTER, LOCK, STOP_EARLY,
        determine_processes, pool_runs
    )
    from parallelize.gradio_process import set_processes, executor_runs


# Stereogram
def _p_worker(args):
    pull_from, zar, mu, dpi, cross, approach, ys_to_build = args
    asg = np.zeros_like(pull_from)
    total = len(ys_to_build)
    prog = int(np.ceil(total / UPDATE_COUNTER))

    for i, y in enumerate(ys_to_build):
        asg[:, :, y] = _asg_row(pull_from, y, zar, mu, dpi, cross, approach)

        if STOP_EARLY.is_set():
            break
        elif (i + 1) % prog == 0 or i + 1 == total:
            with LOCK:
                COUNTER.value += prog if (i + 1) % prog == 0 else total % prog

    return asg


def _g_worker(args):
    pull_from, zar, mu, dpi, cross, approach, ys_to_build = args
    asg = np.zeros_like(pull_from)

    for y in ys_to_build:
        asg[:, :, y] = _asg_row(pull_from, y, zar, mu, dpi, cross, approach)

    return asg


def sirds(
    img, ref_img=None, ref_fit='fit',
    mu=1/3, dpi=72, cross=False, approach='rl',
    normalize=True, invert=False, pad=False, scale=1.0, palette='bw',
    dot_depth=0.0, dot_height='bottom',
    random_seed=1132,
    num_jobs=8
):
    """
    Creates a Single Image Random Dot Stereogram from a depth (Z) map.

    Original algorithm described by Thimbleby, Inglis, & Witten (1994) adapted,
    to Python.

    by default draws dots at the far plane. A depth value of 1
    will draw at the near plane. Values outside of [0, 1] will not draw.
    """
    np.random.seed(random_seed)
    zar = _prepare_z_arr(img, mu, dpi, normalize, invert, pad, scale)
    asg = _sirds_init(zar, ref_img, ref_fit, dpi, approach, palette)

    jobs = determine_processes(num_jobs)
    if 1 < jobs:
        chunks = np.array_split(list(range(zar.shape[1])), jobs)
        _args = [(asg, zar, mu, dpi, cross, approach, c) for c in chunks]
        results = pool_runs(_p_worker, _args, zar.shape[1], jobs)
        asg = np.zeros_like(asg)
        for r in results:
            asg += r
    else:
        for y in range(zar.shape[1]):
            asg[:, :, y] = _asg_row(asg, y, zar, mu, dpi, cross, approach)

    asg = _conv_dots(asg, dot_depth, dot_height, mu, dpi, cross)

    return Image.fromarray(asg.T)


async def sirds_async(
    img, ref_img=None, ref_fit='fit',
    mu=1/3, dpi=72, cross=False, approach='rl',
    normalize=True, invert=False, pad=False, scale=1.0, palette='bw',
    dot_depth=0.0, dot_height='bottom',
    random_seed=1132,
    num_jobs=5, concurrency_limit=4
):
    """
    Creates a Single Image Random Dot Stereogram from a depth (Z) map.

    Original algorithm described by Thimbleby, Inglis, & Witten (1994) adapted,
    to Python.

    draw_dots by default draws dots at the far plane. A value of 1
    will draw at the near plane. Values outside of [0, 1] will not draw.
    """
    np.random.seed(random_seed)
    zar = _prepare_z_arr(img, mu, dpi, normalize, invert, pad, scale)
    asg = _sirds_init(zar, ref_img, ref_fit, dpi, approach, palette)

    jobs = set_processes(num_jobs, concurrency_limit)
    if 1 < jobs:
        chunks = np.array_split(list(range(zar.shape[1])), jobs)
        _args = [(asg, zar, mu, dpi, cross, approach, c) for c in chunks]
        results = await executor_runs(_g_worker, _args)
        asg = np.zeros_like(asg)
        for r in results:
            asg += r
    else:
        for y in range(zar.shape[1]):
            asg[:, :, y] = _asg_row(asg, y, zar, mu, dpi, cross, approach)

    asg = _conv_dots(asg, dot_depth, dot_height, mu, dpi, cross)

    return Image.fromarray(asg.T)


# Animation
def _shift_image(img, dx):
    """Shifts an RGBA image to the right dx pixels."""
    shifted = Image.new('RGBA', img.size, (0, 0, 0, 0))
    shifted.paste(img, (dx, 0))
    return shifted


def _make_overlay(left, right):
    """Overlays two RGBA images."""
    overlay = Image.new('RGBA', left.size, (0, 0, 0, 0))
    overlay.alpha_composite(left)
    overlay.alpha_composite(right)
    return overlay


def sirds_convergence(
        img_path, gif_path,
        mu=1/3, dpi=72,
        ms_per_frame=50, num_gif_loops=3
):
    """
    Overlays saved SIRDS with transparency to reveal depth information in an
    animation. Animation saved as GIF to the path provided.
    """
    far = _pixel_separation(0, mu, dpi, cross_eyed=False)
    near = dpi

    # Treat Stereogram White Dots as Transparent
    s0 = Image.open(img_path).convert('1').convert('L')
    s0_array = np.array(s0)
    alpha = np.where(s0_array == 255, 0, 255).astype('uint8')
    s1_array = np.dstack(tup=[s0_array, s0_array, s0_array, alpha])
    s1 = Image.fromarray(s1_array)

    # Create Overlay Frames
    frames = []
    left = s1
    for dx in range(0, far + 1):
        shifted = _shift_image(s1, dx)
        frame = _make_overlay(left, shifted)

        bw_frame = Image.new('RGB', frame.size, (255, 255, 255))
        bw_frame.paste(frame, mask=frame.split()[3])
        bw_frame.convert('L')

        frames.append(bw_frame)

    # Make and Save Animation
    animation = []
    animation += frames
    animation += [frames[-1]] * 6
    animation += frames[near:][::-1]
    animation += [frames[near]] * 6
    animation += frames[near:]
    animation += [frames[-1]] * 6
    animation += frames[::-1]
    animation[0].save(
        gif_path,
        save_all=True,
        append_images=animation[1:],
        optimize=False,
        duration=ms_per_frame,
        loop=num_gif_loops,
    )

    # Check Whether GIF Was Saved Properly
    try:
        with Image.open(gif_path) as img:
            img.verify()
        print(f"Convergence GIF successfully created and saved to: {gif_path}")
    except FileNotFoundError:
        print("GIF creation failed or could not be opened: FileNotFoundError")

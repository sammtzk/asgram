# asgram/gui.py
"""
gooey. want to specify png output with a path.
"""

import os
import numpy as np
from matplotlib import colormaps
import gradio as gr
from gradio import themes

try:
    from asgram.art import sirds_async
    from asgram.parallelize.gradio_process import set_processes, executor_init
except ModuleNotFoundError:
    from art import sirds_async
    from parallelize.gradio_process import set_processes, executor_init


CONCURRENT_USERS = 2
EXECUTOR = None


async def _make_autostereogram(
    sirds_input_image, sirds_ref_image,
    sirds_ref_image_w_reps, sirds_ref_image_h_reps,
    sirds_depth_of_field, sirds_dots_per_inch,
    sirds_viewing_method, sirds_building_method,
    sirds_normalize_depth_map, sirds_invert_depth_map, sirds_smooth_depth_map,
    sirds_pad_depth_map, sirds_scale_image,
    sirds_bw_or_color, sirds_color_set,
    sirds_draw_conv_dots, sirds_conv_dot_depth,
    sirds_random_seed, sirds_number_of_jobs
):
    if sirds_input_image is None:
        return None

    sirds_pattern = f'tile={sirds_ref_image_w_reps}x{sirds_ref_image_h_reps}'

    sirds_cross_eyed = sirds_viewing_method == 'Cross'

    match sirds_building_method:
        case 'Middle-Out':
            sirds_approach = 'mo'
        case 'Out-In':
            sirds_approach = 'oi'
        case 'Left-to-Right':
            sirds_approach = 'lr'
        case 'Full Random':
            sirds_approach = 'random'
        case _:
            sirds_approach = 'rl'

    if sirds_bw_or_color:
        sirds_color_set = 'bw'

    match sirds_draw_conv_dots:
        case 'Top':
            sirds_conv_dot_height = 'top'
        case 'Center':
            sirds_conv_dot_height = 'center'
        case 'Bottom':
            sirds_conv_dot_height = 'bottom'
        case _:
            sirds_conv_dot_height = 'bottom'
            sirds_conv_dot_depth = -1.0

    return await sirds_async(
        img=sirds_input_image,
        ref_img=sirds_ref_image,
        ref_fit=sirds_pattern,
        mu=sirds_depth_of_field,
        dpi=sirds_dots_per_inch,
        cross=sirds_cross_eyed,
        approach=sirds_approach,
        normalize=sirds_normalize_depth_map,
        invert=sirds_invert_depth_map,
        smooth=sirds_smooth_depth_map,
        pad=sirds_pad_depth_map,
        scale=sirds_scale_image,
        palette=sirds_color_set,
        dot_depth=sirds_conv_dot_depth,
        dot_height=sirds_conv_dot_height,
        random_seed=sirds_random_seed,
        num_jobs=sirds_number_of_jobs,
        concurrency_limit=CONCURRENT_USERS
    )


def _recommend_dpi(rdpi_input_image, rdpi_upscale_image):
    if rdpi_input_image is None:
        return gr.update(value=72)

    image_width = rdpi_input_image.size[0]
    width_scalar = np.sqrt(rdpi_upscale_image)
    rdpi = int(round(image_width * width_scalar / 0.9 / 7))
    return gr.update(value=max(32, min(512, rdpi)))


def _set_color_set(sri_bw_or_color, sri_rand_ref):
    if sri_rand_ref == 'Random' and not sri_bw_or_color:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def _set_ref_image(sri_rand_ref):
    if sri_rand_ref == 'From Reference':
        return gr.update(visible=True)
    else:
        return gr.update(value=None, visible=False)


with gr.Blocks(title="samuelogram") as demo:
    gr.Markdown("# **Custom Autostereogram Maker**")
    gr.Markdown("## *Demo by Samuel A. M. K.*")

    input_image = gr.Image(label="Upload a Depth Map", type='pil')
    gr.Markdown(
        "#### Note: "
        "Depth maps are interpreted with black as far and white as near."
    )

    with gr.Accordion(
        "Depth Map Options", open=False, visible=False
    ) as dmoptions:
        normalize_depth_map = gr.Checkbox(True, label="Normalize")
        invert_depth_map = gr.Checkbox(False, label="Invert")
        smooth_depth_map = gr.Checkbox(False, label="Smooth")
        pad_depth_map = gr.Checkbox(False, label="Pad Horizontal")
        scale_image = gr.Slider(
            0.25, 4.0, value=1.0, step=0.25, label="Resize"
        )
        dots_per_inch = gr.Slider(
            32, 512, value=72, step=1, label="Dots Per Inch"
        )

    with gr.Accordion(
        "Autostereogram Options", open=False, visible=False
    ) as asoptions:
        viewing_method = gr.Dropdown(
            ["Parallel", "Cross"], label="Viewing Method", filterable=False
        )
        building_method = gr.Dropdown(
            [
                "Middle-Out", "Out-In",
                "Left-to-Right", "Right-to-Left",
                "Full Random"
            ],
            label="Building Method", filterable=False
        )
        depth_of_field = gr.Slider(
            0.1, 0.9, value=0.3, step=0.1, label="Depth of Field"
        )
        draw_conv_dots = gr.Dropdown(
            ["None", "Top", "Center", "Bottom"], label="Convergence Dots",
            filterable=False
        )
        conv_dot_depth = gr.Slider(
            0.0, 1.0, value=0.0, step=0.1, label="Dot Depth (far <--> near)",
            visible=False
        )

        rand_ref = gr.Dropdown(
            ["Random", "From Reference"], label="Pattern", filterable=False
        )
        bw_or_color = gr.Checkbox(True, label="Black and White")
        color_set = gr.Dropdown(
            list(colormaps),
            label="Matplotlib Colormaps",
            filterable=False,
            visible=False
        )
        set_random_seed = gr.Slider(
            0, 2 ** 16 - 1, value=1132, step=1, label="Random Seed"
        )
        ref_image = gr.Image(
            label="Upload a Reference Image", type='pil', visible=False
        )
        ref_image_w_reps = gr.Slider(
            1, 24, value=1, step=1,
            label="Horizontal Repetitions of Reference", visible=False
        )
        ref_image_h_reps = gr.Slider(
            1, 24, value=1, step=1,
            label="Vertical Repetitions of Reference", visible=False
        )

    with gr.Accordion(
        "Advanced Options", open=False, visible=False
    ) as avoptions:
        number_of_jobs = gr.Slider(
            1, set_processes(20, CONCURRENT_USERS), value=1, step=1,
            label="Workers for Parallelization"
        )
        gr.Markdown(
            "#### Note: For low-resolution autostereograms, the overhead from "
            "parallelization will be considerably higher than the standard "
            "runtime. Use additional workers only for processing large images."
        )

    make_button = gr.Button("Make Autostereogram")
    output_image = gr.Image(label="Result", visible=False)

    # pylint: disable=no-member
    input_image.change(
        fn=lambda x: gr.update(visible=x is not None),
        inputs=input_image, outputs=dmoptions
    )

    input_image.change(
        fn=lambda x: gr.update(visible=x is not None),
        inputs=input_image, outputs=asoptions
    )

    input_image.change(
        fn=lambda x: gr.update(visible=x is not None),
        inputs=input_image, outputs=avoptions
    )

    input_image.change(
        fn=_recommend_dpi,
        inputs=[input_image, scale_image], outputs=dots_per_inch
    )

    scale_image.change(
        fn=_recommend_dpi,
        inputs=[input_image, scale_image], outputs=dots_per_inch
    )

    draw_conv_dots.change(
        fn=lambda x: gr.update(visible=x != 'None'),
        inputs=draw_conv_dots, outputs=conv_dot_depth
    )

    rand_ref.change(
        fn=lambda x: gr.update(visible=x == 'Random'),
        inputs=rand_ref, outputs=bw_or_color
    )

    rand_ref.change(
        fn=_set_color_set,
        inputs=[bw_or_color, rand_ref], outputs=color_set
    )

    rand_ref.change(
        fn=lambda x: gr.update(visible=x == 'Random'),
        inputs=rand_ref, outputs=set_random_seed
    )

    rand_ref.change(
        fn=_set_ref_image,
        inputs=rand_ref, outputs=ref_image
    )

    rand_ref.change(
        fn=lambda x: gr.update(visible=x == 'From Reference'),
        inputs=rand_ref, outputs=ref_image_w_reps
    )

    rand_ref.change(
        fn=lambda x: gr.update(visible=x == 'From Reference'),
        inputs=rand_ref, outputs=ref_image_h_reps
    )

    bw_or_color.change(
        fn=_set_color_set,
        inputs=[bw_or_color, rand_ref], outputs=color_set
    )

    make_button.click(
        fn=lambda _: gr.update(visible=True),
        inputs=output_image, outputs=output_image
    ).then(
        fn=_make_autostereogram,
        inputs=[
            input_image, ref_image,
            ref_image_w_reps, ref_image_h_reps,
            depth_of_field, dots_per_inch,
            viewing_method, building_method,
            normalize_depth_map, invert_depth_map, smooth_depth_map,
            pad_depth_map, scale_image,
            bw_or_color, color_set,
            draw_conv_dots, conv_dot_depth,
            set_random_seed, number_of_jobs
        ],
        outputs=output_image
    )
    # pylint: enable=no-member


if __name__ == '__main__':
    executor_init(os.cpu_count())
    demo.queue(default_concurrency_limit=CONCURRENT_USERS)
    demo.launch(share=False, theme=themes.Glass(), inbrowser=False)

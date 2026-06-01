# asgram/wgui.py
"""
Functions for making polished single image stereograms.
"""

from io import BytesIO
from ipywidgets import (
    FloatSlider, IntSlider, Dropdown, Checkbox,
    HBox, VBox, HTML, FileUpload, Button, Output
)
from IPython.display import display
from IPython.display import Image as IPyImage
from PIL import Image
from matplotlib import colormaps
try:
    from asgram.depth_map_making import ZMap
    from asgram.source_pattern_making import SrcPat
    from asgram.art import synthesizer
    from asgram.postprocessing import finish
except ModuleNotFoundError:
    from depth_map_making import ZMap
    from source_pattern_making import SrcPat
    from art import synthesizer
    from postprocessing import finish


def asgram_widgets():
    """
    Creates an autostereogram from a depth (Z) map.
    For use in notebooks with ipywidgets.

    Original Single Image Random Dot Stereogram algorithm described by
    Thimbleby, Inglis, & Witten (1994), adapted to Python.
    """
    _zmap = None
    _srcpat = None
    _asg = None
    _final = None

    # General Widgets
    mu = FloatSlider(
        value=0.35, min=0, max=1, step=0.05,
        description='Depth of Field', disabled=False
    )
    dpi = IntSlider(
        value=72, min=1, max=1000, step=1,
        description='Dots per Inch', disabled=False
    )
    cross = Dropdown(
        options=[('Parallel', False), ('Cross', True)],
        value=False,
        description='View Type', disabled=False
    )
    approach = Dropdown(
        options=[
            ('Right to Left', 'rl'),
            ('Left to Right', 'lr'),
            ('Middle Outward', 'mo'),
            ('Outer Inward', 'oi'),
            ('True Random', 'random')
        ],
        value='rl',
        description='Build Direction', disabled=False
    )
    random_seed = IntSlider(
        value=1132, min=0, max=10000, step=1,
        description='Random Seed', disabled=False
    )
    num_jobs = IntSlider(
        value=8, min=1, max=20, step=1,
        description='Parallelized Workers', disabled=False
    )

    gw = VBox([
        HTML(value='<b>General Parameters<b>'),
        HBox([
            VBox([mu, dpi, random_seed]),
            VBox([cross, approach, num_jobs])
        ])
    ])

    # Depth Map Widgets
    img_upload = FileUpload(accept='image/*', multiple=False)

    normalize = Checkbox(
        value=True,
        description='Normalize', disabled=False
    )
    invert = Checkbox(
        value=False,
        description='Invert', disabled=False
    )
    iis = Checkbox(
        value=False,
        description='Integrated Image Smooth', disabled=False
    )
    bil = Checkbox(
        value=False,
        description='Bilateral Smooth', disabled=False
    )
    pad = Checkbox(
        value=False,
        description='Pad', disabled=False
    )
    scale = FloatSlider(
        value=1.0, min=0.25, max=20, step=0.25,
        description='Scale', disabled=False
    )

    zm_make = Button(
        description='Make Depth Map', disabled=False
    )
    zm_output = Output()

    def _zm_make_clicked(b):
        _ = b
        if img_upload.value != ():
            img = Image.open(BytesIO(img_upload.value[0].content))
            nonlocal _zmap
            _zmap = ZMap(
                img,
                mu.value,
                dpi.value,
                scale.value,
                iis.value,
                bil.value,
                invert.value,
                normalize.value,
                pad.value,
                num_jobs.value
            )
            _buffer = BytesIO()
            _zmap.zm_img.convert('RGB').save(_buffer, format='PNG')
            with zm_output:
                display(IPyImage(_buffer.getvalue()))
            if _srcpat is not None:
                _sp_make_clicked(1)
        else:
            with zm_output:
                print('Please upload a Depth Map.')
        zm_output.clear_output(wait=True)
    zm_make.on_click(_zm_make_clicked)

    dw = VBox([
        HTML('<b>Depth Map Options<b>'),
        HBox([HTML('Upload Depth Map:'), img_upload]),
        HBox([
            VBox([normalize, invert, pad]),
            VBox([iis, bil, scale])
        ]),
        zm_make,
        zm_output
    ])

    # Source Pattern Widgets
    ref_upload = FileUpload(accept='image/*', multiple=False)

    rfit = Dropdown(
        options=[
            ('Fit', 'fit'),
            ('Auto Tile', 'tile'),
            ('Horizontal Tile', 'htile'),
            ('Vertical Tile', 'vtile'),
            ('Enforce Approach-Specific Source', 'ES')
        ],
        value='fit',
        description='Pattern Adjustments', disabled=False
    )
    palettes = ['bw']
    palettes.extend(list(colormaps))
    rpal = Dropdown(
        options=palettes,
        value='bw',
        description='SIRDS Palette', disabled=False
    )

    sp_make = Button(
        description='Make Source Pattern', disabled=False
    )
    sp_output = Output()

    def _sp_make_clicked(b):
        _ = b
        if _zmap is not None:
            if ref_upload.value != ():
                ref = Image.open(BytesIO(ref_upload.value[0].content))
            else:
                ref = None
            nonlocal _srcpat
            _srcpat = SrcPat(
                _zmap.size,
                ref,
                cross.value,
                mu.value,
                dpi.value,
                rfit.value,
                approach.value,
                rpal.value
            )
            _buffer = BytesIO()
            _srcpat.sp_img.convert('RGB').save(_buffer, format='PNG')
            with sp_output:
                display(IPyImage(_buffer.getvalue()))
        else:
            with sp_output:
                print('Please process Depth Map first for sizing.')
        sp_output.clear_output(wait=True)
    sp_make.on_click(_sp_make_clicked)

    sw = VBox([
        HTML('<b>Source Pattern Options<b>'),
        HBox([HTML('Upload Source Pattern <i>(Optional)<i>:'), ref_upload]),
        HBox([rfit, rpal]),
        sp_make,
        sp_output
    ])

    # Autostereogram Widgets
    asg_make = Button(
        description='Make Autostereogram', disabled=False
    )
    asg_output = Output()

    def _asg_make_clicked(b):
        _ = b
        if (_zmap is not None) and (_srcpat is not None):
            nonlocal _asg
            _asg = synthesizer(
                _zmap,
                _srcpat,
                mu.value,
                dpi.value,
                cross.value,
                approach.value,
                num_jobs.value
            )
            _buffer = BytesIO()
            Image.fromarray(_asg.T).convert('RGB').save(_buffer, format='PNG')
            with asg_output:
                display(IPyImage(_buffer.getvalue()))
        else:
            with asg_output:
                if _zmap is None:
                    print('Please make Depth Map.')
                if _srcpat is None:
                    print('Please make Source Pattern.')
        asg_output.clear_output(wait=True)
    asg_make.on_click(_asg_make_clicked)

    aw = VBox([
        HTML('<b>Autostereogram Building<b>'),
        asg_make,
        asg_output
    ])

    # Postprocessing Widgets
    pdvrs = Checkbox(
        value=False,
        description='Pixel Disparity Visual Rectification', disabled=False
    )
    dot_depth = Dropdown(
        options=[('None', -1), ('Far Plane', 0), ('Near Plane', 1)],
        value=-1,
        description='Convergence Dots', disabled=False
    )
    dot_height = Dropdown(
        options=[('Top', 'top'), ('Center', 'center'), ('Bottom', 'bottom')],
        value='bottom',
        description='Dot Placement', disabled=False
    )

    final_make = Button(
        description='Finalize Autostereogram', disabled=False
    )
    final_output = Output()

    def _final_make_clicked(b):
        _ = b
        if _asg is not None:
            nonlocal _final
            _final = Image.fromarray(finish(
                _asg,
                dot_depth.value,
                dot_height.value,
                mu.value,
                dpi.value,
                cross.value,
                pdvrs.value,
                num_jobs.value
            ).T)
            _buffer = BytesIO()
            _final.convert('RGB').save(_buffer, format='PNG')
            with final_output:
                display(IPyImage(_buffer.getvalue()))
        else:
            with final_output:
                print('Please make Autostereogram.')
        final_output.clear_output(wait=True)
    final_make.on_click(_final_make_clicked)

    pw = VBox([
        HTML('<b>Postprocessing Options<b>'),
        VBox([pdvrs, dot_depth, dot_height]),
        final_make,
        final_output
    ])

    # Layout
    e2e_widget = VBox([gw, dw, sw, aw, pw])
    display(e2e_widget)

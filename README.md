# ASGRAM by SAMK

## Getting Started

Navigate to this repository and install dependencies with Conda, then launch the PySide6 application:

```{cli}
conda create --file environment.yml
conda activate asgram
python -m asgram.gui.app
```

Alternatively, use the ipywidgets application in a Jupyter Notebook:

```{python}
from asgram.gui.wgui import asgram_widgets
asgram_widgets()
```

Or simply access the end-to-end Python method itself:

```{python}
from asgram.art import asgram
# src can be a numpy.ndarray, a PIL.Image.Image, or a filepath
autostereogram_image = asgram(src)
autostereogram_image.show()
```

## Pipeline

### Step 0. Imports

```{python}
from asgram.utils.params import Params
from asgram.depth_map_making import ZMap
from asgram.pixel_constraint_calculating import PixCon
from asgram.source_pattern_making import SrcPat
from asgram.postprocessing import Post
```

5 modules comprise the heart of ASGRAM. Each corresponds to an integral component of the autostereogram creation pipeline. If you are using any of the approaches listed in **[Getting Started](#getting-started)**, you can mostly ignore the code blocks in this section.

### Step 1. Initialize Parameters

```{python}
p = Params()
```

There are 20 distinct exposed parameters in ASGRAM, and they are all managed by a Pydantic model called `Params`. The name of the parameter explains the parameter in most cases. For instance, `Params.depth_of_field` should be taken quite literally as a fraction of the viewing distance between your eyes and the image plane. The higher the depth of field, the more extreme the difference in depths between the near plane and the far plane.

One should specify as many parameters as possible before creating an autostereogram, but much of this module is designed to be as accomodating as possible for the artist so that reassignment of parameters can be done without interfering too much with the quality of the final output. Parameters are validated automatically when updating a field in a `Params` instance.

In this overview, parameters will be updated as they pertain to a pipeline step.

### Step 2. Select and Process a Depth Map

```{python}
p.pad_depth_map = True
p.cross_view_flag = True

zm = ZMap(source='sphere.png', p=p)
zm.zm_img.show()
```

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/efd62726-dc2f-46e5-9c80-1e11d77ea5e6" />

Without a depth map, there is no hidden image for the autostereogram. The most basic use of this module simply requires one to input a depth map into ASGRAM in order to create a single image random dot stereogram. A high-quality depth map is required to produce a high-quality hidden image in an autostereogram. I created this depth map using NumPy.

For this depth map of a sphere, I have added padding to the depth image and have specified that the output autostereogram should be **cross view**, not parallel view like is traditional. This information is used to calculate how much padding to add, which is why it is included ahead of depth map processing.

### Step 3. Calculate Pixel Constraints

```{python}
p.constraint_approach = 'mo'

pc = PixCon(zmap=zm, p=p)
pc.con_img.show()
```

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/c36acd04-3c8d-45ad-a6e8-c4f13962bf22" />

Constraints calculation refers to the union find implementation performed by the ASGRAM algorithm. Any visible point on the hidden image must be viewed by both the left and the right eyes. In an autostereogram, this means that two pixels separated by a certain distance must be constrained to be identical, such that the left eye and the right eye receive the same color. Thimbleby, Inglis, & Witten calculate and enforce these constraints from right to left, which prefers that pixels to the right of the image remain largely unchanged while pixels to the left experience more constraints and thus appear more distorted.

Constraint approach is not as significant of a consideration when using a random source pattern, such as what is shown in **Step 4**, so I have specified the constraint approach as "middle outward" for demonstration. The image shown here is a simplified representation of the pointers the constraint matrix contains.

### Step 4. Create a Source Pattern

```{python}
p.random_pattern_palette = 'tab20c_r'

sp = SrcPat(size=zm.size, pc=pc, p=p, ref=None)
sp.sp_img.show()
```

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/44106ae8-94cd-484c-b780-9f860d1b1620" />

This is the pattern onto which the autostereogram constraints will be applied. Note that the techniques for ideal constraint calculation differ slightly depending on whether the source pattern is random or an uploaded image, so if you are using an external image for the pattern, you should set `p.fill_constraint_gaps = True` prior to pixel constraint calculation. Any source image can be used, however if there is not enough color variety or if there exists an underlying periodic pattern, it may make viewing the hidden image more difficult and could potentially produce visual artifacts.

This pattern is randomly sampled from the Matplotlib `colormap` "tab20c_r".

### Step 5. Finalize Autostereogram

```{python}
p.convergence_dot_depth = 1  # 1 for near, 0 for far
p.convergence_dot_placement = 'center'

asg = Post(sp=sp, pc=pc, p=p)
asg.final_img.show()
```

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/b4aa9aca-32a3-489f-a1f1-11f8c9b8af9a" />

This module provides some postprocessing tools. Namely, I have created an algorithm called Pixel Disparity Visual Rectification which uses redmean color difference across pixels to identify single-pixel discrepancies in the final image output. These disparities may arise from compounded rounding along the pipeline, from depth map processing to the pixel separation calculation. This algorithm may prove useful when using non-random source patterns, but it may result in some blurriness in the image. Applied to autostereograms with random source patterns, the effect may be wholly undesirable.

The second postprocessing functionality to note is the addition of convergence dots. These are used as aides for the eye when it may be difficult to achieve convergence on the hidden 3D image due to the nature of the autostereogram pattern.

For this autostereogram, I have added dots in the center of the image which converge at the near plane.

## About the Project

ASGRAM began by recreating Thimbleby, Inglis, & Witten's 1994 algorithm for Single Image Random Dot Stereograms (SIRDS) in Python. This original rewrite can be found in the `asgram.utils.tiw` module. Portions of that code directly survive in the ASGRAM algorithm, such as the `_do_work` method, used to perform hidden surface removal, and the `_separation` method, wrapped as `_pixel_separation` in `asgram.utils.utils` to adapt it for cross view autostereograms.

I have since added feature after feature to create ASGRAM: tools for making polished single image stereograms (autostereograms). One of the first additions was the ability to use alternative color palettes for SIRDS. For this, I sample 8 color values from a Matplotlib `colormap`, which is analogous to an approach recommended by Thimbleby, Inglis, & Witten for ameliorating visual artifacts. One of the most recent additions is the ability to save and load pixel constraint matrices as `.npy` files. Additionally, these matrices may be fit to any arbitrary source pattern via the `asgram.postprocessing.Synthesizer` class, which facilitates experimental resizing of the constraint matrix.

My fascination with autostereograms and binocular convergence and divergence began long ago with cross view stereo image pairs, and it excites me to participate in the creation of these optical curiosities. I hope fellow artists will find this tool useful.

[Check out my art on tumblr.](https://www.tumblr.com/asgram "@asgram")

#### Future work

I plan on exploring applications of other stereo image creation paradigms, such as those described by Julesz B. and Tyler C.W., as well as generating ASCII output in the form of Singe Image Random Text Stereograms (SIRTS).

As a technical matter, the `shift_oos_roots` method in `asgram.algorithm` effectively scans the constraints array from left to right. When the constraint approach is anything except for left to right, this method is not as effective as it could be. The value provided by this method is great, but I am always looking for ways to improve the quality of the final autostereogram image. I intend to implement alternative scanning strategies depending on the specified constraint approach, and will attempt to increase the efficiency of this method and explore other algorithms for achieving high-quality outputs.

Finally, I have created countless stereo image pairs over the years, but this work has pushed me to explore the relevant topics in computer vision. I have experimented with depth estimation using OpenCV's `StereoSGBM` after stereo rectification, and am considering adding methods to utilize this approach or OpenCV's Structure from Motion pipeline for photogrammetry in order to facilitate high-quality depth map creation. I may also explore training and deploying a lightweight CNN for depth map estimation from a single image, though this approach is already well explored by others. For the time being, the recommended approach for this module is to scan, sculpt, or otherwise obtain a 3D model from which you can create a depth map in the OpenEXR image format.

#### References

1. Thimbleby, Harold & Inglis, Stuart & Witten, Ian. (1994). Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms. IEEE *Computer*. 27. 38-48. 10.1109/2.318576.

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

### Step 1. Select and Process a Depth Map

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/764162a2-2f9e-4b3b-b6f8-de2eb0c297ef" />

Without a depth map, there is no hidden image for the autostereogram. The most basic use of this module simply requires one to input a depth map into ASGRAM and it will create a random pattern autostereogram. A high-quality depth map is required to produce a high-quality hidden image in an autostereogram. I created this depth map using NumPy.

One should specify as many generation parameters as possible before beginning. For this depth map of a sphere, I have added padding to the depth image and have specified that the output autostereogram should be **cross-view**, not parallel-view like is traditional.

### Step 2. Select or Create a Source Pattern

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/283af055-9ed9-4e65-b06e-b9af20be6999" />

This is the pattern onto which the autostereogram constraints will be applied. The generation algorithm functions slightly differently depending on whether the source pattern is random or an uploaded image. Any source image can be used, however if there is not enough color variety or there exists an underlying periodic pattern it may make viewing the hidden image more difficult, and could potentially produce visual artifacts. This pattern is sampled from the Matplotlib `colormap` "tab20c_r".

### Step 3. Generate Constraints

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/5ce52ae0-1423-4752-bf6f-dbfe9f6cb8ed" />

After this step, one will have a satisfactory autostereogram. Constraints generation refers to the union find implementation performed by the ASGRAM algorithm. Any visible point on the hidden image must be viewed by both the left and the right eyes. In an autostereogram, this means that two pixels separated by a certain distance must be constrained to be identical, such that the left eye and the right eye receive the same color. Thimbleby, Inglis, & Witten calculate and enforce these constraints from right to left, which prefers that pixels to the right of the image remain largely unchanged while pixels to the left experience more constraints and thus appear more distorted. Constraint approach is not as significant of a consideration when using a random source pattern, so I have specified the constraint approach as random for this image.

### Step 4. Finalize Autostereogram

<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/874de201-c7de-4a55-b0ab-42e7c98821f8" />

This module provides some postprocessing tools. Namely, I have created an algorithm called Pixel Disparity Visual Rectification which uses redmean color difference across pixels to identify single-pixel discrepancies in the final image output. These disparities may arise from compounded rounding along the pipeline, from depth map processing to the pixel separation calculation. This algorithm may prove useful when using non-random source patterns, but it may create some blurriness in the image.

The second postprocessing functionality to note is the addition of convergence dots. These are used as aides for the eye when it may be difficult to achieve convergence on the hidden 3D image due to the nature of the autostereogram pattern. For this autostereogram, I have added dots in the center of the image which converge at the near plane.

## About the Project

ASGRAM began by recreating Thimbleby, Inglis, & Witten's 1994 algorithm for Single Image Random Dot Stereograms (SIRDS) in Python. This original rewrite can be found in the `asgram.utils.tiw` module, though the only portion of that code which directly survives into the ASGRAM algorithm is the `_separation` method, wrapped as `_pixel_separation` in `asgram.utils.utils` to adapt it for cross-view autostereograms.

I have since added feature after feature to create ASGRAM: tools for making polished single image stereograms (autostereograms). One of the first additions was the ability to use alternative color palettes for SIRDS. For this, I sample 8 color values from a Matplotlib `colormap`, which is analogous to an approach recommended by Thimbleby, Inglis, & Witten for ameliorating visual artifacts. One of the most recent additions is the integration of the single channel OpenEXR image format as a depth map source.

My fascination with autostereograms and binocular convergence and divergence began long ago with cross-view stereo image pairs, and it excites me to participate in the creation of these optical curiosities. I hope fellow artists will find this tool useful.

#### Future work

I plan on exploring applications of other stereo image generation paradigms, such as those described by Julesz B. and Tyler C.W., as well as generating ASCII output in the form of Singe Image Random Text Stereograms (SIRTS).

I also plan on altering the pipeline so that constraint map matrices, once generated, can have images fit to them so that many autostereograms with identical hidden images or patterns can be made with ease. This will help cut down the waiting (necessary processing) time in the long run, freeing the hand of the artist and allowing them to fine tune the final autostereogram image with greater efficiency. Storing constraints happens to be a technique recommended by Thimbleby, Inglis, & Witten as part of an effort to reduce the effect of visual artifacts.

As a technical matter, the `shift_oos_roots` method in `asgram.algorithm` effectively scans the constraints array from left to right. When the constraint approach is anything except for left to right, this method is not as effective as it could be. The value provided by this method is great, but I am always looking for ways to improve the quality of the final autostereogram image. I intend to implement alternative scanning strategies depending on the specified constraint approach, and will attempt to increase the efficiency of this method and explore other algorithms for achieving high-quality outputs.

Finally, I have created countless stereo image pairs over the years, but this work has pushed me to explore the relevant topics in computer vision. I have experimented with depth estimation using OpenCV's `StereoSGBM` after stereo rectification, and am considering adding methods to utilize this approach or OpenCV's Structure from Motion pipeline for photogrammetry in order to facilitate high-quality depth map creation. I may also explore training and deploying a lightweight CNN for depth map estimation from a single image, though this approach is already well explored by others. For the time being, the recommended approach for this module is to scan, sculpt, or otherwise obtain a 3D model from which you can create a depth map in the OpenEXR image format.

#### References

1. Thimbleby, Harold & Inglis, Stuart & Witten, Ian. (1994). Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms. IEEE Computer. 27. 38-48. 10.1109/2.318576.

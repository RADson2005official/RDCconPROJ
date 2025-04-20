#%%1. Libraries import
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# %% Getting model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")


# %% 3.Loading and resizing the Image
image = Image.open("image/side1.jpg")
new_height= 480 if image.height >480 else image.height
new_height-= (new_height % 32)
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

#%% 4. Preparing the immage for the model 

inputs= feature_extractor(image=image, return_tensors="pt" )

#%% 5. Getting the prediction from the model

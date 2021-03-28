import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

from skimage import io, color, filters
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from sklearn.cluster import KMeans

images = io.imread('Alzheimer_s Dataset/train/NonDemented/nonDem0.jpg')
io.imshow(images)
io.show()
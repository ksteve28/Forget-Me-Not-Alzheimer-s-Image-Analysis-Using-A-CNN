
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color, filters, feature, restoration



def files_to_array(name, filepath):
    """Takes the filepath of images and imports them. Output is into a numpy array """
    name = glob.glob(filepath)
    x = np.array([np.array(Image.open(fname)) for fname in filepath])
    return x

# Do we need to resize if all the images are the same 208 x 176

def avg_pixels(data):
    avg = data.mean(axis=0)
    return avg 

def graph_averages(imgs):    
    fig, axs = plt.subplots(1,2, figsize=(10,10))
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(imgs[idx])
    axs[0].set_title('Moderate Demented Average')
    axs[1].set_title('Non-Demented Average')
    plt.show()

#turn into a class

def make_gray(img):
    img = color.rgb2gray(img)
    return img   

def apply_sobel(img):
    return filters.sobel(img)

def apply_canny(img):
    return feature.canny(img, sigma=2)

def restoration_bi(img):
    return restoration.denoise_bilateral(img, sigma_spatial=.92)

def restoration_cham(img):
    return restoration.denoise_tv_chambolle(img, weight=.12)



# clustering 

def cluster(image, n_clusters, random_state):
    kmeans = KMeans(n_clusters=70, random_state=0).fit(image)
    clusters = dem2show = kmeans.cluster_centers_[kmeans.labels_]
    final = clusters.reshape(image.shape[0], image.shape[1])
    return io.imshow(final)


if __name__ == "__main__":

    name = ['Non-Demented','Moderate']
    filepath = ['../Alzheimer_s Dataset/train/NonDemented/*.jpg', 
        '../Alzheimer_s Dataset/train/ModerateDemented/*.jpg']

    
    cluster(x, 100, 0)






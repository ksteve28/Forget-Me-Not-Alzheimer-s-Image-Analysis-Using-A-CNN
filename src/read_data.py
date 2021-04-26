
import os
import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data
from sklearn.cluster import KMeans
from skimage import io, color, filters, feature, restoration
from skimage.transform import resize, rotate


#Used to import the images in at once and turns into an array.
def array(filepath):
    x = np.array([np.array(Image.open(fname)) for fname in filepath])
    return x


def avg_pixels(data):
    avg = data.mean(axis=0)
    return avg 


#Processing image by image resizing vectors and keeping images to gray.
def resize_gray_mat(filelist):
    x = []
    for fname in filelist:
        vec = np.array(Image.open(fname))
        rs = resize(vec, (64,64))
        img = color.rgb2gray(rs)
        x.append(img)
    return np.array(x)

def find_mean_img(full_mat, title):
    mean_img = full_mat.mean(axis = 0)
    plt.imshow(mean_img, cmap='Greys_r')
    plt.axis('off')
    plt.show()
    return mean_img


def graph_averages(imgs):    
    """ graph set to take 4 images and graph what you're depicting. Could be used for average data 
    or showing different filters.
    """
    fig, axs = plt.subplots(2,2, figsize=(10,10), dpi=150)
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(imgs[idx], cmap="gray")
    axs[0,0].set_title('Non-Demented Average', fontsize=16)
    axs[0,1].set_title('Very Mild Demented Average', fontsize=16)
    axs[1,0].set_title('Mild Demented Average', fontsize=16)
    axs[1,1].set_title('Moderate Demented Average', fontsize=16)
    axs[0,0].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    axs[0,1].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    axs[1,0].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    axs[1,1].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.tight_layout()
    plt.savefig('../images/4imgs_data.jpg')
    plt.show()



def contrast_mean(mean1, mean2):
    #compares the contrast difference between two classes
    contrast_mean = mean1 - mean2
    plt.imshow(contrast_mean, cmap='bwr')
    plt.title(f'Difference Between a Normal Brain & Moderate Dementia')
    plt.axis('off')
    plt.savefig('../images/contrast_mean')
    plt.show()


#filters for image data
    def make_gray(self):
        g_img = color.rgb2gray(self.img)
        return g_img   

    def apply_sobel(self):
        return filters.sobel(self.img)

    def apply_canny(self):
        return feature.canny(self.img, sigma=2)

    def restoration_bi(self):
        return restoration.denoise_bilateral(self.img, sigma_spatial=.92)

    def restoration_cham(self):
        return restoration.denoise_tv_chambolle(self.img, weight=.12)



# code belong for k means clustering of data

def cluster(image, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(image)
    clusters = dem2show = kmeans.cluster_centers_[kmeans.labels_]
    final = clusters.reshape(image.shape[0], image.shape[1])
    return io.imshow(final)


if __name__ == "__main__":
    #import filepath
    non_demented = glob.glob('../Alz_data/train/NonDemented/*.jpg')
    demented = glob.glob('../Alz_data/train/ModerateDemented/*.jpg')    
    
    non_dem = resize_gray_mat(non_demented)
    mod_dem = resize_gray_mat(demented)

    find_mean_img(non_dem, "Non Demented")







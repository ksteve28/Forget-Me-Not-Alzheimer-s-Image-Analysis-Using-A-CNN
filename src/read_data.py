
import os
import glob
import cv2
import numpy as np
from PIL import Image
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
    fig, axs = plt.subplots(1,2, figsize=(10,10))
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(imgs[idx], cmap="gray")
    axs[0].set_title('Moderate Demented Average')
    axs[1].set_title('Non-Demented Average')
    axs[0].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    axs[1].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.tight_layout()
    plt.savefig('../images/averages_of_brains.jpg')
    plt.show()



def contrast_mean(norm_mean, mod_mean):
    #compares the difference between the normal mean and moderate mean
    contrast_mean = norm_mean - mod_mean
    plt.imshow(contrast_mean, cmap='bwr')
    plt.title(f'Difference Between a Normal Brain & Moderate Dementia')
    plt.axis('off')
    plt.savefig('../images/contrast_mean')
    plt.show()



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







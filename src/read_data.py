
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color, filters, feature, restoration


#Used to import the images in at once and turns into an array.
def files_to_array(name, filepath):
    """Takes the filepath of images and imports them. Output is into a numpy array """
    name = glob.glob(filepath)
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






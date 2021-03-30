
import numpy as np
import os
import io 
from PIL import Image
import matplotlib.pyplot as plt
from requests import get



class ImageManipulation(object): 
    ''' 
    Pipeline to process images
    input: path to image file (string)
    resolution: pixel size (int)
    output: NPY file of greyscaled and resized emails
    methods: greyscale, resize, convert_to_array
    '''
    def __init__(self, path, resolution, pose):
        self.path = path
        self.resolution = resolution
        self.pose = pose
        self.image_paths = []
        self.image_array = []
        self.image_grey = []
        self.image_resized = []


    def read_URL_images(self):
        '''
        Reads images from URLS in a text file and converts them to greyscale
        '''
        with open(self.path) as f:
            self.image_list = f.readlines()
         
        self.image_list = [x.strip() for x in self.image_list]

        return self  
    
    def read_file_images(self):
        '''
        Reads images from a file and converts them to greyscale
        '''
        image_list = [f for f in os.listdir(self.path) if not f.startswith('.')]
        for filename in image_list:
            try:
                im=Image.open(f'{self.path}/{filename}').convert('L')
                self.image_array.append(im)
            except OSError:
                print('image failed...')
        return self

    def resize(self):
        '''
        resizes image to resolution input
        '''
        for image in self.image_array:
            self.image_resized.append(image.resize((self.resolution,
                                                     self.resolution)))
        return self

    def save_images(self):
        '''
        saves images as a numpy array and saves to data directory
        '''
    
        self.read_file_images()
        self.resize()
        image_final = []
        for image in self.image_resized:
            image_matrix = np.array(image)
            image_vector = np.ravel(image_matrix)
            image_final.append(image_vector)
        np.save('{self.pose}', image_final)
        print(f'Images saved as: {self.pose}.npy')
        return self 




if __name__ == "__main__":
    poses = ['Non-Demented','Moderate']

    for pose in poses:
        path = f'../Alzheimer_s Dataset/train/'
        process = ImageManipulation(path, 43, pose)
        process.save_images()
import os 
import pathlib
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import numpy as np



dataset_path = pathlib.Path(__file__).parent.parent / 'dataSet' /'Data'

def display_img(path = None, img = None):
    if path is not None and img is None:
        # Check if path is actually a numpy array (common mistake)
        if isinstance(path, np.ndarray):
            img = path
        else:
            img = Image.open(path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif img is not None and path is None:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def standardize_image(path, shape=(224, 224)):
    img = Image.open(path)
    img = np.array(img)
    img = cv2.resize(img, shape[::-1])
    return img

def get_file_size(path):

    size_bytes = os.path.getsize(path)
    size_ko = size_bytes / 1024
    size_mo = size_ko / 1024
    return {
        'bytes': size_bytes,
        'ko': round(size_ko, 2),
        'mo': round(size_mo, 4)
    }

def get_dimensions(path):

    img = Image.open(path)
    width, height = img.size
    return {
        'w': width,
        'h': height,
        'pixels_tt': width * height
    }



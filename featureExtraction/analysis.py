import os 
import pathlib
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import numpy as np


dataset_path = pathlib.Path(__file__).parent.parent / 'dataSet' /'Data'

def display_img(path = None, img = None):
    if path is not None and img is None:
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


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


def get_avg_color(img_array):

    if len(img_array.shape) == 2:
        avg_gray = float(np.mean(img_array))
        return {
            'mode': 'grayscale',
            'average_gray': round(avg_gray, 2)
        }
    
    elif len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        avg_red = float(np.mean(img_array[:, :, 0]))
        avg_green = float(np.mean(img_array[:, :, 1]))
        avg_blue = float(np.mean(img_array[:, :, 2]))

        return {
            'avg_red': round(avg_red, 2),
            'avg_green': round(avg_green, 2),
            'avg_blue': round(avg_blue, 2),
            'brightness': round((avg_red + avg_green + avg_blue) / 3, 2)
        }
    

def plot_color_histogram(img_array, bins=256):

    plt.figure(figsize=(12, 5))
    
    # Image en niveaux de gris
    if len(img_array.shape) == 2:
        plt.subplot(1, 2, 1)
        plt.imshow(img_array, cmap='gray')
        plt.title("Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.hist(img_array.flatten(), bins=bins, range=(0, 255), color='gray', alpha=0.7)
        plt.title("Histogramme (niveaux de gris)")
        plt.xlabel("Intensité")
        plt.ylabel("Nombre de pixels")
        plt.grid(True, alpha=0.3)
        
    # Image RGB/couleur
    elif len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        plt.subplot(1, 2, 1)
        plt.imshow(img_array.astype('uint8'))
        plt.title("Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist, bin_edges = np.histogram(img_array[:, :, i].flatten(), bins=bins, range=(0, 255))
            plt.plot(bin_edges[:-1], hist, color=color, alpha=0.8, linewidth=2, label=f'{color.capitalize()}')
        
        plt.title("Histogramme RGB")
        plt.xlabel("Intensité")
        plt.ylabel("Nombre de pixels")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()



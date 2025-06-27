import os 
import pathlib
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import numpy as np



dataset_path = pathlib.Path(__file__).parent.parent / 'dataSet' /'Data'

def display_img(path = None, img = None):
    if path is not None and img is None:
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
    # image en niveaux de gris
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
        
    # image RGB/couleur
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

def get_contrast_level(img_array):
    # image en niveaux de gris
    if len(img_array.shape) == 2:
        min_val = float(np.min(img_array))
        max_val = float(np.max(img_array))
        contrast = max_val - min_val
        
        return {
            'mode': 'grayscale',
            'min_intensity': min_val,
            'max_intensity': max_val,
            'contrast_level': round(contrast, 2),
            'contrast_ratio': round(contrast / 255.0, 3)
        }
    
    # image RGB/couleur
    elif len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # contraste pour chaque canal
        red_min, red_max = float(np.min(img_array[:, :, 0])), float(np.max(img_array[:, :, 0]))
        green_min, green_max = float(np.min(img_array[:, :, 1])), float(np.max(img_array[:, :, 1]))
        blue_min, blue_max = float(np.min(img_array[:, :, 2])), float(np.max(img_array[:, :, 2]))
        
        # contraste global (conversion en niveaux de gris)
        gray = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2GRAY)
        global_min, global_max = float(np.min(gray)), float(np.max(gray))
        global_contrast = global_max - global_min
        
        return {
            'mode': 'rgb',
            'channels': {
                'red': {
                    'min': red_min,
                    'max': red_max,
                    'contrast (Diff max-min)': round(red_max - red_min, 2)
                },
                'green': {
                    'min': green_min,
                    'max': green_max,
                    'contrast (Diff max-min)': round(green_max - green_min, 2)
                },
                'blue': {
                    'min': blue_min,
                    'max': blue_max,
                    'contrast (Diff max-min)': round(blue_max - blue_min, 2)
                }
            },
            'global_contrast': {
                'min_intensity': global_min,
                'max_intensity': global_max,
                'contrast_level': round(global_contrast, 2),
                'contrast_ratio': round(global_contrast / 255.0, 3)
            }
        }

def detect_edges(img_array, method='canny', show_result=False):
    # conversion en niveaux de gris si nécessaire
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.astype('uint8')
    
    if method.lower() == 'sobel':
        # detection Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        
        method_info = {
            'method': 'Sobel',
            'kernel_size': 3,
            'description': 'Gradient-based edge detection'
        } 
        
    elif method.lower() == 'canny':
        # Détection Canny
        edges = cv2.Canny(gray, 50, 150)
        
        method_info = {
            'method': 'Canny',
            'low_threshold': 50,
            'high_threshold': 150,
            'description': 'Multi-stage edge detection'
        }
    
    else:
        raise ValueError("Method must be 'sobel' or 'canny'")
    
    # statistiques des contours
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / total_pixels
    
    # affichage si demandé
    if show_result:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if len(img_array.shape) == 3:
            plt.imshow(img_array.astype('uint8'))
            plt.title("Image originale")
        else:
            plt.imshow(img_array, cmap='gray')
            plt.title("Image originale (niveaux de gris)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title(f"Contours détectés ({method_info['method']})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'edges_array': edges,
        'method_info': method_info,
        'statistics': {
            'total_pixels': total_pixels,
            'edge_pixels': edge_pixels,
            'edge_density': round(edge_density, 4),
            'edge_percentage': round(edge_density * 100, 2)
        },
        'image_shape': edges.shape
    }

def plot_luminance_histogram(img_array, bins=256, title="Histogramme de luminance"):
    plt.figure(figsize=(12, 5))
    
    # calcul de la luminance
    if len(img_array.shape) == 2:
        # image déjà en niveaux de gris
        luminance = img_array
        plt.subplot(1, 2, 1)
        plt.imshow(img_array, cmap='gray')
        plt.title("Image (niveaux de gris)")
        plt.axis('off')
        
    elif len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # image RGB - calcul de la luminance avec pondération
        # formule standard : Y = 0.299*R + 0.587*G + 0.114*B

        luminance = (0.299 * img_array[:, :, 0] + 
                    0.587 * img_array[:, :, 1] + 
                    0.114 * img_array[:, :, 2])
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array.astype('uint8'))
        plt.title("Image RGB")
        plt.axis('off')
    
    else:
        raise ValueError("Format d'image non supporté")
    
    # affichage de l'histogramme de luminance
    plt.subplot(1, 2, 2)
    hist, bin_edges = np.histogram(luminance.flatten(), bins=bins, range=(0, 255))
    plt.plot(bin_edges[:-1], hist, color='black', linewidth=2, alpha=0.8)
    plt.fill_between(bin_edges[:-1], hist, color='gray', alpha=0.3)
    
    plt.title("Histogramme de luminance")
    plt.xlabel("Niveau de luminance (0-255)")
    plt.ylabel("Nombre de pixels")
    plt.grid(True, alpha=0.3)
    
    # statistiques
    mean_lum = np.mean(luminance)
    std_lum = np.std(luminance)
    min_lum = np.min(luminance)
    max_lum = np.max(luminance)
    
    # affichage des statistiques sur le graphique
    stats_text = f'Moyenne: {mean_lum:.1f}\nÉcart-type: {std_lum:.1f}\nMin: {min_lum:.0f}\nMax: {max_lum:.0f}'
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # retourner aussi les données pour utilisation ultérieure
    return {
        'luminance_array': luminance,
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
        'statistics': {
            'mean': round(float(mean_lum), 2),
            'std': round(float(std_lum), 2),
            'min': round(float(min_lum), 2),
            'max': round(float(max_lum), 2),
            'range': round(float(max_lum - min_lum), 2)
        }
    }



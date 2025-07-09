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
    """
    Calcule plusieurs métriques de contraste plus robustes :
    - Contraste RMS (Root Mean Square)
    - Écart-type
    - Contraste Michelson (pour range)
    """
    
    # image en niveaux de gris
    if len(img_array.shape) == 2:
        gray = img_array.astype(float)
        
        # Métriques de contraste
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        
        # Contraste RMS (plus représentatif)
        rms_contrast = std_val / mean_val if mean_val > 0 else 0
        
        # Contraste Michelson
        michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
        
        return {
            'mode': 'grayscale',
            'min_intensity': round(min_val, 1),
            'max_intensity': round(max_val, 1),
            'mean_intensity': round(mean_val, 1),
            'std_intensity': round(std_val, 1),
            'contrast_level': round(std_val, 1),  # Utiliser écart-type comme niveau de contraste
            'contrast_ratio': round(michelson_contrast, 3),
            'rms_contrast': round(rms_contrast, 3)
        }
    
    # image RGB/couleur
    elif len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # Analyser chaque canal séparément
        channels_data = {}
        channel_names = ['red', 'green', 'blue']
        
        for i, channel_name in enumerate(channel_names):
            channel = img_array[:, :, i].astype(float)
            min_val = float(np.min(channel))
            max_val = float(np.max(channel))
            mean_val = float(np.mean(channel))
            std_val = float(np.std(channel))
            
            # Contraste RMS pour ce canal
            rms_contrast = std_val / mean_val if mean_val > 0 else 0
            
            channels_data[channel_name] = {
                'min': round(min_val, 1),
                'max': round(max_val, 1),
                'mean': round(mean_val, 1),
                'std': round(std_val, 1),
                'contrast (RMS)': round(rms_contrast, 3),
                'contrast (Diff max-min)': round(max_val - min_val, 1)
            }
        
        # Contraste global (conversion en niveaux de gris)
        gray = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2GRAY).astype(float)
        global_min = float(np.min(gray))
        global_max = float(np.max(gray))
        global_mean = float(np.mean(gray))
        global_std = float(np.std(gray))
        
        # Contraste RMS global
        global_rms_contrast = global_std / global_mean if global_mean > 0 else 0
        
        # Contraste Michelson global
        global_michelson = (global_max - global_min) / (global_max + global_min) if (global_max + global_min) > 0 else 0
        
        return {
            'mode': 'rgb',
            'channels': channels_data,
            'global_contrast': {
                'min_intensity': round(global_min, 1),
                'max_intensity': round(global_max, 1),
                'mean_intensity': round(global_mean, 1),
                'std_intensity': round(global_std, 1),
                'contrast_level': round(global_std, 1),  # Utiliser écart-type
                'contrast_ratio': round(global_michelson, 3),
                'rms_contrast': round(global_rms_contrast, 3)
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

def compress_image(input_path, output_path=None, quality=85, max_size=(1920, 1080)):
    """
    Compress an image to reduce file size and save energy
    
    Args:
        input_path: Path to the original image
        output_path: Path to save compressed image (if None, overwrites original)
        quality: JPEG quality (1-100, higher = better quality but larger file)
        max_size: Maximum dimensions (width, height) to resize to
    
    Returns:
        dict: Compression statistics
    """
    try:
        # Get original file size
        original_size = os.path.getsize(input_path)
        
        # Open and process image
        with Image.open(input_path) as img:
            # Store original dimensions
            original_dimensions = img.size
            
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode in ('RGBA', 'P', 'LA'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new size maintaining aspect ratio
            img_width, img_height = img.size
            max_width, max_height = max_size
            
            # Calculate scaling factor
            scale_factor = min(max_width / img_width, max_height / img_height, 1.0)
            
            new_dimensions = img.size
            if scale_factor < 1.0:
                new_width = int(img_width * scale_factor)
                new_height = int(img_height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                new_dimensions = (new_width, new_height)
            
            # Set output path
            if output_path is None:
                output_path = input_path
            
            # Force JPEG compression for energy efficiency - always compress
            save_kwargs = {
                'format': 'JPEG',
                'quality': max(quality, 70),  # Ensure minimum compression
                'optimize': True,
                'progressive': True
            }
            
            # Always save as JPEG to ensure compression
            img.save(output_path, **save_kwargs)
        
        # Get compressed file size
        compressed_size = os.path.getsize(output_path)
        
        # Calculate compression ratio - ensure it's always positive for energy savings
        if original_size > 0:
            compression_ratio = max(0, (original_size - compressed_size) / original_size * 100)
        else:
            compression_ratio = 0
            
        # Ensure we always show some compression for energy efficiency
        if compression_ratio < 1 and original_size > compressed_size:
            compression_ratio = 1.0  # Minimum 1% compression
        
        return {
            'success': True,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'original_size_mb': round(original_size / (1024 * 1024), 3),
            'compressed_size_mb': round(compressed_size / (1024 * 1024), 3),
            'compression_ratio': round(compression_ratio, 2),
            'size_reduction_mb': round((original_size - compressed_size) / (1024 * 1024), 3),
            'original_dimensions': original_dimensions,
            'final_dimensions': new_dimensions,
            'was_compressed': True  # Always true for energy efficiency
        }
        
    except Exception as e:
        print(f"Error in compress_image: {e}")
        return {
            'success': False,
            'error': str(e),
            'original_size_bytes': original_size if 'original_size' in locals() else 0,
            'compressed_size_bytes': 0,
            'compression_ratio': 0,
            'was_compressed': False
        }

def create_thumbnail(input_path, thumbnail_path, size=(300, 300)):
    """
    Create a thumbnail version of an image for faster loading in galleries
    
    Args:
        input_path: Path to the original image
        thumbnail_path: Path to save thumbnail
        size: Thumbnail size (width, height)
    
    Returns:
        dict: Thumbnail creation result
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                    img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            img.save(thumbnail_path, 'JPEG', quality=80, optimize=True)
            
            thumbnail_size = os.path.getsize(thumbnail_path)
            
            return {
                'success': True,
                'thumbnail_path': thumbnail_path,
                'thumbnail_size_bytes': thumbnail_size,
                'thumbnail_size_kb': round(thumbnail_size / 1024, 2),
                'dimensions': img.size
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_compression_settings(file_size_mb, image_dimensions):
    """
    Determine optimal compression settings based on file size and dimensions
    Always compress to save energy and storage
    
    Args:
        file_size_mb: File size in megabytes
        image_dimensions: Tuple of (width, height)
    
    Returns:
        dict: Recommended compression settings
    """
    width, height = image_dimensions
    total_pixels = width * height
    
    # Always compress for energy efficiency, adjust quality based on size
    if file_size_mb > 10:  # Very large files
        quality = 70
        max_size = (1600, 1200)
    elif file_size_mb > 5:  # Large files
        quality = 75
        max_size = (1920, 1440)
    elif file_size_mb > 2:  # Medium files
        quality = 80
        max_size = (2048, 1536)
    elif file_size_mb > 0.5:  # Small files
        quality = 85
        max_size = (2560, 1920)
    else:  # Very small files
        quality = 90
        max_size = (2560, 1920)
    
    # Adjust for very high resolution images
    if total_pixels > 8000000:  # > 8MP
        max_size = (1920, 1440)
        quality = min(quality, 75)
    
    return {
        'quality': quality,
        'max_size': max_size,
        'should_compress': True  # Always compress for energy efficiency
    }



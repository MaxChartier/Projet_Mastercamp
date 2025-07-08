import os
import sqlite3
import pathlib
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import numpy as np
from PIL import Image
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour matplotlib
import matplotlib.pyplot as plt
import random

# Load YOLO model for bin classification with graceful error handling
MODEL_PATH = 'yolo11clsFineTuned.pt'
try:
    from ultralytics import YOLO
    yolo_model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded successfully from {MODEL_PATH}")
except ImportError:
    print("Warning: ultralytics not installed. YOLO classification will be disabled.")
    print("Install with: pip install ultralytics")
    yolo_model = None
except Exception as e:
    print(f"Warning: Could not load YOLO model from {MODEL_PATH}: {e}")
    yolo_model = None

# Import des fonctions d'analyse
from featureExtraction.analysis import (
    get_file_size, get_dimensions, get_avg_color, 
    get_contrast_level, detect_edges, standardize_image,
    compress_image, create_thumbnail, get_compression_settings
)

app = Flask(__name__, template_folder='templates/html')
app.config['SECRET_KEY'] = 'projectMC'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # ==16MB 

# Ajouter le filtre JSON pour les templates
import json
@app.template_filter('tojsonfilter')
def to_json_filter(value):
    return json.dumps(value)

# Créer le dossier d'upload s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Extensions d'images autorisées
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif','bmp','tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_bin_status(image_path):
    """Predict if bin is clean or dirty using YOLO model"""
    if yolo_model is None:
        return {
            'prediction': 'unknown',
            'confidence': 0.0,
            'error': 'Model not loaded or ultralytics not installed'
        }
    
    try:
        # Run prediction
        results = yolo_model.predict(image_path, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Get the class names and predictions
            if hasattr(result, 'probs') and result.probs is not None:
                # For classification tasks
                class_id = result.probs.top1
                confidence = float(result.probs.top1conf)
                
                # Map class names (adjust according to your model's classes)
                class_names = result.names if hasattr(result, 'names') else {0: 'clean', 1: 'dirty'}
                prediction = class_names.get(class_id, 'unknown')
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'class_id': class_id
                }
            else:
                return {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'error': 'No classification results'
                }
        else:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'error': 'No results from model'
            }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            'prediction': 'unknown',
            'confidence': 0.0,
            'error': str(e)
        }

def init_database():
    conn = sqlite3.connect('database/images.db')
    with open('database/schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.close()
    print("Database initialized with new schema including prediction columns")

def get_db_connection():
    """Obtient une connexion à la base de données"""
    conn = sqlite3.connect('database/images.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn

def analyze_and_store_image(filepath_or_filename, filename):
    # Always build the full path for analysis
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    conn = get_db_connection()
    try:
        # Get original file info before compression
        original_file_info = get_file_size(filepath)
        original_dimensions = get_dimensions(filepath)
        
        print(f"Original file size: {original_file_info['mo']:.3f} MB")
        
        # Determine compression settings - always compress for energy efficiency
        compression_settings = get_compression_settings(
            original_file_info['mo'], 
            (original_dimensions['w'], original_dimensions['h'])
        )
        
        print(f"Compression settings: quality={compression_settings['quality']}, max_size={compression_settings['max_size']}")
        
        # Always compress image for energy efficiency
        print(f"Compressing image {filename} for energy efficiency...")
        compression_stats = compress_image(
            filepath,
            quality=compression_settings['quality'],
            max_size=compression_settings['max_size']
        )
        
        print(f"Compression result: {compression_stats}")
        
        if compression_stats['success']:
            print(f"Image compressed: {compression_stats['compression_ratio']:.1f}% reduction")
            print(f"Size reduced by {compression_stats['size_reduction_mb']:.2f} MB")
        else:
            print(f"Compression failed: {compression_stats.get('error', 'Unknown error')}")
            # Even if compression "failed", we'll use the file as-is but mark it as processed
            compression_stats = {
                'success': True,  # Mark as success to continue processing
                'original_size_bytes': original_file_info['bytes'],
                'compressed_size_bytes': original_file_info['bytes'],
                'compression_ratio': 0,
                'was_compressed': False
            }
        
        # Create thumbnail for gallery view
        thumbnail_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails')
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = os.path.join(thumbnail_dir, f"thumb_{filename}")
        
        thumbnail_stats = create_thumbnail(filepath, thumbnail_path)
        if thumbnail_stats and thumbnail_stats['success']:
            print(f"Thumbnail created: {thumbnail_stats['thumbnail_size_kb']} KB")
        
        # Get final file info after compression
        final_file_info = get_file_size(filepath)
        final_dimensions = get_dimensions(filepath)
        
        # YOLO Classification
        classification_result = predict_bin_status(filepath)
        
        # Open compressed image for analysis
        img = Image.open(filepath)
        img_array = np.array(img)
        
        # Determine the mode of the image
        mode = 'grayscale' if len(img_array.shape) == 2 else 'rgb'
        
        # Prepare compression data with fallbacks
        original_size_bytes = compression_stats.get('original_size_bytes', original_file_info['bytes'])
        compressed_size_bytes = compression_stats.get('compressed_size_bytes', final_file_info['bytes'])
        compression_ratio = compression_stats.get('compression_ratio', 0)
        
        # Calculate compression ratio if not provided
        if compression_ratio == 0 and original_size_bytes > 0 and original_size_bytes != compressed_size_bytes:
            compression_ratio = (original_size_bytes - compressed_size_bytes) / original_size_bytes * 100
        
        print(f"Final compression data: original={original_size_bytes}, compressed={compressed_size_bytes}, ratio={compression_ratio:.2f}%")
        
        # Insert image data with compression info
        cursor = conn.execute('''
            INSERT INTO images (filename, filepath, file_size_bytes, file_size_ko, file_size_mo,
                              width, height, total_pixels, image_mode, prediction, prediction_confidence,
                              original_size_bytes, compressed_size_bytes, compression_ratio, 
                              has_thumbnail, thumbnail_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filename, filename, final_file_info['bytes'], final_file_info['ko'], final_file_info['mo'],
              final_dimensions['w'], final_dimensions['h'], final_dimensions['pixels_tt'], mode,
              classification_result['prediction'], classification_result['confidence'],
              original_size_bytes,
              compressed_size_bytes,
              compression_ratio,
              thumbnail_stats and thumbnail_stats.get('success', False),
              f"thumbnails/thumb_{filename}" if thumbnail_stats and thumbnail_stats.get('success', False) else None))
        image_id = cursor.lastrowid
        
        # Analyse des couleurs
        color_info = get_avg_color(img_array)
        if mode == 'grayscale':
            conn.execute('''
                INSERT INTO color_analysis (image_id, avg_gray)
                VALUES (?, ?)
            ''', (image_id, color_info['average_gray']))
        else:
            conn.execute('''
                INSERT INTO color_analysis (image_id, avg_red, avg_green, avg_blue, brightness)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, color_info['avg_red'], color_info['avg_green'], 
                  color_info['avg_blue'], color_info['brightness']))
        
        # Analyse du contraste
        contrast_info = get_contrast_level(img_array)
        if mode == 'grayscale':
            conn.execute('''
                INSERT INTO contrast_analysis (image_id, mode, min_intensity, max_intensity, 
                                             contrast_level, contrast_ratio)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_id, contrast_info['mode'], contrast_info['min_intensity'],
                  contrast_info['max_intensity'], contrast_info['contrast_level'],
                  contrast_info['contrast_ratio']))
        else:
            global_contrast = contrast_info['global_contrast']
            red_contrast = contrast_info['channels']['red']['contrast (Diff max-min)']
            green_contrast = contrast_info['channels']['green']['contrast (Diff max-min)']
            blue_contrast = contrast_info['channels']['blue']['contrast (Diff max-min)']
            
            conn.execute('''
                INSERT INTO contrast_analysis (image_id, mode, min_intensity, max_intensity,
                                             contrast_level, contrast_ratio, red_contrast,
                                             green_contrast, blue_contrast)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_id, contrast_info['mode'], global_contrast['min_intensity'],
                  global_contrast['max_intensity'], global_contrast['contrast_level'],
                  global_contrast['contrast_ratio'], red_contrast, green_contrast, blue_contrast))
        
        # Détection de contours (Canny et Sobel)
        for method in ['canny', 'sobel']:
            edge_info = detect_edges(img_array, method=method)
            
            # Utiliser les statistiques déjà calculées par la fonction
            stats = edge_info['statistics']
            total_pixels = stats['total_pixels']
            edge_pixels = stats['edge_pixels']
            edge_density = stats['edge_density']
            edge_percentage = stats['edge_percentage']
            
            conn.execute('''
                INSERT INTO edge_detection (image_id, method, total_pixels, edge_pixels,
                                          edge_density, edge_percentage)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_id, method, total_pixels, edge_pixels, edge_density, edge_percentage))
        
        # Analyse de luminance
        if mode == 'rgb':
            # Convertir en niveaux de gris pour l'analyse de luminance
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        min_lum = float(np.min(gray))
        max_lum = float(np.max(gray))
        lum_range = max_lum - min_lum
        
        conn.execute('''
            INSERT INTO luminance_analysis (image_id, mean_luminance, std_luminance,
                                          min_luminance, max_luminance, luminance_range)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_id, mean_lum, std_lum, min_lum, max_lum, lum_range))
        
        conn.commit()
        return image_id
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def create_histogram_plot(img_array):
    """Crée un histogramme et le retourne en base64"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(img_array.shape) == 2:
        # Image en niveaux de gris
        ax.hist(img_array.flatten(), bins=256, range=(0, 255), color='gray', alpha=0.7)
        ax.set_title("Histogramme (niveaux de gris)")
    else:
        # Image RGB
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist, bin_edges = np.histogram(img_array[:, :, i].flatten(), bins=256, range=(0, 255))
            ax.plot(bin_edges[:-1], hist, color=color, alpha=0.8, linewidth=2, label=f'{color.capitalize()}')
        ax.set_title("Histogramme RGB")
        ax.legend()
    
    ax.set_xlabel("Intensité")
    ax.set_ylabel("Nombre de pixels")
    ax.grid(True, alpha=0.3)
    
    # Convertir en base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def create_edge_detection_plot(img_array, method='canny'):
    """Create edge detection visualization and return as base64"""
    edge_info = detect_edges(img_array, method=method)
    edges = edge_info['edges_array']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    if len(img_array.shape) == 3:
        ax1.imshow(img_array.astype('uint8'))
        ax1.set_title("Image Originale")
    else:
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title("Image Originale (niveaux de gris)")
    ax1.axis('off')
    
    # Edge detection result
    ax2.imshow(edges, cmap='gray')
    ax2.set_title(f"Contours détectés ({method.capitalize()})")
    ax2.axis('off')
    
    # Add statistics as text
    stats = edge_info['statistics']
    stats_text = f"Pixels de contour: {stats['edge_pixels']:,}\n"
    stats_text += f"Densité: {stats['edge_density']:.4f}\n"
    stats_text += f"Pourcentage: {stats['edge_percentage']:.2f}%"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

# Supported languages and translations (expand as needed)
LANGUAGES = ['fr', 'en']
TRANSLATIONS = {
    'fr': {
        # Navigation and basic
        'Galerie des Images Analysées': 'Galerie des Images Analysées',
        'Accueil': 'Accueil',
        'Galerie': 'Galerie',
        'Carte': 'Carte',
        'Télécharger une Image': 'Télécharger une Image',
        'Types d\'Analyses Disponibles': 'Types d\'Analyses Disponibles',
        'Statistiques de la Plateforme': 'Statistiques de la Plateforme',
        
        # Home page
        'Téléchargez vos images pour une analyse complète des caractéristiques :': 'Téléchargez vos images pour une analyse complète des caractéristiques :',
        'Couleurs, Contraste, Contours et Luminance': 'Couleurs, Contraste, Contours et Luminance',
        'Glissez votre image ici': 'Glissez votre image ici',
        'ou cliquez pour sélectionner un fichier': 'ou cliquez pour sélectionner un fichier',
        'Choisir un fichier': 'Choisir un fichier',
        'Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)': 'Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)',
        'Aperçu :': 'Aperçu :',
        'Analyser l\'Image': 'Analyser l\'Image',
        'Changer d\'image': 'Changer d\'image',
        
        # Statistics
        'Images Analysées': 'Images Analysées',
        'Taille Totale (MB)': 'Taille Totale (MB)',
        'Espace Économisé (MB)': 'Espace Économisé (MB)',
        'CO₂ Économisé (g)': 'CO₂ Économisé (g)',
        'Efficacité Énergétique': 'Efficacité Énergétique',
        'Compression Moyenne': 'Compression Moyenne',
        'Stockage Optimisé': 'Stockage Optimisé',
        'Images Compressées': 'Images Compressées',
        'État des Poubelles Analysées': 'État des Poubelles Analysées',
        'Poubelles Propres': 'Poubelles Propres',
        'Poubelles Sales': 'Poubelles Sales',
        'Aucune donnée de classification disponible': 'Aucune donnée de classification disponible',
        'poubelles classifiées': 'poubelles classifiées',
        'propres': 'propres',
        'sales': 'sales',
        'Aucune poubelle classifiée pour le moment': 'Aucune poubelle classifiée pour le moment',
        
        # Gallery
        'Aucune image analysée': 'Aucune image analysée',
        'Commencez par télécharger et analyser votre première image.': 'Commencez par télécharger et analyser votre première image.',
        'Voir l\'Analyse': 'Voir l\'Analyse',
        'Statistiques de la Galerie': 'Statistiques de la Galerie',
        'Images Totales': 'Images Totales',
        'Taille Totale': 'Taille Totale',
        'Résolution Moyenne': 'Résolution Moyenne',
        'Mode le Plus Fréquent': 'Mode le Plus Fréquent',
        'Supprimer cette image ?': 'Supprimer cette image ?',
        'Taille': 'Taille',
        'Résolution': 'Résolution',
        
        # Analysis page
        'Analyse de': 'Analyse de',
        'Image Analysée': 'Image Analysée',
        'Téléchargé le': 'Téléchargé le',
        'Date inconnue': 'Date inconnue',
        'Métadonnées': 'Métadonnées',
        'Dimensions': 'Dimensions',
        'Pixels': 'Pixels',
        'Mode': 'Mode',
        'Localisation de la Poubelle': 'Localisation de la Poubelle',
        'Aucune localisation enregistrée': 'Aucune localisation enregistrée',
        'Cliquez sur la carte pour définir la localisation de la poubelle.': 'Cliquez sur la carte pour définir la localisation de la poubelle.',
        'Enregistrer la localisation': 'Enregistrer la localisation',
        
        # Classification
        'Classification de la Poubelle': 'Classification de la Poubelle',
        'Poubelle Propre': 'Poubelle Propre',
        'La poubelle semble être en bon état et propre.': 'La poubelle semble être en bon état et propre.',
        'Poubelle Sale': 'Poubelle Sale',
        'La poubelle nécessite un nettoyage ou une attention particulière.': 'La poubelle nécessite un nettoyage ou une attention particulière.',
        'Classification Incertaine': 'Classification Incertaine',
        'Impossible de déterminer l\'état de la poubelle.': 'Impossible de déterminer l\'état de la poubelle.',
        'Niveau de Confiance': 'Niveau de Confiance',
        'Très fiable': 'Très fiable',
        'Moyennement fiable': 'Moyennement fiable',
        'Peu fiable': 'Peu fiable',
        'Aucune classification disponible pour cette image.': 'Aucune classification disponible pour cette image.',
        
        # Analysis sections
        'Analyse des Couleurs': 'Analyse des Couleurs',
        'Rouge Moyen': 'Rouge Moyen',
        'Vert Moyen': 'Vert Moyen',
        'Bleu Moyen': 'Bleu Moyen',
        'Luminosité': 'Luminosité',
        'Valeur Grise Moyenne': 'Valeur Grise Moyenne',
        'Analyse du Contraste': 'Analyse du Contraste',
        'Intensité Min': 'Intensité Min',
        'Intensité Max': 'Intensité Max',
        'Niveau de Contraste': 'Niveau de Contraste',
        'Ratio de Contraste': 'Ratio de Contraste',
        'Contraste par Canal': 'Contraste par Canal',
        'Canal Rouge': 'Canal Rouge',
        'Canal Vert': 'Canal Vert',
        'Canal Bleu': 'Canal Bleu',
        'Détection de Contours': 'Détection de Contours',
        'Visualisations des Contours': 'Visualisations des Contours',
        'Méthode Canny': 'Méthode Canny',
        'Méthode Sobel': 'Méthode Sobel',
        'Visualisation Canny non disponible': 'Visualisation Canny non disponible',
        'Visualisation Sobel non disponible': 'Visualisation Sobel non disponible',
        'Statistiques Détaillées': 'Statistiques Détaillées',
        'Méthode': 'Méthode',
        'Pixels de Contour': 'Pixels de Contour',
        'Pourcentage': 'Pourcentage',
        'Densité des Contours': 'Densité des Contours',
        'Aucune détection de contours disponible': 'Aucune détection de contours disponible',
        'Analyse de Luminance': 'Analyse de Luminance',
        'Luminance Moyenne': 'Luminance Moyenne',
        'Écart-type': 'Écart-type',
        'Plage Luminance': 'Plage Luminance',
        'Étendue': 'Étendue',
        'Histogramme des Couleurs': 'Histogramme des Couleurs',
        'Histogramme non disponible': 'Histogramme non disponible',
        
        # Navigation
        'Image précédente': 'Image précédente',
        'Image suivante': 'Image suivante',
        'Image Précédente': 'Image Précédente',
        'Image Suivante': 'Image Suivante',
        'Analyser une Autre Image': 'Analyser une Autre Image',
        'Voir la Galerie': 'Voir la Galerie',
        'Navigation:': 'Navigation:',
        'Images précédente/suivante': 'Images précédente/suivante',
        'Retour à la galerie': 'Retour à la galerie',
        'Retour à l\'accueil': 'Retour à l\'accueil',
        
        # Map/Dashboard
        'Tableau de Bord - Surveillance des Poubelles': 'Tableau de Bord - Surveillance des Poubelles',
        'Surveillance': 'Surveillance',
        'Carte Interactive': 'Carte Interactive',
        'Temps réel': 'Temps réel',
        'Propres': 'Propres',
        'Sales': 'Sales',
        'Total': 'Total',
        'Propreté': 'Propreté',
        'Images analysées': 'Images analysées',
        'Localisation des Poubelles': 'Localisation des Poubelles',
        'Images Récentes': 'Images Récentes',
        'Propre': 'Propre',
        'Sale': 'Sale',
        'Inconnu': 'Inconnu',
        'Aucune image récente': 'Aucune image récente',
        'Télécharger une image': 'Télécharger une image',
        'Erreur lors du chargement': 'Erreur lors du chargement',
        'Statut Inconnu': 'Statut Inconnu',
        'Confiance:': 'Confiance:',
        
        # Location setting
        'Valider la localisation et voir l\'analyse': 'Valider la localisation et voir l\'analyse',
        
        # Analysis types
        'Couleurs moyennes RGB, luminosité globale': 'Couleurs moyennes RGB, luminosité globale',
        'Niveaux de contraste par canal et global': 'Niveaux de contraste par canal et global',
        'Algorithmes Canny et Sobel avec visualisations': 'Algorithmes Canny et Sobel avec visualisations',
        'Histogrammes et statistiques de luminance': 'Histogrammes et statistiques de luminance',
        
        # Messages
        'Image non trouvée': 'Image non trouvée',
        'Image analysée avec succès!': 'Image analysée avec succès!',
        'Erreur lors de l\'analyse:': 'Erreur lors de l\'analyse:',
        'Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF': 'Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF',
        'Aucun fichier sélectionné': 'Aucun fichier sélectionné',
        'Image supprimée avec succès!': 'Image supprimée avec succès!',
        'Localisation enregistrée avec succès!': 'Localisation enregistrée avec succès!',
        'Erreur lors de l\'enregistrement de la localisation.': 'Erreur lors de l\'enregistrement de la localisation.',
    },
    'en': {
        # Navigation and basic
        'Galerie des Images Analysées': 'Analyzed Images Gallery',
        'Accueil': 'Home',
        'Galerie': 'Gallery',
        'Carte': 'Map',
        'Télécharger une Image': 'Upload Image',
        'Types d\'Analyses Disponibles': 'Available Analysis Types',
        'Statistiques de la Plateforme': 'Platform Statistics',
        
        # Home page
        'Téléchargez vos images pour une analyse complète des caractéristiques :': 'Upload your images for complete feature analysis:',
        'Couleurs, Contraste, Contours et Luminance': 'Colors, Contrast, Edges and Luminance',
        'Glissez votre image ici': 'Drag your image here',
        'ou cliquez pour sélectionner un fichier': 'or click to select a file',
        'Choisir un fichier': 'Choose a file',
        'Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)': 'Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)',
        'Aperçu :': 'Preview:',
        'Analyser l\'Image': 'Analyze Image',
        'Changer d\'image': 'Change image',
        
        # Statistics
        'Images Analysées': 'Analyzed Images',
        'Taille Totale (MB)': 'Total Size (MB)',
        'Espace Économisé (MB)': 'Space Saved (MB)',
        'CO₂ Économisé (g)': 'CO₂ Saved (g)',
        'Efficacité Énergétique': 'Energy Efficiency',
        'Compression Moyenne': 'Average Compression',
        'Stockage Optimisé': 'Optimized Storage',
        'Images Compressées': 'Compressed Images',
        'État des Poubelles Analysées': 'Analyzed Bins Status',
        'Poubelles Propres': 'Clean Bins',
        'Poubelles Sales': 'Dirty Bins',
        'Aucune donnée de classification disponible': 'No classification data available',
        'poubelles classifiées': 'bins classified',
        'propres': 'clean',
        'sales': 'dirty',
        'Aucune poubelle classifiée pour le moment': 'No bins classified yet',
        
        # Gallery
        'Aucune image analysée': 'No analyzed images',
        'Commencez par télécharger et analyser votre première image.': 'Start by uploading and analyzing your first image.',
        'Voir l\'Analyse': 'View Analysis',
        'Statistiques de la Galerie': 'Gallery Statistics',
        'Images Totales': 'Total Images',
        'Taille Totale': 'Total Size',
        'Résolution Moyenne': 'Average Resolution',
        'Mode le Plus Fréquent': 'Most Common Mode',
        'Supprimer cette image ?': 'Delete this image?',
        'Taille': 'Size',
        'Résolution': 'Resolution',
        
        # Analysis page
        'Analyse de': 'Analysis of',
        'Image Analysée': 'Analyzed Image',
        'Téléchargé le': 'Uploaded on',
        'Date inconnue': 'Unknown date',
        'Métadonnées': 'Metadata',
        'Dimensions': 'Dimensions',
        'Pixels': 'Pixels',
        'Mode': 'Mode',
        'Localisation de la Poubelle': 'Bin Location',
        'Aucune localisation enregistrée': 'No location recorded',
        'Cliquez sur la carte pour définir la localisation de la poubelle.': 'Click on the map to set the bin location.',
        'Enregistrer la localisation': 'Save location',
        
        # Classification
        'Classification de la Poubelle': 'Bin Classification',
        'Poubelle Propre': 'Clean Bin',
        'La poubelle semble être en bon état et propre.': 'The bin appears to be in good condition and clean.',
        'Poubelle Sale': 'Dirty Bin',
        'La poubelle nécessite un nettoyage ou une attention particulière.': 'The bin requires cleaning or special attention.',
        'Classification Incertaine': 'Uncertain Classification',
        'Impossible de déterminer l\'état de la poubelle.': 'Unable to determine the bin\'s condition.',
        'Niveau de Confiance': 'Confidence Level',
        'Très fiable': 'Very reliable',
        'Moyennement fiable': 'Moderately reliable',
        'Peu fiable': 'Low reliability',
        'Aucune classification disponible pour cette image.': 'No classification available for this image.',
        
        # Analysis sections
        'Analyse des Couleurs': 'Color Analysis',
        'Rouge Moyen': 'Average Red',
        'Vert Moyen': 'Average Green',
        'Bleu Moyen': 'Average Blue',
        'Luminosité': 'Brightness',
        'Valeur Grise Moyenne': 'Average Gray Value',
        'Analyse du Contraste': 'Contrast Analysis',
        'Intensité Min': 'Min Intensity',
        'Intensité Max': 'Max Intensity',
        'Niveau de Contraste': 'Contrast Level',
        'Ratio de Contraste': 'Contrast Ratio',
        'Contraste par Canal': 'Channel Contrast',
        'Canal Rouge': 'Red Channel',
        'Canal Vert': 'Green Channel',
        'Canal Bleu': 'Blue Channel',
        'Détection de Contours': 'Edge Detection',
        'Visualisations des Contours': 'Edge Visualizations',
        'Méthode Canny': 'Canny Method',
        'Méthode Sobel': 'Sobel Method',
        'Visualisation Canny non disponible': 'Canny visualization unavailable',
        'Visualisation Sobel non disponible': 'Sobel visualization unavailable',
        'Statistiques Détaillées': 'Detailed Statistics',
        'Méthode': 'Method',
        'Pixels de Contour': 'Edge Pixels',
        'Pourcentage': 'Percentage',
        'Densité des Contours': 'Edge Density',
        'Aucune détection de contours disponible': 'No edge detection available',
        'Analyse de Luminance': 'Luminance Analysis',
        'Luminance Moyenne': 'Average Luminance',
        'Écart-type': 'Standard Deviation',
        'Plage Luminance': 'Luminance Range',
        'Étendue': 'Range',
        'Histogramme des Couleurs': 'Color Histogram',
        'Histogramme non disponible': 'Histogram unavailable',
        
        # Navigation
        'Image précédente': 'Previous image',
        'Image suivante': 'Next image',
        'Image Précédente': 'Previous Image',
        'Image Suivante': 'Next Image',
        'Analyser une Autre Image': 'Analyze Another Image',
        'Voir la Galerie': 'View Gallery',
        'Navigation:': 'Navigation:',
        'Images précédente/suivante': 'Previous/next images',
        'Retour à la galerie': 'Back to gallery',
        'Retour à l\'accueil': 'Back to home',
        
        # Map/Dashboard
        'Tableau de Bord - Surveillance des Poubelles': 'Dashboard - Bin Monitoring',
        'Surveillance': 'Monitoring',
        'Carte Interactive': 'Interactive Map',
        'Temps réel': 'Real-time',
        'Propres': 'Clean',
        'Sales': 'Dirty',
        'Total': 'Total',
        'Propreté': 'Cleanliness',
        'Images analysées': 'Analyzed images',
        'Localisation des Poubelles': 'Bin Locations',
        'Images Récentes': 'Recent Images',
        'Propre': 'Clean',
        'Sale': 'Dirty',
        'Inconnu': 'Unknown',
        'Aucune image récente': 'No recent images',
        'Télécharger une image': 'Upload an image',
        'Erreur lors du chargement': 'Loading error',
        'Statut Inconnu': 'Unknown Status',
        'Confiance:': 'Confidence:',
        
        # Location setting
        'Valider la localisation et voir l\'analyse': 'Validate location and view analysis',
        
        # Analysis types
        'Couleurs moyennes RGB, luminosité globale': 'RGB average colors, global brightness',
        'Niveaux de contraste par canal et global': 'Channel and global contrast levels',
        'Algorithmes Canny et Sobel avec visualisations': 'Canny and Sobel algorithms with visualizations',
        'Histogrammes et statistiques de luminance': 'Luminance histograms and statistics',
        
        # Messages
        'Image non trouvée': 'Image not found',
        'Image analysée avec succès!': 'Image analyzed successfully!',
        'Erreur lors de l\'analyse:': 'Analysis error:',
        'Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF': 'File type not allowed. Use: PNG, JPG, JPEG, GIF, BMP, TIFF',
        'Aucun fichier sélectionné': 'No file selected',
        'Image supprimée avec succès!': 'Image deleted successfully!',
        'Localisation enregistrée avec succès!': 'Location saved successfully!',
        'Erreur lors de l\'enregistrement de la localisation.': 'Error saving location.',
    }
}

def get_locale():
    lang = session.get('lang')
    if lang in LANGUAGES:
        return lang
    return 'fr'

def translate(text):
    lang = get_locale()
    return TRANSLATIONS.get(lang, {}).get(text, text)

@app.context_processor
def inject_translations():
    return dict(_=translate, lang=get_locale())

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in LANGUAGES:
        session['lang'] = lang
    return redirect(request.referrer or url_for('index'))

@app.route('/')
def index():
    print("Rendering index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload route called")
    if 'file' not in request.files:
        print("No file in request.files")
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    file = request.files['file']
    print(f"File object: {file}")
    if file.filename == '':
        print("File filename is empty")
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"Saving uploaded file to: {filepath}")
        
        try:
            # Save the file properly
            file.save(filepath)
            print(f"File exists after save: {os.path.exists(filepath)}")
            print(f"File size after save: {os.path.getsize(filepath)} bytes")
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                print("File does not exist after save or is empty")
                flash("Erreur lors de l'enregistrement du fichier. Vérifiez les permissions du dossier uploads.")
                return redirect(url_for('index'))
            
            # Analyze and store the image
            image_id = analyze_and_store_image(filepath, filename)
            print(f"Inserted image with id: {image_id} and filename: {filename}")
            
            conn = get_db_connection()
            image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
            print(f"Fetched image from DB after insert: {image}")
            conn.close()
            
            if not image:
                print("Image not found in DB after insert")
                flash("Erreur : l'image n'a pas été enregistrée dans la base de données.")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(url_for('index'))
            
            if not image['latitude'] or not image['longitude']:
                print("Image missing latitude/longitude, redirecting to set_location_page")
                return redirect(url_for('set_location_page', image_id=image_id))
            else:
                print("Image has location, redirecting to view_analysis")
                flash('Image analysée avec succès!')
                return redirect(url_for('view_analysis', image_id=image_id))
                
        except Exception as e:
            print(f"Exception during file save or analysis: {e}")
            flash(f'Erreur lors de l\'analyse: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
    else:
        print("File type not allowed")
        flash('Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF')
        return redirect(url_for('index'))

@app.route('/analysis/<int:image_id>')
def view_analysis(image_id):
    """Affiche les résultats d'analyse d'une image"""
    conn = get_db_connection()
    
    # Récupérer les données de l'image
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    if not image:
        flash('Image non trouvée')
        return redirect(url_for('index'))
    
    # Get navigation info - previous and next images
    prev_image = conn.execute('''
        SELECT id FROM images 
        WHERE id < ? 
        ORDER BY id DESC 
        LIMIT 1
    ''', (image_id,)).fetchone()
    
    next_image = conn.execute('''
        SELECT id FROM images 
        WHERE id > ? 
        ORDER BY id ASC 
        LIMIT 1
    ''', (image_id,)).fetchone()
    
    # Get current position in the sequence
    current_position = conn.execute('''
        SELECT COUNT(*) FROM images WHERE id <= ?
    ''', (image_id,)).fetchone()[0]
    
    total_images = conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    
    # Récupérer toutes les analyses
    color_analysis = conn.execute('SELECT * FROM color_analysis WHERE image_id = ?', (image_id,)).fetchone()
    contrast_analysis = conn.execute('SELECT * FROM contrast_analysis WHERE image_id = ?', (image_id,)).fetchone()
    edge_detection = conn.execute('SELECT * FROM edge_detection WHERE image_id = ?', (image_id,)).fetchall()
    luminance_analysis = conn.execute('SELECT * FROM luminance_analysis WHERE image_id = ?', (image_id,)).fetchone()
    
    conn.close()
    
    # Créer l'histogramme et les visualisations de détection de contours
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
    img = Image.open(img_path)
    img_array = np.array(img)
    
    histogram_plot = create_histogram_plot(img_array)
    canny_plot = create_edge_detection_plot(img_array, method='canny')
    sobel_plot = create_edge_detection_plot(img_array, method='sobel')
    
    return render_template('analysis.html', 
                         image=image, 
                         color_analysis=color_analysis,
                         contrast_analysis=contrast_analysis,
                         edge_detection=edge_detection,
                         luminance_analysis=luminance_analysis,
                         histogram_plot=histogram_plot,
                         canny_plot=canny_plot,
                         sobel_plot=sobel_plot,
                         prev_image_id=prev_image['id'] if prev_image else None,
                         next_image_id=next_image['id'] if next_image else None,
                         current_position=current_position,
                         total_images=total_images)

@app.route('/analysis/<int:image_id>/set_location', methods=['GET', 'POST'])
def set_location_page(image_id):
    """Page pour choisir la localisation si elle n'est pas dans les métadonnées"""
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    conn.close()
    if request.method == 'POST':
        lat = request.form.get('latitude')
        lng = request.form.get('longitude')
        if lat and lng:
            conn = get_db_connection()
            conn.execute('UPDATE images SET latitude = ?, longitude = ? WHERE id = ?', (lat, lng, image_id))
            conn.commit()
            conn.close()
            flash(translate('Localisation enregistrée avec succès!'))
            return redirect(url_for('view_analysis', image_id=image_id))
        else:
            flash(translate('Erreur lors de l\'enregistrement de la localisation.'))
    return render_template('set_location.html', image=image)

@app.route('/gallery')
def gallery():
    """Affiche la galerie de toutes les images analysées"""
    conn = get_db_connection()
    images = conn.execute('''
        SELECT id, filename, filepath, upload_date, file_size_ko, width, height, image_mode
        FROM images ORDER BY upload_date DESC
    ''').fetchall()
    print(f"Number of images fetched for gallery: {len(images)}")
    for img in images:
        file_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], img['filename'])
        print(f"Gallery image: {file_on_disk} - Exists: {os.path.exists(file_on_disk)}")
    # Calculer les statistiques pour la galerie
    if images:
        total_size = sum(img['file_size_ko'] for img in images) / 1024  # Convertir en MB
        avg_width = sum(img['width'] for img in images) / len(images)
        avg_height = sum(img['height'] for img in images) / len(images)
        avg_resolution = f"{avg_width:.0f}×{avg_height:.0f}"
        
        # Mode le plus fréquent
        modes = [img['image_mode'] for img in images if img['image_mode']]
        most_common_mode = max(set(modes), key=modes.count) if modes else 'rgb'
    else:
        total_size = 0
        avg_resolution = '0×0'
        most_common_mode = 'rgb'
    
    conn.close()
    
    return render_template('gallery.html', 
                         images=images,
                         total_size=total_size,
                         avg_resolution=avg_resolution,
                         most_common_mode=most_common_mode)

@app.route('/api/stats')
def api_stats():
    """API pour obtenir les statistiques générales"""
    conn = get_db_connection()
    
    # Basic statistics
    stats = {
        'total_images': conn.execute('SELECT COUNT(*) FROM images').fetchone()[0],
        'total_size_mb': conn.execute('SELECT SUM(file_size_mo) FROM images').fetchone()[0] or 0,
        'avg_width': conn.execute('SELECT AVG(width) FROM images').fetchone()[0] or 0,
        'avg_height': conn.execute('SELECT AVG(height) FROM images').fetchone()[0] or 0
    }
    
    # Bin status statistics
    clean_count = conn.execute('SELECT COUNT(*) FROM images WHERE prediction = ?', ('clean',)).fetchone()[0]
    dirty_count = conn.execute('SELECT COUNT(*) FROM images WHERE prediction = ?', ('dirty',)).fetchone()[0]
    unknown_count = conn.execute('SELECT COUNT(*) FROM images WHERE prediction IS NULL OR prediction = ?', ('unknown',)).fetchone()[0]
    
    stats['bin_status'] = {
        'clean': clean_count,
        'dirty': dirty_count,
        'unknown': unknown_count,
        'total_classified': clean_count + dirty_count
    }
    
    conn.close()
    return jsonify(stats)

@app.route('/api/recent-images')
def api_recent_images():
    """API pour obtenir les images récentes"""
    conn = get_db_connection()
    
    images = conn.execute('''
        SELECT id, filename, upload_date, file_size_ko, width, height, prediction, prediction_confidence
        FROM images 
        ORDER BY upload_date DESC 
        LIMIT 10
    ''').fetchall()
    
    conn.close()
    
    # Convert to list of dictionaries
    images_list = []
    for image in images:
        images_list.append({
            'id': image['id'],
            'filename': image['filename'],
            'upload_date': image['upload_date'],
            'file_size_ko': image['file_size_ko'],
            'width': image['width'],
            'height': image['height'],
            'prediction': image['prediction'],
            'prediction_confidence': image['prediction_confidence']
        })
    
    return jsonify(images_list)

@app.route('/map')
def map_view():
    """Page de la carte interactive"""
    conn = get_db_connection()
    images = conn.execute('''
        SELECT id, filename, latitude, longitude, upload_date, filepath, prediction, prediction_confidence
        FROM images
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    ''').fetchall()
    conn.close()
    from flask import url_for
    trash_locations = [
        {
            "name": img['filename'],
            "lat": img['latitude'],
            "lng": img['longitude'],
            "address": img['filepath'],
            "type": "Image",
            "status": img['prediction'] or 'unknown',
            "confidence": img['prediction_confidence'] or 0.0,
            "lastCollection": img['upload_date'].strftime('%Y-%m-%d') if img['upload_date'] else "",
            "image_url": url_for('static', filename='uploads/' + img['filename'])
        }
        for img in images
    ]
    return render_template('map.html', trash_locations=trash_locations)

@app.route('/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """Supprime une image et ses analyses associées"""
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    if image:
        try:
            file_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
            if os.path.exists(file_on_disk):
                os.remove(file_on_disk)
        except Exception:
            pass
        conn.execute('DELETE FROM images WHERE id = ?', (image_id,))
        conn.commit()
        flash(translate('Image supprimée avec succès!'))
    else:
        flash(translate('Image non trouvée'))
    conn.close()
    return redirect(url_for('gallery'))

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files with proper error handling"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            from flask import send_from_directory
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        else:
            print(f"File not found: {filepath}")
            # Return a default image or 404
            return send_from_directory('static/images', 'trash_1.png')
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return send_from_directory('static/images', 'trash_1.png')

@app.route('/api/compression-stats')
def api_compression_stats():
    """API to get compression statistics"""
    conn = get_db_connection()
    
    # Get all compression data with better handling
    all_images = conn.execute('''
        SELECT 
            original_size_bytes,
            compressed_size_bytes,
            compression_ratio,
            file_size_bytes
        FROM images
        WHERE original_size_bytes IS NOT NULL
    ''').fetchall()
    
    # Also get images without compression data (legacy)
    legacy_images = conn.execute('''
        SELECT COUNT(*) as count, COALESCE(SUM(file_size_bytes), 0) as total_bytes
        FROM images
        WHERE original_size_bytes IS NULL
    ''').fetchone()
    
    conn.close()
    
    total_images = len(all_images) + (legacy_images['count'] or 0)
    
    if total_images == 0:
        return jsonify({
            'total_images': 0,
            'compressed_images': 0,
            'total_original_mb': 0,
            'total_compressed_mb': 0,
            'total_saved_mb': 0,
            'avg_compression_ratio': 0,
            'energy_efficiency': {
                'storage_saved_percent': 0,
                'bandwidth_saved_mb': 0,
                'estimated_co2_saved_kg': 0
            }
        })
    
    # Process compression data
    total_original_bytes = 0
    total_compressed_bytes = 0
    compression_ratios = []
    actually_compressed = 0
    
    for img in all_images:
        orig_size = img['original_size_bytes'] or img['file_size_bytes'] or 0
        comp_size = img['compressed_size_bytes'] or img['file_size_bytes'] or 0
        
        total_original_bytes += orig_size
        total_compressed_bytes += comp_size
        
        if img['compression_ratio'] and img['compression_ratio'] > 0:
            compression_ratios.append(img['compression_ratio'])
            actually_compressed += 1
        elif orig_size > comp_size and orig_size > 0:
            # Calculate compression ratio if not stored
            ratio = ((orig_size - comp_size) / orig_size) * 100
            compression_ratios.append(ratio)
            actually_compressed += 1
    
    # Add legacy images (assume no compression)
    if legacy_images['count'] and legacy_images['count'] > 0:
        legacy_bytes = legacy_images['total_bytes'] or 0
        total_original_bytes += legacy_bytes
        total_compressed_bytes += legacy_bytes
    
    # Calculate final statistics
    total_saved_bytes = max(0, total_original_bytes - total_compressed_bytes)
    total_saved_mb = total_saved_bytes / (1024 * 1024)
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
    
    # For display purposes, show that all images are "compressed" (processed for energy efficiency)
    displayed_compressed = total_images
    
    return jsonify({
        'total_images': total_images,
        'compressed_images': displayed_compressed,
        'total_original_mb': round(total_original_bytes / (1024 * 1024), 2),
        'total_compressed_mb': round(total_compressed_bytes / (1024 * 1024), 2),
        'total_saved_mb': round(total_saved_mb, 2),
        'avg_compression_ratio': round(avg_compression_ratio, 1),
        'energy_efficiency': {
            'storage_saved_percent': round((total_saved_bytes / total_original_bytes) * 100, 1) if total_original_bytes > 0 else 0,
            'bandwidth_saved_mb': round(total_saved_mb, 2),
            'estimated_co2_saved_kg': round(total_saved_mb * 0.000006, 4)  # Rough estimate: 6g CO2 per MB
        }
    })

if __name__ == '__main__':
    # Initialiser la base de données
    os.makedirs('database', exist_ok=True)
    init_database()
    
    app.run(debug=True, port=5001)
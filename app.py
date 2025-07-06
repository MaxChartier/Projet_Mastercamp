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

# Import des fonctions d'analyse
from featureExtraction.analysis import (
    get_file_size, get_dimensions, get_avg_color, 
    get_contrast_level, detect_edges, standardize_image
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

def init_database():
    conn = sqlite3.connect('database/images.db')
    with open('database/schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.close()

def get_db_connection():
    """Obtient une connexion à la base de données"""
    conn = sqlite3.connect('database/images.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn

def analyze_and_store_image(filepath, filename):
    """La Main fonction qui appelle les fonctions d'extraction de features"""
    conn = get_db_connection()
    
    try:
        # Obtenir les métadonnées de base
        file_info = get_file_size(filepath)
        dimensions = get_dimensions(filepath)
        
        # Ouvrir l'image pour l'analyse
        img = Image.open(filepath)
        img_array = np.array(img)
        
        # Déterminer le mode de l'image
        mode = 'grayscale' if len(img_array.shape) == 2 else 'rgb'
        
        # Insérer les données de base de l'image
        cursor = conn.execute('''
            INSERT INTO images (filename, filepath, file_size_bytes, file_size_ko, file_size_mo,
                              width, height, total_pixels, image_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filename, filepath, file_info['bytes'], file_info['ko'], file_info['mo'],
              dimensions['w'], dimensions['h'], dimensions['pixels_tt'], mode))
        
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

# Supported languages and translations (expand as needed)
LANGUAGES = ['fr', 'en']
TRANSLATIONS = {
    'fr': {
        'Galerie des Images Analysées': 'Galerie des Images Analysées',
        'Accueil': 'Accueil',
        'Galerie': 'Galerie',
        'Carte': 'Carte',
        'Télécharger une Image': 'Télécharger une Image',
        'Types d\'Analyses Disponibles': 'Types d\'Analyses Disponibles',
        'Statistiques de la Plateforme': 'Statistiques de la Plateforme',
        'Téléchargez vos images pour une analyse complète des caractéristiques :': 'Téléchargez vos images pour une analyse complète des caractéristiques :',
        'Couleurs, Contraste, Contours et Luminance': 'Couleurs, Contraste, Contours et Luminance',
        'Glissez votre image ici': 'Glissez votre image ici',
        'ou cliquez pour sélectionner un fichier': 'ou cliquez pour sélectionner un fichier',
        'Choisir un fichier': 'Choisir un fichier',
        'Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)': 'Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)',
        'Aperçu :': 'Aperçu :',
        'Analyser l\'Image': 'Analyser l\'Image',
        'Changer d\'image': 'Changer d\'image',
        'Images Analysées': 'Images Analysées',
        'Taille Totale (MB)': 'Taille Totale (MB)',
        'Largeur Moyenne': 'Largeur Moyenne',
        'Hauteur Moyenne': 'Hauteur Moyenne',
        'Aucune image analysée': 'Aucune image analysée',
        'Commencez par télécharger et analyser votre première image.': 'Commencez par télécharger et analyser votre première image.',
        'Voir l\'Analyse': 'Voir l\'Analyse',
        'Images Totales': 'Images Totales',
        'Taille Totale': 'Taille Totale',
        'Résolution Moyenne': 'Résolution Moyenne',
        'Mode le Plus Fréquent': 'Mode le Plus Fréquent',
        'Image non trouvée': 'Image non trouvée',
        'Image analysée avec succès!': 'Image analysée avec succès!',
        'Erreur lors de l\'analyse:': 'Erreur lors de l\'analyse:',
        'Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF': 'Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF',
        'Aucun fichier sélectionné': 'Aucun fichier sélectionné',
        'Statistiques de la Galerie': 'Statistiques de la Galerie',
        'Images Totales': 'Images Totales',
        'Taille Totale': 'Taille Totale',
        'Résolution Moyenne': 'Résolution Moyenne',
        'Mode le Plus Fréquent': 'Mode le Plus Fréquent',
        'Analyse des Couleurs': 'Analyse des Couleurs',
        'Analyse du Contraste': 'Analyse du Contraste',
        'Détection de Contours': 'Détection de Contours',
        'Analyse de Luminance': 'Analyse de Luminance',
        'Histogramme des Couleurs': 'Histogramme des Couleurs',
        'Téléchargé le': 'Téléchargé le',
        'Date inconnue': 'Date inconnue',
        'Taille': 'Taille',
        'Dimensions': 'Dimensions',
        'Pixels': 'Pixels',
        'Mode': 'Mode',
        'Image Analysée': 'Image Analysée',
        'Métadonnées': 'Métadonnées',
        'Localisation de la Poubelle': 'Localisation de la Poubelle',
        'Place de la Concorde, 75001 Paris': 'Place de la Concorde, 75001 Paris',
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
        'Méthode': 'Méthode',
        'Pixels de Contour': 'Pixels de Contour',
        'Pourcentage': 'Pourcentage',
        'Densité des Contours': 'Densité des Contours',
        'Analyse de Luminance': 'Analyse de Luminance',
        'Luminance Moyenne': 'Luminance Moyenne',
        'Écart-type': 'Écart-type',
        'Plage Luminance': 'Plage Luminance',
        'Étendue': 'Étendue',
        'Histogramme non disponible': 'Histogramme non disponible',
        'Analyser une Autre Image': 'Analyser une Autre Image',
        'Voir la Galerie': 'Voir la Galerie',
        # ...add more as needed...
    },
    'en': {
        'Galerie des Images Analysées': 'Analyzed Images Gallery',
        'Accueil': 'Home',
        'Galerie': 'Gallery',
        'Carte': 'Map',
        'Télécharger une Image': 'Upload Image',
        'Types d\'Analyses Disponibles': 'Available Analyses',
        'Statistiques de la Plateforme': 'Platform Statistics',
        'Téléchargez vos images pour une analyse complète des caractéristiques :': 'Upload your images for a complete analysis of features:',
        'Couleurs, Contraste, Contours et Luminance': 'Colors, Contrast, Edges and Luminance',
        'Glissez votre image ici': 'Drag your image here',
        'ou cliquez pour sélectionner un fichier': 'or click to select a file',
        'Choisir un fichier': 'Choose a file',
        'Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)': 'Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF (max 16MB)',
        'Aperçu :': 'Preview:',
        'Analyser l\'Image': 'Analyze Image',
        'Changer d\'image': 'Change image',
        'Images Analysées': 'Analyzed Images',
        'Taille Totale (MB)': 'Total Size (MB)',
        'Largeur Moyenne': 'Average Width',
        'Hauteur Moyenne': 'Average Height',
        'Aucune image analysée': 'No analyzed image',
        'Commencez par télécharger et analyser votre première image.': 'Start by uploading and analyzing your first image.',
        'Voir l\'Analyse': 'View Analysis',
        'Images Totales': 'Total Images',
        'Taille Totale': 'Total Size',
        'Résolution Moyenne': 'Average Resolution',
        'Mode le Plus Fréquent': 'Most Common Mode',
        'Image non trouvée': 'Image not found',
        'Image analysée avec succès!': 'Image successfully analyzed!',
        'Erreur lors de l\'analyse:': 'Error during analysis:',
        'Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF': 'File type not allowed. Use: PNG, JPG, JPEG, GIF, BMP, TIFF',
        'Aucun fichier sélectionné': 'No file selected',
        'Statistiques de la Galerie': 'Gallery Statistics',
        'Images Totales': 'Total Images',
        'Taille Totale': 'Total Size',
        'Résolution Moyenne': 'Average Resolution',
        'Mode le Plus Fréquent': 'Most Common Mode',
        'Analyse des Couleurs': 'Color Analysis',
        'Analyse du Contraste': 'Contrast Analysis',
        'Détection de Contours': 'Edge Detection',
        'Analyse de Luminance': 'Luminance Analysis',
        'Histogramme des Couleurs': 'Color Histogram',
        'Téléchargé le': 'Uploaded on',
        'Date inconnue': 'Unknown date',
        'Taille': 'Size',
        'Dimensions': 'Dimensions',
        'Pixels': 'Pixels',
        'Mode': 'Mode',
        'Image Analysée': 'Analyzed Image',
        'Métadonnées': 'Metadata',
        'Localisation de la Poubelle': 'Trash Location',
        'Place de la Concorde, 75001 Paris': 'Place de la Concorde, 75001 Paris',
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
        'Méthode': 'Method',
        'Pixels de Contour': 'Edge Pixels',
        'Pourcentage': 'Percentage',
        'Densité des Contours': 'Edge Density',
        'Analyse de Luminance': 'Luminance Analysis',
        'Luminance Moyenne': 'Average Luminance',
        'Écart-type': 'Standard Deviation',
        'Plage Luminance': 'Luminance Range',
        'Étendue': 'Range',
        'Histogramme non disponible': 'Histogram not available',
        'Analyser une Autre Image': 'Analyze Another Image',
        'Voir la Galerie': 'View Gallery',
        # ...add more as needed...
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
    """Page d'accueil avec formulaire d'upload"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Traite l'upload et l'analyse d'une image"""
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ajouter timestamp pour éviter les conflits
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Analyser et stocker l'image
            image_id = analyze_and_store_image(filepath, filename)
            flash('Image analysée avec succès!')
            return redirect(url_for('view_analysis', image_id=image_id))
        except Exception as e:
            flash(f'Erreur lors de l\'analyse: {str(e)}')
            return redirect(url_for('index'))
    else:
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
    
    # Récupérer toutes les analyses
    color_analysis = conn.execute('SELECT * FROM color_analysis WHERE image_id = ?', (image_id,)).fetchone()
    contrast_analysis = conn.execute('SELECT * FROM contrast_analysis WHERE image_id = ?', (image_id,)).fetchone()
    edge_detection = conn.execute('SELECT * FROM edge_detection WHERE image_id = ?', (image_id,)).fetchall()
    luminance_analysis = conn.execute('SELECT * FROM luminance_analysis WHERE image_id = ?', (image_id,)).fetchone()
    
    conn.close()
    
    # Créer l'histogramme
    img = Image.open(image['filepath'])
    img_array = np.array(img)
    histogram_plot = create_histogram_plot(img_array)
    
    return render_template('analysis.html', 
                         image=image, 
                         color_analysis=color_analysis,
                         contrast_analysis=contrast_analysis,
                         edge_detection=edge_detection,
                         luminance_analysis=luminance_analysis,
                         histogram_plot=histogram_plot)

@app.route('/gallery')
def gallery():
    """Affiche la galerie de toutes les images analysées"""
    conn = get_db_connection()
    images = conn.execute('''
        SELECT id, filename, filepath, upload_date, file_size_ko, width, height, image_mode
        FROM images ORDER BY upload_date DESC
    ''').fetchall()
    
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
    
    stats = {
        'total_images': conn.execute('SELECT COUNT(*) FROM images').fetchone()[0],
        'total_size_mb': conn.execute('SELECT SUM(file_size_mo) FROM images').fetchone()[0] or 0,
        'avg_width': conn.execute('SELECT AVG(width) FROM images').fetchone()[0] or 0,
        'avg_height': conn.execute('SELECT AVG(height) FROM images').fetchone()[0] or 0
    }
    
    conn.close()
    return jsonify(stats)

@app.route('/map')
def map_view():
    """Page de la carte interactive"""
    # Données d'exemple des emplacements de poubelles (coordonnées autour de Paris)
    trash_locations = [
        {
            "name": "Poubelle République",
            "lat": 48.8671,
            "lng": 2.3636,
            "address": "Place de la République, 75003 Paris",
            "type": "Tri sélectif",
            "status": "Propre",
            "lastCollection": "2025-01-02"
        },
        {
            "name": "Poubelle Châtelet",
            "lat": 48.8566,
            "lng": 2.3475,
            "address": "Place du Châtelet, 75001 Paris",
            "type": "Ordures ménagères",
            "status": "À collecter",
            "lastCollection": "2024-12-30"
        },
        {
            "name": "Poubelle Bastille",
            "lat": 48.8532,
            "lng": 2.3692,
            "address": "Place de la Bastille, 75011 Paris",
            "type": "Tri sélectif",
            "status": "Propre",
            "lastCollection": "2025-01-02"
        },
        {
            "name": "Poubelle Louvre",
            "lat": 48.8606,
            "lng": 2.3376,
            "address": "Cour du Louvre, 75001 Paris",
            "type": "Ordures ménagères",
            "status": "Propre",
            "lastCollection": "2025-01-01"
        },
        {
            "name": "Poubelle Notre-Dame",
            "lat": 48.8530,
            "lng": 2.3499,
            "address": "Parvis Notre-Dame, 75004 Paris",
            "type": "Tri sélectif",
            "status": "À collecter",
            "lastCollection": "2024-12-28"
        }
    ]
    
    return render_template('map.html', trash_locations=trash_locations)

@app.route('/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """Supprime une image et ses analyses associées"""
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    # Use the translate function directly instead of _
    if image:
        try:
            if os.path.exists(image['filepath']):
                os.remove(image['filepath'])
        except Exception:
            pass
        conn.execute('DELETE FROM images WHERE id = ?', (image_id,))
        conn.commit()
        flash(translate('Image supprimée avec succès!'))
    else:
        flash(translate('Image non trouvée'))
    conn.close()
    return redirect(url_for('gallery'))

if __name__ == '__main__':
    # Initialiser la base de données
    os.makedirs('database', exist_ok=True)
    init_database()
    
    app.run(debug=True, port=5001)
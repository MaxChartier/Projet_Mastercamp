# Projet_Mastercamp

Tran Anthony, Musquin Maxime, Sancesario Tom, Fils-de-Ahmed Ilyes, Chartier Max

## Description

Cette interface web Flask permet de télécharger et analyser des images pour extraire diverses caractéristiques des images de poubelles, et utilise un modèle YOLO pour classifier automatiquement si la poubelle est propre ou sale :

- **Classification IA** : Modèle YOLO fine-tuné pour détecter l'état des poubelles
- **Analyse des couleurs** : Moyennes RGB, luminosité
- **Analyse du contraste** : Niveaux par canal et global
- **Détection de contours** : Algorithmes Canny et Sobel
- **Analyse de luminance** : Histogrammes et statistiques
- **Métadonnées** : Taille, dimensions, format

## Installation

1. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

2. **S'assurer que le modèle YOLO est présent :**
   - Placez votre modèle `yolo11clsFineTuned.pt` dans le dossier `machinelearning/`

3. **Lancer l'application :**
```bash
python app.py
```

4. **Accéder à l'interface :**
Ouvrez votre navigateur et allez sur `http://127.0.0.1:5001`

## Fonctionnalités

### Interface Multilingue
- **Support français/anglais** : Basculez entre les langues avec les drapeaux
- **Traduction complète** : Toute l'interface est traduite
- **Persistance** : La langue choisie est conservée dans la session

### Page d'Accueil
- **Upload drag & drop** : Glissez-déposez vos images
- **Formulaire classique** : Cliquez pour sélectionner un fichier
- **Aperçu temps réel** : Prévisualisez avant l'analyse
- **Statistiques globales** : Métriques de la plateforme

### Page d'Analyse
- **Classification IA** : Détection automatique de l'état des poubelles
- **Visualisation complète** : Toutes les métriques extraites
- **Histogrammes interactifs** : Graphiques générés automatiquement
- **Navigation intuitive** : Liens vers galerie et nouvelle analyse
- **Carte interactive** : Localisation géographique des poubelles

### Galerie
- **Vue grille** : Aperçu de toutes les images
- **Recherche visuelle** : Navigation rapide
- **Statistiques de collection** : Métriques aggregées
- **Gestion des images** : Suppression et navigation

## Base de Données

L'application utilise SQLite avec le schéma suivant :

- `images` : Métadonnées principales
- `color_analysis` : Analyse des couleurs
- `contrast_analysis` : Analyse du contraste  
- `edge_detection` : Détection de contours
- `luminance_analysis` : Analyse de luminance

## Formats Supportés

- PNG, JPG, JPEG, GIF, BMP, TIFF
- Taille maximale : 16MB
- Images couleur et niveaux de gris

## Architecture

```
app.py              # Application Flask principale
templates/          # Templates HTML
├── base.html      # Template de base
├── index.html     # Page d'accueil
├── analysis.html  # Page de résultats
└── gallery.html   # Galerie d'images
static/
└── uploads/       # Images téléchargées
database/
├── schema.sql     # Schéma de base de données
└── images.db      # Base SQLite (créée automatiquement)
featureExtraction/
└── analysis.py    # Fonctions d'extraction
```

## API

### Endpoints disponibles

- `GET /` : Page d'accueil
- `POST /upload` : Upload et analyse d'image
- `GET /analysis/<id>` : Résultats d'analyse
- `GET /gallery` : Galerie d'images
- `GET /api/stats` : Statistiques JSON

## Développement

Pour modifier l'interface :

1. **CSS** : Styles dans `templates/base.html`
2. **JavaScript** : Scripts dans chaque template
3. **Analyses** : Fonctions dans `featureExtraction/analysis.py`
4. **Routes** : Endpoints dans `app.py`

## Troubleshooting

### Erreurs communes

- **Port occupé** : Changez le port dans `app.run(port=5001)`
- **Permissions** : Vérifiez les droits d'écriture dans `static/uploads/`
- **Dépendances** : Réinstallez avec `pip install -r requirements.txt --force-reinstall`

### Logs

L'application affiche les erreurs dans la console. Pour plus de détails :
```python
app.run(debug=True)
```

## Contribution

1. Fork le projet
2. Créez une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Créez une Pull Request
   
### User Experience Previezs 

## Home Page

<img width="1905" height="869" alt="Screenshot 2025-08-12 at 11 55 46 AM" src="https://github.com/user-attachments/assets/5436f27f-d9b6-4aff-b8d9-86a3a1a34744" />

## Details & Upload Feature 

<img width="402" height="763" alt="Screenshot 2025-08-12 at 11 56 15 AM" src="https://github.com/user-attachments/assets/cad64e33-50ac-449e-b839-65110764ce97" />

## Glassmorphism Navigation Bar

<img width="1029" height="74" alt="Screenshot 2025-08-12 at 11 56 01 AM" src="https://github.com/user-attachments/assets/1cd6f48a-440d-42e9-893b-3b561b530fa9" />

## Gallery of Uploads 

<img width="1915" height="864" alt="Screenshot 2025-08-12 at 11 56 26 AM" src="https://github.com/user-attachments/assets/1272a5cd-1a3f-4d7a-8ef2-5eb04c08e57b" />

## Details & Analysis

<img width="1904" height="869" alt="Screenshot 2025-08-12 at 11 56 38 AM" src="https://github.com/user-attachments/assets/7258833f-eec0-486c-a4f1-425e21b48a5a" />

##
<img width="649" height="865" alt="Screenshot 2025-08-12 at 11 56 54 AM" src="https://github.com/user-attachments/assets/5e86b329-399c-41e4-ab64-c12f9717e3cd" />

## Trash Maps (Global Visualization 

<img width="1060" height="776" alt="Screenshot 2025-08-12 at 11 57 08 AM" src="https://github.com/user-attachments/assets/948fde23-2e53-4b7e-a9cc-7ff3ae064309" />

## Mobile Version

<img width="377" height="798" alt="Screenshot 2025-08-12 at 12 01 08 PM" src="https://github.com/user-attachments/assets/794438ff-f9ce-4a57-b377-4748eb02578d" />

## Language Switch EN/FR

<img width="373" height="802" alt="Screenshot 2025-08-12 at 12 02 10 PM" src="https://github.com/user-attachments/assets/9f24287d-0bdc-43d7-aecf-dcadf7ecec02" />




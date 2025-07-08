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

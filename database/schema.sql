CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size_bytes INTEGER,
    file_size_ko REAL,
    file_size_mo REAL,
    width INTEGER,
    height INTEGER,
    total_pixels INTEGER,
    image_mode TEXT,
    latitude REAL,
    longitude REAL,
    prediction TEXT,
    prediction_confidence REAL,
    original_size_bytes INTEGER DEFAULT 0,
    compressed_size_bytes INTEGER DEFAULT 0,
    compression_ratio REAL DEFAULT 0,
    has_thumbnail BOOLEAN DEFAULT FALSE,
    thumbnail_path TEXT
);

CREATE TABLE IF NOT EXISTS color_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    avg_red REAL,
    avg_green REAL,
    avg_blue REAL,
    avg_gray REAL,
    brightness REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS contrast_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    mode VARCHAR(20),
    min_intensity REAL,
    max_intensity REAL,
    contrast_level REAL,
    contrast_ratio REAL,
    red_contrast REAL,
    green_contrast REAL,
    blue_contrast REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edge_detection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    method VARCHAR(20), -- canny ou sobel
    total_pixels INTEGER,
    edge_pixels INTEGER,
    edge_density REAL,
    edge_percentage REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS luminance_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    mean_luminance REAL,
    std_luminance REAL,
    min_luminance REAL,
    max_luminance REAL,
    luminance_range REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

-- Index pour les performances
CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);
CREATE INDEX IF NOT EXISTS idx_color_analysis_image_id ON color_analysis(image_id);
CREATE INDEX IF NOT EXISTS idx_contrast_analysis_image_id ON contrast_analysis(image_id);
CREATE INDEX IF NOT EXISTS idx_edge_detection_image_id ON edge_detection(image_id);
CREATE INDEX IF NOT EXISTS idx_luminance_analysis_image_id ON luminance_analysis(image_id);
-- Add indexes for compression queries
CREATE INDEX IF NOT EXISTS idx_images_compression ON images(compression_ratio, original_size_bytes);
CREATE INDEX IF NOT EXISTS idx_images_thumbnail ON images(has_thumbnail);

-- Add migration for existing records
UPDATE images 
SET original_size_bytes = file_size_bytes, 
    compressed_size_bytes = file_size_bytes 
WHERE original_size_bytes IS NULL OR original_size_bytes = 0;
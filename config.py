"""
Configuración general del proyecto
Sistema de Análisis de Sentimiento - Amazon Reviews
"""

import os
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Ruta base del proyecto
BASE_DIR = Path(__file__).parent

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de notebooks
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Rutas de modelos
MODELS_DIR = BASE_DIR / "models"

# Rutas de reportes
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# CONFIGURACIÓN DE DATOS
# =============================================================================

# Archivo de datos original
DATASET_FILENAME = "Reviews.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Archivos procesados
CLEAN_DATA_FILENAME = "reviews_clean.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / CLEAN_DATA_FILENAME

# =============================================================================
# PARÁMETROS DE PROCESAMIENTO
# =============================================================================

# Muestra de datos para desarrollo rápido (None para usar todo el dataset)
SAMPLE_SIZE = None  # Ejemplo: 50000 para trabajar con 50k reviews

# Conversión de scores a sentimiento binario
POSITIVE_THRESHOLD = 4  # Score >= 4 → Positivo
NEGATIVE_THRESHOLD = 2  # Score <= 2 → Negativo
# Score = 3 se elimina (neutral)

# Random seed para reproducibilidad
RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# =============================================================================
# PARÁMETROS DE NLP
# =============================================================================

# Stopwords
STOPWORDS_LANGUAGE = 'english'

# Stemming/Lemmatization
USE_LEMMATIZATION = True  # True: usar lemmatization, False: usar stemming

# N-gramas
NGRAM_RANGE = (1, 2)  # Unigramas y bigramas

# Longitud mínima de tokens
MIN_TOKEN_LENGTH = 2

# =============================================================================
# PARÁMETROS DE VECTORIZACIÓN
# =============================================================================

# TF-IDF
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.8

# Bag of Words
BOW_MAX_FEATURES = 5000

# =============================================================================
# PARÁMETROS DE MODELOS ML
# =============================================================================

# Regresión Logística
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000

# =============================================================================
# PARÁMETROS DE DEEP LEARNING
# =============================================================================

# Tokenizer
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200

# Embedding
EMBEDDING_DIM = 100

# Arquitectura
LSTM_UNITS = 128
DROPOUT_RATE = 0.5

# Training
BATCH_SIZE = 64
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

# =============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# =============================================================================

# Tamaño de figuras
FIGURE_SIZE = (12, 6)

# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8'

# Paleta de colores
COLOR_POSITIVE = '#2ecc71'  # Verde
COLOR_NEGATIVE = '#e74c3c'  # Rojo
COLOR_NEUTRAL = '#95a5a6'   # Gris

# WordCloud
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_BACKGROUND = 'white'

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

# Nivel de logging
LOG_LEVEL = 'INFO'

# Formato de logs
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def create_directories():
    """Crear todas las carpetas necesarias del proyecto"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado/verificado: {directory}")

def get_dataset_info():
    """Obtener información sobre el dataset"""
    info = {
        'path': str(DATASET_PATH),
        'exists': DATASET_PATH.exists(),
        'size_mb': DATASET_PATH.stat().st_size / (1024**2) if DATASET_PATH.exists() else 0
    }
    return info

if __name__ == "__main__":
    print("="*80)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("="*80)
    print(f"\n📁 Directorio base: {BASE_DIR}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"🔢 Random State: {RANDOM_STATE}")
    print(f"✂️ Test Size: {TEST_SIZE}")
    print(f"🎯 Positive Threshold: {POSITIVE_THRESHOLD}")
    print(f"🎯 Negative Threshold: {NEGATIVE_THRESHOLD}")
    print("\n" + "="*80)
    
    # Crear directorios
    create_directories()

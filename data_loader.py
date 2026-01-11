"""
Módulo de carga y manejo de datos
Amazon Reviews - Sentiment Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def load_raw_data(sample_size=None, random_state=RANDOM_STATE):
    """
    Cargar datos crudos del CSV
    
    Parameters:
    -----------
    sample_size : int, optional
        Número de muestras a cargar (None para cargar todo)
    random_state : int
        Semilla para reproducibilidad
    
    Returns:
    --------
    pd.DataFrame : Dataset cargado
    """
    print("📂 Cargando dataset...")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"❌ Dataset no encontrado en: {DATASET_PATH}")
    
    # Cargar dataset
    df = pd.read_csv(DATASET_PATH)
    
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Si se especifica sample_size, tomar muestra
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"📊 Muestra tomada: {sample_size:,} filas")
    
    return df


def convert_to_binary_sentiment(df, score_column='Score', 
                                positive_threshold=POSITIVE_THRESHOLD,
                                negative_threshold=NEGATIVE_THRESHOLD):
    """
    Convertir scores a sentimiento binario
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con columna de scores
    score_column : str
        Nombre de la columna con scores
    positive_threshold : int
        Score mínimo para considerar positivo
    negative_threshold : int
        Score máximo para considerar negativo
    
    Returns:
    --------
    pd.DataFrame : Dataset con nueva columna 'Sentiment'
    """
    print("\n🎯 Convirtiendo scores a sentimiento binario...")
    
    # Crear copia del dataframe
    df = df.copy()
    
    # Función de conversión
    def score_to_sentiment(score):
        if score >= positive_threshold:
            return 1  # Positivo
        elif score <= negative_threshold:
            return 0  # Negativo
        else:
            return None  # Neutral (se eliminará)
    
    # Aplicar conversión
    df['Sentiment'] = df[score_column].apply(score_to_sentiment)
    
    # Eliminar neutrales
    initial_size = len(df)
    df = df.dropna(subset=['Sentiment'])
    removed = initial_size - len(df)
    
    # Convertir a int
    df['Sentiment'] = df['Sentiment'].astype(int)
    
    print(f"✅ Conversión completada:")
    print(f"   - Reviews positivos (Score {positive_threshold}-5): {(df['Sentiment']==1).sum():,}")
    print(f"   - Reviews negativos (Score 1-{negative_threshold}): {(df['Sentiment']==0).sum():,}")
    print(f"   - Neutrales eliminados (Score 3): {removed:,}")
    print(f"   - Total final: {len(df):,}")
    
    return df


def get_basic_info(df):
    """
    Obtener información básica del dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    
    Returns:
    --------
    dict : Diccionario con información del dataset
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    return info


def print_dataset_info(df):
    """
    Imprimir información del dataset de forma legible
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    """
    info = get_basic_info(df)
    
    print("\n" + "="*80)
    print("📊 INFORMACIÓN DEL DATASET")
    print("="*80)
    print(f"\n🔢 Dimensiones: {info['shape'][0]:,} filas × {info['shape'][1]} columnas")
    print(f"💾 Memoria: {info['memory_usage_mb']:.2f} MB")
    print(f"🔄 Duplicados: {info['duplicates']:,}")
    
    print("\n📋 Columnas:")
    for col in info['columns']:
        dtype = info['dtypes'][col]
        missing = info['missing_values'][col]
        missing_pct = (missing / len(df) * 100)
        print(f"   - {col:20s} | {str(dtype):10s} | Missing: {missing:6,} ({missing_pct:5.2f}%)")
    
    print("\n" + "="*80)


def save_processed_data(df, filename=CLEAN_DATA_FILENAME):
    """
    Guardar datos procesados
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a guardar
    filename : str
        Nombre del archivo
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    print(f"\n💾 Guardando datos procesados en: {filepath}")
    df.to_csv(filepath, index=False)
    print(f"✅ Datos guardados exitosamente")


def load_processed_data(filename=CLEAN_DATA_FILENAME):
    """
    Cargar datos procesados
    
    Parameters:
    -----------
    filename : str
        Nombre del archivo
    
    Returns:
    --------
    pd.DataFrame : Dataset procesado
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Archivo no encontrado: {filepath}")
    
    print(f"📂 Cargando datos procesados desde: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Datos cargados: {df.shape[0]:,} filas")
    
    return df


# =============================================================================
# FUNCIONES DE EJEMPLO Y TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTING: Módulo de carga de datos")
    print("="*80)
    
    try:
        # Cargar muestra pequeña
        df = load_raw_data(sample_size=1000)
        
        # Mostrar información
        print_dataset_info(df)
        
        # Convertir a binario
        df = convert_to_binary_sentiment(df)
        
        # Mostrar distribución
        print("\n📊 Distribución de sentimientos:")
        print(df['Sentiment'].value_counts())
        
        print("\n✅ Testing completado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

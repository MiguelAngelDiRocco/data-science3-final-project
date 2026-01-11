"""
Script de verificación de instalación
Verifica que todas las librerías estén correctamente instaladas
"""

import sys

print("="*80)
print("🧪 VERIFICACIÓN DE INSTALACIÓN")
print("="*80)
print()

# Lista de librerías a verificar
libraries = {
    'pandas': 'Manipulación de datos',
    'numpy': 'Operaciones numéricas',
    'matplotlib': 'Visualización básica',
    'seaborn': 'Visualización avanzada',
    'sklearn': 'Machine Learning (scikit-learn)',
    'nltk': 'Natural Language Toolkit',
    'textblob': 'Análisis de sentimiento',
    'wordcloud': 'Nubes de palabras',
    'vaderSentiment': 'Sentiment Analysis',
    'tensorflow': 'Deep Learning',
    'keras': 'Deep Learning API'
}

print("📦 Verificando librerías...\n")

failed = []
success = []

for lib, description in libraries.items():
    try:
        __import__(lib)
        print(f"✅ {lib:20s} - {description}")
        success.append(lib)
    except ImportError as e:
        print(f"❌ {lib:20s} - {description} - ERROR")
        failed.append(lib)

print()
print("="*80)
print(f"✅ Instaladas correctamente: {len(success)}/{len(libraries)}")
if failed:
    print(f"❌ Con problemas: {len(failed)}")
    print(f"   Librerías faltantes: {', '.join(failed)}")
else:
    print("🎉 ¡TODAS LAS LIBRERÍAS INSTALADAS CORRECTAMENTE!")
print("="*80)
print()

# Verificar versiones de las principales
if not failed:
    print("📋 VERSIONES PRINCIPALES:")
    print("-"*80)
    import pandas as pd
    import numpy as np
    import sklearn
    import tensorflow as tf
    
    print(f"Python:      {sys.version.split()[0]}")
    print(f"Pandas:      {pd.__version__}")
    print(f"NumPy:       {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"TensorFlow:  {tf.__version__}")
    print("-"*80)
    print()

# Verificar dataset
print("📂 VERIFICACIÓN DE DATASET:")
print("-"*80)
from pathlib import Path

dataset_path = Path("data/raw/Reviews.csv")
if dataset_path.exists():
    size_mb = dataset_path.stat().st_size / (1024**2)
    print(f"✅ Dataset encontrado: {dataset_path}")
    print(f"   Tamaño: {size_mb:.2f} MB")
else:
    print(f"❌ Dataset NO encontrado en: {dataset_path}")
    print("   Por favor, coloca Reviews.csv en data/raw/")

print("-"*80)
print()

if not failed and dataset_path.exists():
    print("🎯 SIGUIENTE PASO:")
    print("   1. Abre VSCode: code .")
    print("   2. Abre: notebooks/00_Setup_and_DataLoad.ipynb")
    print("   3. Selecciona el kernel del venv")
    print("   4. Ejecuta el notebook!")
    print()
    print("🚀 ¡TODO LISTO PARA EMPEZAR!")
else:
    print("⚠️  Hay problemas que resolver antes de continuar.")
    if failed:
        print("   Intenta reinstalar: pip install -r requirements.txt")

print("="*80)

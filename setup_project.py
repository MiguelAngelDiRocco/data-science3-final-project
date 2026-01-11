"""
Script para crear la estructura completa del proyecto NLP
"""
import os

# Estructura de carpetas
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "reports/figures",
    "models",
]

# Crear estructura
base_path = "nlp-sentiment-analysis"
for folder in folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    print(f"✅ Creada: {folder}")

# Crear archivos __init__.py
init_files = [
    "src/__init__.py",
]

for init_file in init_files:
    with open(os.path.join(base_path, init_file), 'w') as f:
        f.write("# Init file\n")
    print(f"✅ Creado: {init_file}")

print("\n✅ Estructura del proyecto creada exitosamente!")

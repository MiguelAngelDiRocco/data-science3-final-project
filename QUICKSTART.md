# 🚀 GUÍA RÁPIDA DE INICIO

## 📋 Checklist Inicial

### Paso 1: Verificar que tienes todo
- [ ] Python 3.10+ instalado
- [ ] VSCode instalado
- [ ] Archivo `Reviews.csv` descargado de Kaggle
- [ ] Esta carpeta en: `C:\Users\maike\OneDrive\Escritorio\Proyecto Datascience 3`

---

## ⚙️ Instalación (Primera Vez)

### Opción A: Script Automático (RECOMENDADO)
1. Haz doble clic en `setup.bat`
2. Espera a que termine la instalación (5-10 minutos)
3. Listo!

### Opción B: Manual
Abre PowerShell o CMD en esta carpeta y ejecuta:

```bash
# Crear ambiente virtual
python -m venv venv

# Activar ambiente virtual
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar recursos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 📂 Colocar el Dataset

1. Busca el archivo `Reviews.csv` (o `Reviews` sin extensión) que descargaste
2. Cópialo a: `data\raw\Reviews.csv`

**IMPORTANTE:** El archivo debe llamarse exactamente `Reviews.csv`

---

## 🎯 Ejecutar el Proyecto

### 1. Abrir en VSCode
```bash
# Desde la carpeta del proyecto
code .
```

### 2. Activar ambiente virtual en VSCode
- Presiona `Ctrl + Shift + P`
- Escribe: "Python: Select Interpreter"
- Selecciona el que dice `venv` o `.\venv\Scripts\python.exe`

### 3. Abrir el primer notebook
- Navega a: `notebooks\00_Setup_and_DataLoad.ipynb`
- Click en "Run All" o ejecuta celda por celda

---

## 📚 Orden de los Notebooks

Ejecuta en este orden:

1. **00_Setup_and_DataLoad.ipynb**
   - Carga y exploración inicial del dataset
   - Conversión a clasificación binaria
   - ⏱️ Tiempo: 5-10 minutos

2. **01_EDA.ipynb** (próximo a crear)
   - Análisis Exploratorio de Datos completo
   - Visualizaciones
   - ⏱️ Tiempo: 15-20 minutos

3. **02_NLP_Processing.ipynb** (ETAPA 1)
   - Limpieza de texto
   - Tokenización
   - Lemmatization
   - Nubes de palabras
   - N-gramas
   - Análisis de sentimiento
   - ⏱️ Tiempo: 20-30 minutos

4. **03_ML_Models.ipynb** (ETAPA 2)
   - TF-IDF
   - Bag of Words
   - Regresión Logística
   - Naive Bayes
   - ⏱️ Tiempo: 15-25 minutos

5. **04_DL_Models.ipynb** (ETAPA 2)
   - Text to Sequence
   - Embeddings
   - LSTM
   - GRU
   - ⏱️ Tiempo: 30-60 minutos

---

## 🐛 Solución de Problemas

### Error: "Python no encontrado"
- Asegúrate de que Python esté instalado
- Verifica que Python esté en PATH
- Reinicia CMD/PowerShell después de instalar Python

### Error: "Module not found"
```bash
# Activa el ambiente virtual
venv\Scripts\activate

# Reinstala dependencias
pip install -r requirements.txt
```

### Error: "Dataset no encontrado"
- Verifica que `Reviews.csv` esté en `data\raw\`
- El archivo debe llamarse exactamente `Reviews.csv`

### VSCode no reconoce el ambiente virtual
1. Presiona `Ctrl + Shift + P`
2. "Python: Select Interpreter"
3. Selecciona `.\venv\Scripts\python.exe`
4. Recarga VSCode

---

## 💡 Consejos

### Para trabajar más rápido
Si tu PC es lenta, puedes usar una **muestra del dataset**:

En el notebook, cambia:
```python
df = load_raw_data(sample_size=None)  # Todo el dataset
```
Por:
```python
df = load_raw_data(sample_size=50000)  # Solo 50k reviews
```

### Para liberar memoria
Cierra notebooks que no estés usando activamente.

### Para guardar progreso
Los notebooks se guardan automáticamente. Los datos procesados se guardan en `data/processed/`.

---

## 📞 Recursos Adicionales

- **Documentación NLTK:** https://www.nltk.org/
- **Documentación scikit-learn:** https://scikit-learn.org/
- **Documentación TensorFlow:** https://www.tensorflow.org/

---

## ✅ Checklist de Progreso

- [ ] Setup completado
- [ ] Dataset cargado
- [ ] Notebook 00 ejecutado
- [ ] Notebook 01 (EDA) completado
- [ ] Notebook 02 (NLP) completado - ETAPA 1
- [ ] Notebook 03 (ML) completado - ETAPA 2
- [ ] Notebook 04 (DL) completado - ETAPA 2
- [ ] Reporte final escrito
- [ ] Proyecto subido a GitHub

---

**¿Listo para empezar? ¡Ejecuta `setup.bat` y luego abre el primer notebook!** 🚀

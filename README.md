# 🔤 Sistema de Análisis de Sentimiento con NLP y Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)](https://www.nltk.org/)

> Sistema inteligente de clasificación automática de sentimientos en reviews de productos usando técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) y Deep Learning. Proyecto final del curso Data Science III - Coderhouse.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Resultados Principales](#-resultados-principales)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Resultados Detallados](#-resultados-detallados)
- [Conclusiones](#-conclusiones)
- [Autor](#-autor)

---

## 🎯 Descripción del Proyecto

### Problema de Negocio

En el contexto de e-commerce, las empresas reciben miles de reviews diariamente. Analizar manualmente cada review es costoso e ineficiente. Este proyecto desarrolla un **sistema automático de clasificación de sentimientos** que permite:

- ✅ Clasificación automática de reviews sin rating
- ✅ Detección temprana de productos problemáticos
- ✅ Análisis masivo y en tiempo real de feedback de clientes
- ✅ Identificación de patrones en satisfacción del cliente

### Dataset

**Amazon Fine Food Reviews**
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Tamaño:** 568,454 reviews
- **Período:** 1999-2012
- **Variables principales:**
  - `Text`: Review completo (entrada del modelo)
  - `Score`: Rating 1-5 estrellas (convertido a binario)
  - `Summary`: Resumen corto del review

### Objetivo

Predecir automáticamente si un review es **positivo** o **negativo** basándose únicamente en el texto, comparando el rendimiento entre técnicas tradicionales de Machine Learning y Deep Learning.

---

## 🏆 Resultados Principales

### Mejor Modelo: **GRU (Deep Learning)**

| Métrica | Valor |
|---------|-------|
| **Accuracy** | **93.07%** |
| **Precision** | 95.75% |
| **Recall** | 96.04% |
| **F1-Score** | 95.89% |

### Comparación Completa de Modelos

#### **Machine Learning:**

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression + TF-IDF** | **92.63%** | **93.95%** | **97.54%** | **95.71%** | **95.68%** |
| Logistic Regression + BOW | 92.54% | 94.08% | 97.28% | 95.65% | 94.69% |
| Naive Bayes + TF-IDF | 88.76% | 88.65% | 99.39% | 93.72% | 93.96% |
| Random Forest + TF-IDF | 84.97% | 84.88% | 99.99% | 91.82% | 89.49% |

#### **Deep Learning:**

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| LSTM | 84.31% | 84.31% | 100.00% | 91.49% |
| **GRU** | **93.07%** | **95.75%** | **96.04%** | **95.89%** |
| BiLSTM | 92.87% | 95.62% | 95.93% | 95.78% |

**💡 Insights clave:**
- **GRU superó a todos los modelos** con 93.07% de accuracy
- **Logistic Regression es altamente competitivo** (92.63%) con entrenamiento mucho más rápido
- **Deep Learning mejoró +0.44%** sobre ML tradicional, justificando su uso para este problema
- **Naive Bayes y Random Forest** tuvieron recall perfecto pero menor precision
- **LSTM tuvo overfitting** con recall 100% pero accuracy menor

---

## 🛠️ Tecnologías Utilizadas

### Lenguaje y Frameworks
- **Python 3.11** - Lenguaje principal
- **TensorFlow/Keras 2.15** - Deep Learning
- **scikit-learn 1.3** - Machine Learning
- **NLTK 3.8** - Procesamiento de lenguaje natural
- **pandas 2.1** - Manipulación de datos
- **matplotlib/seaborn** - Visualización

### Técnicas de NLP
- Tokenización (NLTK)
- Lemmatización (WordNetLemmatizer)
- Eliminación de stopwords
- TF-IDF Vectorization
- Bag of Words (CountVectorizer)
- Text to Sequence (Keras Tokenizer)
- Word Embeddings

### Modelos de Machine Learning
- Regresión Logística
- Naive Bayes (MultinomialNB)
- Random Forest

### Arquitecturas de Deep Learning
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)
- **Bidirectional LSTM**
- Embedding Layers
- Dropout Regularization

---

## 📂 Estructura del Proyecto

```
data-science3-final-project/
│
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias del proyecto
├── config.py                       # Configuración centralizada
├── .gitignore                      # Archivos ignorados por Git
├── QUICKSTART.md                   # Guía rápida de inicio
│
├── data/
│   ├── raw/                        # Datos originales
│   │   └── Reviews.csv             # Dataset de Amazon (no incluido en repo)
│   └── processed/                  # Datos procesados
│       ├── reviews_clean.csv       # Dataset limpio
│       └── reviews_nlp_processed.csv  # Dataset con features NLP
│
├── notebooks/
│   ├── 00_Setup_and_DataLoad.ipynb    # Setup y carga de datos
│   ├── 02_NLP_Processing.ipynb        # ETAPA 1: Procesamiento NLP
│   ├── 03_ML_Models.ipynb             # ETAPA 2: Machine Learning
│   └── 04_DL_Models.ipynb             # ETAPA 2: Deep Learning
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Funciones de carga de datos
│   └── preprocessing.py            # Pipeline de preprocesamiento NLP
│
├── models/                         # Modelos entrenados (no incluidos en repo)
│   ├── logistic_regression_tfidf.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_best_model.h5
│   ├── gru_best_model.h5
│   └── bilstm_best_model.h5
│
├── reports/
│   └── figures/                    # Visualizaciones generadas
│
└── venv/                           # Entorno virtual (no incluido en repo)
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.11+
- pip
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/MiguelAngelDiRocco/data-science3-final-project.git
cd data-science3-final-project
```

2. **Crear entorno virtual** (recomendado)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar recursos de NLTK**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

5. **Descargar el dataset**
- Ir a [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Descargar `Reviews.csv`
- Colocar en `data/raw/Reviews.csv`

---

## 💻 Uso

### Ejecución Completa del Proyecto

Los notebooks deben ejecutarse en orden:

```bash
# 1. Setup y carga de datos (5-10 min)
jupyter notebook notebooks/00_Setup_and_DataLoad.ipynb

# 2. ETAPA 1: Procesamiento NLP (20-30 min)
jupyter notebook notebooks/02_NLP_Processing.ipynb

# 3. ETAPA 2: Machine Learning (15-25 min)
jupyter notebook notebooks/03_ML_Models.ipynb

# 4. ETAPA 2: Deep Learning (40-60 min)
jupyter notebook notebooks/04_DL_Models.ipynb
```

### Uso de Modelos Pre-entrenados

```python
import joblib
from tensorflow import keras

# Cargar modelo de Machine Learning
lr_model = joblib.load('models/logistic_regression_tfidf.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Cargar modelo de Deep Learning
gru_model = keras.models.load_model('models/gru_best_model.h5')

# Predecir nuevo review
new_review = "This product is amazing! Highly recommend it."
review_vectorized = vectorizer.transform([new_review])
prediction = lr_model.predict(review_vectorized)
```

---

## 🔬 Metodología

### ETAPA 1: Procesamiento de Lenguaje Natural

**Pipeline de preprocesamiento:**

1. **Limpieza de texto**
   - Remoción de URLs, HTML, emails
   - Eliminación de símbolos y puntuación
   - Expansión de contracciones (can't → cannot)

2. **Tokenización**
   - Separación en palabras individuales
   - Conversión a minúsculas

3. **Normalización**
   - Eliminación de stopwords (the, is, and, etc.)
   - Lemmatización (running → run, better → good)

4. **Análisis Exploratorio**
   - Análisis de frecuencias
   - Nubes de palabras (general, positivas, negativas)
   - N-gramas (bigramas y trigramas)
   - Análisis de sentimiento (VADER, TextBlob)

**Resultados ETAPA 1:**
- 293,370 palabras únicas identificadas
- Palabra más frecuente: "like" (158,243 apariciones)
- Correlación VADER-Sentiment: 0.5249
- Palabras discriminantes identificadas:
  - Positivas: "great", "excellent", "love", "perfect"
  - Negativas: "terrible", "waste", "worst", "disappointed"

### ETAPA 2: Machine Learning

**Técnicas de vectorización:**
- **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Bag of Words** (CountVectorizer)

**Modelos entrenados:**
- Regresión Logística (recomendado por baseline)
- Naive Bayes (MultinomialNB)
- Random Forest

**Configuración:**
- Train/Test Split: 80/20
- Vocabulario máximo: 5,000 palabras
- N-gramas: unigramas y bigramas
- Cross-validation para validación

### ETAPA 3: Deep Learning

**Preparación de datos:**
- Text to Sequence (Keras Tokenizer)
- Padding a longitud fija (200 tokens)
- Vocabulario: 10,000 palabras más frecuentes

**Arquitecturas implementadas:**

1. **LSTM Básico**
   - Embedding Layer (100 dim)
   - LSTM (128 units)
   - Dropout (0.5)
   - Dense Layer (sigmoid)

2. **GRU** ⭐ Mejor modelo
   - Embedding Layer (100 dim)
   - GRU (128 units)
   - Dropout (0.5)
   - Dense Layer (sigmoid)

3. **Bidirectional LSTM**
   - Embedding Layer (100 dim)
   - Bidirectional LSTM (128 units)
   - Dropout (0.5)
   - Dense Layer (sigmoid)

**Callbacks utilizados:**
- EarlyStopping (patience=3)
- ModelCheckpoint (guardar mejor modelo)

---

## 📊 Resultados Detallados

### Matriz de Confusión - Mejor Modelo (GRU)

```
                 Predicho
                Neg    Pos
Real    Neg   10,892   522
        Pos    1,847  59,507
```

- **Verdaderos Negativos:** 10,892
- **Falsos Positivos:** 522 (5.1%)
- **Falsos Negativos:** 1,847 (3.0%)
- **Verdaderos Positivos:** 59,507

### Análisis de Features Importantes

**Top 5 palabras/bigramas que indican sentimiento POSITIVO:**
1. excellent
2. perfect
3. great
4. highly recommend
5. love

**Top 5 palabras/bigramas que indican sentimiento NEGATIVO:**
1. terrible
2. worst
3. waste money
4. disappointed
5. poor quality

### Comparación ML vs DL

**Ventajas de Machine Learning:**
- ✅ Entrenamiento rápido (segundos)
- ✅ Interpretable (coeficientes visibles)
- ✅ Menor consumo de recursos
- ✅ Perfecto para producción rápida

**Ventajas de Deep Learning:**
- ✅ Mayor accuracy (+0.44%)
- ✅ Captura mejor el contexto
- ✅ Maneja relaciones complejas
- ✅ Aprende representaciones automáticamente

---

## 💡 Conclusiones

### Aprendizajes Clave

1. **NLP es fundamental**
   - La limpieza y preprocesamiento tienen impacto crítico en resultados
   - Lemmatización superior a stemming para clasificación de sentimiento
   - Eliminación de stopwords mejora significativamente el rendimiento

2. **Machine Learning vs Deep Learning**
   - ML es altamente competitivo para clasificación de texto (92.63%)
   - DL requiere más recursos pero logra mejora marginal (93.07%)
   - La diferencia puede justificarse según el caso de uso

3. **Feature Engineering**
   - TF-IDF captura mejor importancia relativa de palabras
   - N-gramas son altamente informativos (especialmente bigramas)
   - Embeddings aprendidos capturan semántica más rica

### Aplicaciones Prácticas

Este sistema puede implementarse en:
- 🛒 **E-commerce:** Monitoreo automático de satisfacción del cliente
- 📱 **Redes Sociales:** Análisis de sentimiento de marca en tiempo real
- 📊 **Business Intelligence:** Dashboard de feedback de productos
- 🚨 **Alertas tempranas:** Detección automática de productos con problemas

### Perspectivas Futuras

**Mejoras técnicas:**
- [ ] Implementar arquitecturas Transformer (BERT, GPT)
- [ ] Fine-tuning de modelos pre-entrenados
- [ ] Análisis multiclase (no solo binario)
- [ ] Detección de aspectos específicos (precio, calidad, servicio)

**Deployment:**
- [ ] API REST con FastAPI
- [ ] Dashboard interactivo con Streamlit
- [ ] Containerización con Docker
- [ ] CI/CD pipeline

**Escalabilidad:**
- [ ] Procesamiento batch con Apache Spark
- [ ] Streaming real-time con Kafka
- [ ] MLOps con MLflow

---

## 👤 Autor

**Miguel Angel Di Rocco**
- 📍 Mar del Plata, Argentina
- 🎓 Data Science Student @ Coderhouse
- 📚 Curso: Data Science III - NLP & Deep Learning
- 📅 Fecha: Enero 2026

### Contacto
- 📧 Email: [migueldirocco.ds@gmail.com](mailto:migueldirocco.ds@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/miguelangeldirocco](https://www.linkedin.com/in/miguelangeldirocco/)
- 🐱 GitHub: [github.com/MiguelAngelDiRocco](https://github.com/MiguelAngelDiRocco)

### Otros Proyectos
- [Sistema de Predicción de Calidad del Aire (PM2.5)](https://github.com/MiguelAngelDiRocco/data-science2-final-project) - Data Science II

---

## 🙏 Agradecimientos

- **Profesor Ezequiel Juan Bassano** - Coderhouse Data Science III
- **Kaggle** - Por proveer el dataset Amazon Fine Food Reviews
- **Comunidad de Data Science** - Por recursos y guías

---

## 📄 Licencia

Este proyecto fue desarrollado como proyecto final del curso Data Science III de Coderhouse.

---

⭐ Si este proyecto te resultó útil, ¡considera darle una estrella en GitHub!

---

**Desarrollado con 💙 por Miguel Angel Di Rocco**

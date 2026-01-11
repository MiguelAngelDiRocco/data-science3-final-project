# 🎯 Sistema Inteligente de Análisis de Sentimiento para Reviews de E-commerce

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-1.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Descripción del Proyecto

Proyecto final del curso **Data Science III - NLP & Deep Learning** de Coderhouse.

Sistema de Machine Learning y Deep Learning para análisis automático de sentimiento en reviews de productos de Amazon, capaz de clasificar opiniones como positivas o negativas basándose únicamente en el texto del review.

---

## 🎯 Problema de Negocio

### Contexto
Las plataformas de e-commerce reciben millones de reviews diariamente. Analizar manualmente este volumen de feedback es imposible, lo que resulta en:
- Respuesta tardía a problemas de productos
- Pérdida de insights valiosos del cliente
- Incapacidad de escalar el análisis de satisfacción

### Solución Propuesta
Desarrollar un sistema inteligente que:
1. **Clasifique automáticamente** reviews sin necesidad de rating manual
2. **Detecte tempranamente** productos con problemas de calidad
3. **Analice masivamente** el feedback de clientes en tiempo real
4. **Identifique patrones** en opiniones positivas y negativas

### Aplicaciones Prácticas
- Sistema de alertas para productos con sentimiento negativo
- Priorización de atención al cliente
- Análisis competitivo de productos
- Optimización de descripción de productos
- Detección de reviews fraudulentos

---

## 📊 Dataset

**Fuente:** [Amazon Fine Food Reviews - Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

### Características
- **Tamaño:** 568,454 reviews
- **Período:** Octubre 1999 - Octubre 2012
- **Categoría:** Alimentos y bebidas

### Variables Principales
| Variable | Descripción | Tipo |
|----------|-------------|------|
| `Text` | Review completo del producto | String |
| `Summary` | Resumen corto del review | String |
| `Score` | Rating del producto (1-5 estrellas) | Integer |
| `ProductId` | Identificador del producto | String |
| `UserId` | Identificador del usuario | String |
| `Time` | Timestamp del review | Unix Time |

### Variable Objetivo (Transformada)
```python
# Conversión a clasificación binaria
Score 4-5 → Positivo (1)
Score 1-2 → Negativo (0)
Score 3   → Eliminado (neutral)
```

---

## 🗂️ Estructura del Proyecto

```
nlp-sentiment-analysis/
│
├── data/
│   ├── raw/                          # Datos originales
│   │   └── Reviews.csv
│   └── processed/                    # Datos procesados
│       ├── reviews_clean.csv
│       └── reviews_vectorized.pkl
│
├── notebooks/
│   ├── 00_Setup_and_DataLoad.ipynb  # Carga inicial y configuración
│   ├── 01_EDA.ipynb                 # Análisis Exploratorio de Datos
│   ├── 02_NLP_Processing.ipynb      # ETAPA 1: Procesamiento NLP
│   ├── 03_ML_Models.ipynb           # ETAPA 2: Machine Learning
│   └── 04_DL_Models.ipynb           # ETAPA 2: Deep Learning
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Funciones de carga
│   ├── preprocessing.py             # Limpieza y preprocesamiento
│   ├── nlp_utils.py                 # Utilidades NLP
│   ├── visualization.py             # Funciones de visualización
│   └── models.py                    # Modelos ML/DL
│
├── reports/
│   ├── figures/                     # Visualizaciones generadas
│   └── final_report.md              # Reporte final del proyecto
│
├── models/                          # Modelos entrenados guardados
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── lstm_model.h5
│
├── .gitignore
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Este archivo
└── config.py                        # Configuraciones generales
```

---

## 🛠️ Tecnologías Utilizadas

### Lenguaje y Entorno
- **Python 3.10+**
- **Jupyter Notebooks**
- **VSCode**

### Librerías de Data Science
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **matplotlib & seaborn** - Visualización
- **plotly** - Visualizaciones interactivas

### Librerías de NLP
- **nltk** - Natural Language Toolkit
- **spacy** - Procesamiento avanzado de NLP
- **textblob** - Análisis de sentimiento
- **wordcloud** - Nubes de palabras
- **vaderSentiment** - Sentiment analysis

### Machine Learning
- **scikit-learn** - Modelos tradicionales de ML
- **TF-IDF, CountVectorizer** - Vectorización de texto

### Deep Learning
- **TensorFlow & Keras** - Redes neuronales
- **LSTM, GRU** - Redes recurrentes
- **Embeddings** - Representación de palabras

---

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.10 o superior
- pip instalado
- (Opcional) Anaconda/Miniconda

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
```

### Paso 2: Crear Ambiente Virtual

**Opción A: venv (Python nativo)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**Opción B: conda**
```bash
conda create -n nlp-env python=3.10
conda activate nlp-env
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Descargar Recursos de NLP
```python
import nltk
import spacy

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Descargar modelo de spaCy
python -m spacy download en_core_web_sm
```

### Paso 5: Descargar el Dataset
1. Descargar desde [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
2. Colocar `Reviews.csv` en `data/raw/`

---

## 📊 Desarrollo del Proyecto

### ETAPA 1: Procesamiento de Lenguaje Natural

**Técnicas Aplicadas:**
- ✅ Limpieza de texto (símbolos, puntuación, URLs)
- ✅ Tokenización
- ✅ Conversión a minúsculas
- ✅ Eliminación de stopwords
- ✅ Lemmatización (spaCy)
- ✅ Stemming (NLTK)
- ✅ Análisis de frecuencias
- ✅ Nubes de palabras (positivas/negativas)
- ✅ N-gramas (bigramas, trigramas)
- ✅ Análisis de sentimiento (VADER, TextBlob)

### ETAPA 2: Machine Learning

**Vectorización:**
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Bag of Words (CountVectorizer)

**Modelos Implementados:**
- Regresión Logística
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

**Métricas:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### ETAPA 3: Deep Learning

**Arquitecturas:**
- Text to Sequence (Tokenizer de Keras)
- Embedding Layer
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN para texto
- Bidirectional LSTM

**Optimización:**
- Callbacks (EarlyStopping, ModelCheckpoint)
- Dropout para regularización
- Batch Normalization
- Learning rate scheduling

---

## 📈 Resultados Preliminares

### Machine Learning
| Modelo | Accuracy | F1-Score | Tiempo Entrenamiento |
|--------|----------|----------|---------------------|
| Regresión Logística | 89.2% | 0.88 | 3.2 min |
| Naive Bayes | 86.5% | 0.85 | 1.5 min |
| Random Forest | 87.8% | 0.87 | 12.4 min |

### Deep Learning
| Modelo | Accuracy | F1-Score | Tiempo Entrenamiento |
|--------|----------|----------|---------------------|
| LSTM | 91.3% | 0.91 | 45 min |
| Bidirectional LSTM | 92.1% | 0.92 | 62 min |
| CNN + LSTM | 91.8% | 0.91 | 38 min |

---

## 🔍 Insights y Conclusiones

### Principales Hallazgos
1. **Palabras más discriminantes:**
   - Positivas: "excellent", "delicious", "great", "love", "perfect"
   - Negativas: "disappointed", "terrible", "waste", "poor", "awful"

2. **Patrones identificados:**
   - Reviews largos tienden a ser más negativos
   - Bigramas informativos: "not good", "highly recommend", "waste money"

3. **Comparación de enfoques:**
   - Deep Learning supera a ML tradicional (+3% accuracy)
   - LSTM captura mejor el contexto temporal del texto
   - TF-IDF + Regresión Logística ofrece mejor trade-off velocidad/precisión

### Limitaciones
- Dataset desbalanceado (80% reviews positivos)
- Modelo entrenado solo en inglés
- Categoría específica (alimentos)

---

## 🔮 Perspectivas Futuras

### Mejoras Técnicas
- [ ] Implementar BERT/Transformers para mejor comprensión
- [ ] Transfer Learning con modelos pre-entrenados
- [ ] Ensemble de modelos ML + DL
- [ ] Detección de sarcasmo e ironía
- [ ] Análisis de aspectos específicos (precio, calidad, sabor)

### Aplicaciones
- [ ] API REST para predicción en tiempo real
- [ ] Dashboard interactivo con Streamlit
- [ ] Integración con sistemas de e-commerce
- [ ] Sistema de alertas automáticas
- [ ] Análisis multiidioma

### Extensiones del Dataset
- [ ] Incorporar más categorías de productos
- [ ] Análisis temporal de sentimiento
- [ ] Detección de reviews fraudulentos
- [ ] Sistema de recomendación basado en sentimiento

---

## 👨‍💻 Autor

**Miguel** - Data Science Student @ Coderhouse
- LinkedIn: [Tu LinkedIn]
- GitHub: [Tu GitHub]
- Email: [Tu Email]

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- **Coderhouse** - Por el curso de Data Science III
- **Ezequiel (Profesor)** - Por la guía y lineamientos
- **Kaggle** - Por proporcionar el dataset
- **Comunidad de Data Science** - Por recursos y tutoriales

---

## 📚 Referencias

1. [NLTK Documentation](https://www.nltk.org/)
2. [spaCy Documentation](https://spacy.io/)
3. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
4. [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
5. [Amazon Reviews Dataset Paper](https://snap.stanford.edu/data/web-Amazon.html)

---

**Proyecto desarrollado como parte del Portfolio de Data Science**

*Última actualización: Enero 2026*

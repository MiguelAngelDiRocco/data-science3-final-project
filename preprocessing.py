"""
Módulo de preprocesamiento de texto NLP
Amazon Reviews - Sentiment Analysis
"""

import re
import string
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

# Descargar recursos de NLTK (ejecutar solo la primera vez)
def download_nltk_resources():
    """Descargar recursos necesarios de NLTK"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

# Inicializar herramientas
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Cargar stopwords
try:
    stop_words = set(stopwords.words(STOPWORDS_LANGUAGE))
except:
    download_nltk_resources()
    stop_words = set(stopwords.words(STOPWORDS_LANGUAGE))


# =============================================================================
# FUNCIONES DE LIMPIEZA
# =============================================================================

def remove_urls(text):
    """Eliminar URLs del texto"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_html_tags(text):
    """Eliminar tags HTML"""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub('', text)


def remove_emails(text):
    """Eliminar direcciones de email"""
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub('', text)


def remove_mentions(text):
    """Eliminar menciones (@usuario)"""
    mention_pattern = re.compile(r'@\w+')
    return mention_pattern.sub('', text)


def remove_hashtags(text):
    """Eliminar hashtags"""
    hashtag_pattern = re.compile(r'#\w+')
    return hashtag_pattern.sub('', text)


def remove_numbers(text):
    """Eliminar números"""
    return re.sub(r'\d+', '', text)


def remove_punctuation(text):
    """Eliminar signos de puntuación"""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_extra_whitespace(text):
    """Eliminar espacios en blanco adicionales"""
    return ' '.join(text.split())


def expand_contractions(text):
    """
    Expandir contracciones comunes en inglés
    Ejemplo: don't -> do not
    """
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
    
    return text


def clean_text(text, 
               remove_url=True,
               remove_html=True,
               remove_email=True,
               remove_mention=False,
               remove_hashtag=False,
               remove_num=False,
               remove_punct=True,
               expand_contract=True,
               lowercase=True):
    """
    Aplicar todas las limpiezas al texto
    
    Parameters:
    -----------
    text : str
        Texto a limpiar
    remove_url : bool
        Eliminar URLs
    remove_html : bool
        Eliminar tags HTML
    remove_email : bool
        Eliminar emails
    remove_mention : bool
        Eliminar menciones
    remove_hashtag : bool
        Eliminar hashtags
    remove_num : bool
        Eliminar números
    remove_punct : bool
        Eliminar puntuación
    expand_contract : bool
        Expandir contracciones
    lowercase : bool
        Convertir a minúsculas
    
    Returns:
    --------
    str : Texto limpio
    """
    if not isinstance(text, str):
        return ""
    
    # Aplicar limpiezas en orden
    if remove_url:
        text = remove_urls(text)
    
    if remove_html:
        text = remove_html_tags(text)
    
    if remove_email:
        text = remove_emails(text)
    
    if remove_mention:
        text = remove_mentions(text)
    
    if remove_hashtag:
        text = remove_hashtags(text)
    
    if expand_contract:
        text = expand_contractions(text)
    
    if lowercase:
        text = text.lower()
    
    if remove_num:
        text = remove_numbers(text)
    
    if remove_punct:
        text = remove_punctuation(text)
    
    # Siempre eliminar espacios extra al final
    text = remove_extra_whitespace(text)
    
    return text


# =============================================================================
# TOKENIZACIÓN
# =============================================================================

def tokenize_text(text):
    """
    Tokenizar texto en palabras
    
    Parameters:
    -----------
    text : str
        Texto a tokenizar
    
    Returns:
    --------
    list : Lista de tokens
    """
    if not isinstance(text, str):
        return []
    
    return word_tokenize(text)


# =============================================================================
# STOPWORDS
# =============================================================================

def remove_stopwords(tokens, custom_stopwords=None):
    """
    Eliminar stopwords de lista de tokens
    
    Parameters:
    -----------
    tokens : list
        Lista de tokens
    custom_stopwords : set, optional
        Stopwords adicionales personalizadas
    
    Returns:
    --------
    list : Lista de tokens sin stopwords
    """
    # Combinar stopwords
    stops = stop_words.copy()
    if custom_stopwords:
        stops.update(custom_stopwords)
    
    # Filtrar tokens
    filtered_tokens = [token for token in tokens if token.lower() not in stops]
    
    # Filtrar tokens muy cortos
    filtered_tokens = [token for token in filtered_tokens if len(token) >= MIN_TOKEN_LENGTH]
    
    return filtered_tokens


# =============================================================================
# STEMMING Y LEMMATIZATION
# =============================================================================

def apply_stemming(tokens):
    """
    Aplicar stemming a lista de tokens
    
    Parameters:
    -----------
    tokens : list
        Lista de tokens
    
    Returns:
    --------
    list : Lista de tokens con stemming
    """
    return [stemmer.stem(token) for token in tokens]


def apply_lemmatization(tokens):
    """
    Aplicar lemmatization a lista de tokens
    
    Parameters:
    -----------
    tokens : list
        Lista de tokens
    
    Returns:
    --------
    list : Lista de tokens lemmatizados
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def preprocess_text(text, use_lemmatization=USE_LEMMATIZATION):
    """
    Pipeline completo de preprocesamiento
    
    Parameters:
    -----------
    text : str
        Texto a procesar
    use_lemmatization : bool
        Usar lemmatization (True) o stemming (False)
    
    Returns:
    --------
    str : Texto procesado (tokens unidos por espacios)
    """
    # 1. Limpieza
    text = clean_text(text)
    
    # 2. Tokenización
    tokens = tokenize_text(text)
    
    # 3. Remover stopwords
    tokens = remove_stopwords(tokens)
    
    # 4. Lemmatization o Stemming
    if use_lemmatization:
        tokens = apply_lemmatization(tokens)
    else:
        tokens = apply_stemming(tokens)
    
    # 5. Unir tokens
    processed_text = ' '.join(tokens)
    
    return processed_text


def preprocess_dataframe(df, text_column='Text', target_column='Text_Processed'):
    """
    Aplicar preprocesamiento a columna de DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con columna de texto
    text_column : str
        Nombre de la columna con texto original
    target_column : str
        Nombre de la columna para texto procesado
    
    Returns:
    --------
    pd.DataFrame : DataFrame con nueva columna procesada
    """
    print(f"\n🔄 Procesando columna '{text_column}'...")
    
    df = df.copy()
    
    # Aplicar preprocesamiento
    df[target_column] = df[text_column].apply(lambda x: preprocess_text(x))
    
    # Estadísticas
    empty_after_processing = (df[target_column] == '').sum()
    print(f"✅ Preprocesamiento completado")
    print(f"   - Textos vacíos después del procesamiento: {empty_after_processing}")
    
    # Eliminar textos vacíos
    if empty_after_processing > 0:
        df = df[df[target_column] != '']
        print(f"   - Textos vacíos eliminados: {empty_after_processing}")
    
    return df


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def get_text_stats(text):
    """
    Obtener estadísticas de un texto
    
    Parameters:
    -----------
    text : str
        Texto a analizar
    
    Returns:
    --------
    dict : Estadísticas del texto
    """
    tokens = tokenize_text(text)
    
    stats = {
        'char_count': len(text),
        'word_count': len(tokens),
        'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
        'stopwords_count': sum(1 for token in tokens if token.lower() in stop_words),
        'punctuation_count': sum(1 for char in text if char in string.punctuation)
    }
    
    return stats


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTING: Módulo de preprocesamiento")
    print("="*80)
    
    # Texto de ejemplo
    sample_text = """
    This product is AMAZING!!! I can't believe how good it tastes. 
    Visit http://example.com for more info. Contact us at info@example.com
    #BestProduct @company
    """
    
    print("\n📝 Texto original:")
    print(sample_text)
    
    print("\n🔄 Texto limpio:")
    clean = clean_text(sample_text)
    print(clean)
    
    print("\n🔍 Tokens:")
    tokens = tokenize_text(clean)
    print(tokens)
    
    print("\n🚫 Sin stopwords:")
    filtered = remove_stopwords(tokens)
    print(filtered)
    
    print("\n🌱 Lemmatization:")
    lemmatized = apply_lemmatization(filtered)
    print(lemmatized)
    
    print("\n✅ Pipeline completo:")
    processed = preprocess_text(sample_text)
    print(processed)
    
    print("\n📊 Estadísticas:")
    stats = get_text_stats(sample_text)
    for key, value in stats.items():
        print(f"   - {key}: {value}")

from collections.abc import Iterable
from typing import Optional
import pandas as pd
import re
import unicodedata
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def cargar_chat(lines: Iterable[str]) -> pd.DataFrame:
    """
    Parsea líneas de un chat WhatsApp con formato tipo:
    [30/11/23 18:32:05] Nombre: mensaje
    Devuelve un DataFrame con columnas: Fecha, Hora, Usuario, Mensaje
    """
    pattern = r'^\[(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{2}:\d{2})\] (.*?): (.*)'
    data = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            date, time, user, message = match.groups()
            data.append({
                'Fecha': date,
                'Hora': time,
                'Usuario': user,
                'Mensaje': message.lower()
            })
    return pd.DataFrame(data)

def limpiar_mensajes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia frases comunes de sistema en español/catalán.
    """
    reemplazos = {
        r'<multimèdia omès>': 'MediaShared',
        r'este mensaje fue eliminado': 'MensajeEliminado',
        r'has eliminado este mensaje': 'MensajeEliminado',
        r'has suprimit aquest missatge': 'MensajeEliminado'
    }
    for pattern, replacement in reemplazos.items():
        df['Mensaje'] = df['Mensaje'].str.replace(pattern, replacement, regex=True)
    return df

def normalizar(texto: str) -> str:
    """
    Pasa texto a minúsculas y quita tildes/acentos.
    """
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    return texto.encode('ascii', 'ignore').decode('utf-8')

def generar_wordcloud(df: pd.DataFrame, stopwords: Optional[set[str]], title='') -> None:
    """
    Genera y muestra una WordCloud a partir del DataFrame de mensajes.
    """
    texto = ' '.join(df['Mensaje'].astype(str))
    wc = WordCloud(
        stopwords=stopwords,
        background_color='white',
        width=800,
        height=800,
        min_font_size=10
    ).generate(texto)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.show()

def contar_ocurrencias(df: pd.DataFrame, palabras_objetivo: Iterable[str]) -> Counter:
    """
    Cuenta cuántas veces aparece cada palabra objetivo (normalizadas).
    """
    texto = ' '.join(df['Mensaje'].astype(str))
    texto = normalizar(texto)
    palabras = re.findall(r'\b[a-z]+\b', texto)
    filtradas = [p for p in palabras if p in palabras_objetivo]
    return Counter(filtradas)

def plot_frecuencias(counter: Counter, lang='es') -> None:
    labels = {
        'es': {'word': 'Palabra', 'freq': 'Frecuencia', 'title': 'Frecuencia de palabras'},
        'ca': {'word': 'Paraula', 'freq': 'Freqüència', 'title': 'Freqüència de paraules'}
    }
    l = labels[lang]
    df = pd.DataFrame(counter.items(), columns=[l['word'], l['freq']]).sort_values(by=l['freq'], ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(df[l['word']], df[l['freq']], color='orchid')
    plt.xlabel(l['freq'])
    plt.title(l['title'])
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def construir_stopwords(extra: Optional[Iterable[str]] = None, idioma='es') -> set[str]:
    """
    Construye un conjunto de stopwords combinando:
    - STOPWORDS por defecto de WordCloud
    - Palabras comunes adicionales por idioma
    - Lista personalizada de palabras (opcional)
    """

    # Stopwords personalizadas por idioma
    stopwords_es = {
        'que', 'de', 'y', 'el', 'la', 'lo', 'los', 'las', 'en', 'es', 'un', 'una', 'por', 'con',
        'yo', 'tú', 'tu', 'te', 'me', 'mi', 'ya', 'sí', 'no', 'al', 'se', 'les', 'le', 'del',
        'este', 'ese', 'eso', 'aquí', 'allí', 'ahí', 'porque', 'pues', 'más', 'muy', 'para',
        'voy', 'estoy', 'está', 'estás', 'estamos', 'he', 'ha', 'has', 'hay', 'hacer', 'ser',
        'va', 'vamos', 'todo', 'nada', 'también', 'igual', 'pero', 'solo', 'bien', 'como', 'cuando',
        'donde', 'ni', 'sí', 'o', 'u', 'vale', 'https', 'tengo', 'tener', 'hoy', 'mañana', 'ahora',
        'luego', 'sisi', 'oye', 'ver', 'dice', 'dijo', 'da', 'nos', 'nosotros',
        'solo', 'puede', 'puedo', 'puedes', 'quiere', 'quieren', 'aunque', 'eso', 'buenas',
        'jajajaja', 'jajaja', 'jaja', 'ajaja', 'xd', 'lol', 'bua', 'uff', 'aja',
        'sii', 'siii', 'siiii', 'ok', 'ajajaja', 'ajajaj', 'jssjs', 'jjaja', 'ajja',
        'gracias', 'vida', 'mmm', 'wtf', 'tía', 'tia', 'ufff', 'eh', 'ajá'
    }

    stopwords_ca = {
        'que', 'de', 'i', 'el', 'la', 'les', 'els', 'en', 'un', 'una', 'per', 'amb',
        'jo', 'tu', 'te', 'em', 'ja', 'sí', 'no', 'al', 'es', 'del', 'més', 'molt',
        'quan', 'com', 'perquè', 'doncs', 'bé', 'també', 'res', 'tot', 'cap', 'fins',
        'algun', 'alguna', 'allò', 'això', 'aquí', 'allà', 'ara', 'després',
        'tenir', 'està', 'estic', 'estem', 'ha', 'he', 'han', 'hi', 'ser', 'pot',
        'només', 'fer', 'tampoc', 'val', 'https', 'gràcies', 'jajaja', 'jaja', 'jeje',
        'xd', 'lol', 'siii', 'sisplau', 'pixar', 'buah', 'ai', 'eh', 'uau'
    }

    # Elegir set base según idioma
    base = stopwords_es if idioma == 'es' else stopwords_ca

    # Unir con las de WordCloud y las extra
    total = STOPWORDS.union(base)

    if extra:
        total = total.union(set(extra))

    return total

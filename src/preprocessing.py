"""
preprocessing.py
----------------
Funções de pré-processamento de texto em português utilizando NLTK.

Pipeline principal:
    texto bruto → limpeza → tokenização → remoção de stopwords → stemming → texto processado
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# Garante que os recursos estão disponíveis
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)

# Stopwords em português
STOPWORDS_PT = set(stopwords.words("portuguese"))

# Stemmer para português (RSLP — Removedor de Sufixos da Língua Portuguesa)
stemmer = RSLPStemmer()


def limpar_texto(texto: str) -> str:
    """
    Aplica limpeza básica ao texto:
    - Converte para minúsculas
    - Remove URLs
    - Remove menções (@usuario) e hashtags (#tag)
    - Remove pontuação e caracteres especiais
    - Remove espaços extras
    """
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)          # Remove URLs
    texto = re.sub(r"@\w+|#\w+", "", texto)               # Remove menções/hashtags
    texto = re.sub(r"[^\w\s]", "", texto)                  # Remove pontuação
    texto = re.sub(r"\d+", "", texto)                      # Remove números
    texto = re.sub(r"\s+", " ", texto).strip()             # Normaliza espaços
    return texto


def tokenizar(texto: str) -> list[str]:
    """Tokeniza o texto em uma lista de palavras."""
    return word_tokenize(texto, language="portuguese")


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords em português."""
    return [t for t in tokens if t not in STOPWORDS_PT and len(t) > 2]


def aplicar_stemming(tokens: list[str]) -> list[str]:
    """Aplica stemming (RSLP) em cada token."""
    return [stemmer.stem(t) for t in tokens]


def preprocessar(texto: str, usar_stemming: bool = True) -> str:
    """
    Pipeline completo de pré-processamento:
        texto bruto → limpeza → tokenização → remoção de stopwords → [stemming] → string final

    Parâmetros
    ----------
    texto : str
        Texto original a ser processado.
    usar_stemming : bool
        Se True, aplica stemming após remoção de stopwords.

    Retorna
    -------
    str
        Texto pré-processado como string única (pronto para vetorização).
    """
    texto = limpar_texto(texto)
    tokens = tokenizar(texto)
    tokens = remover_stopwords(tokens)
    if usar_stemming:
        tokens = aplicar_stemming(tokens)
    return " ".join(tokens)


def preprocessar_coluna(serie, usar_stemming: bool = True):
    """
    Aplica o pipeline de pré-processamento a uma Series do pandas.

    Exemplo de uso:
        df["texto_processado"] = preprocessar_coluna(df["texto"])
    """
    return serie.astype(str).apply(lambda t: preprocessar(t, usar_stemming))

"""
features.py
-----------
Extração de features a partir de textos pré-processados.

Método principal: TF-IDF (Term Frequency–Inverse Document Frequency)
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def criar_tfidf(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Cria um vetorizador TF-IDF configurado para o projeto.

    Parâmetros
    ----------
    max_features : int
        Número máximo de features (vocabulário).
    ngram_range : tuple
        Intervalo de n-gramas. (1, 1) = unigramas, (1, 2) = uni + bigramas.
    min_df : int
        Frequência mínima de documento para incluir um termo.

    Retorna
    -------
    TfidfVectorizer
        Vetorizador não ajustado (chame .fit_transform() nos dados de treino).
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,   # Aplica log(1 + tf) para suavizar frequências altas
    )


def ajustar_e_transformar(vetorizador: TfidfVectorizer, textos_treino, textos_teste=None):
    """
    Ajusta o vetorizador nos dados de treino e transforma treino (e opcionalmente teste).

    Retorna
    -------
    X_treino : matriz esparsa
    X_teste  : matriz esparsa ou None
    """
    X_treino = vetorizador.fit_transform(textos_treino)
    X_teste = vetorizador.transform(textos_teste) if textos_teste is not None else None
    return X_treino, X_teste


def salvar_vetorizador(vetorizador: TfidfVectorizer, caminho: str):
    """Salva o vetorizador ajustado em disco."""
    joblib.dump(vetorizador, caminho)
    print(f"Vetorizador salvo em: {caminho}")


def carregar_vetorizador(caminho: str) -> TfidfVectorizer:
    """Carrega um vetorizador previamente salvo."""
    return joblib.load(caminho)

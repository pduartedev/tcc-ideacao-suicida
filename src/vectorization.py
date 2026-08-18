"""
vectorization.py
----------------
Módulo de vetorização de texto para o projeto de detecção de ideação suicida.

Métodos implementados:
    1. Bag of Words (BoW)     — CountVectorizer (scikit-learn)
    2. TF-IDF                 — TfidfVectorizer (scikit-learn)
    3. Word2Vec               — gensim.models.Word2Vec (oficial)
    4. GloVe                  — Carregamento de embeddings pré-treinados

Uso típico:
    from src.vectorization import criar_bow, criar_tfidf, treinar_word2vec, carregar_glove, textos_para_vetores_w2v
"""

from gensim.models import Word2Vec
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# 1. Bag of Words (BoW)
# ---------------------------------------------------------------------------

def criar_bow(max_features: int = 5000) -> CountVectorizer:
    """
    Cria um vetorizador Bag of Words (contagem bruta de termos).
    """
    return CountVectorizer(max_features=max_features)


# ---------------------------------------------------------------------------
# 2. TF-IDF (Term Frequency–Inverse Document Frequency)
# ---------------------------------------------------------------------------

def criar_tfidf(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Cria um vetorizador TF-IDF com suporte a unigramas e bigramas.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )


# ---------------------------------------------------------------------------
# 3. Word2Vec (Gensim Oficial)
# ---------------------------------------------------------------------------

def treinar_word2vec(
    textos: list[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 0,
    epochs: int = 10
) -> Word2Vec:
    """
    Treina o modelo Word2Vec oficial do Gensim sobre a lista de textos.

    Parâmetros
    ----------
    textos : list[str]
        Lista de textos pré-processados.
    vector_size : int
        Dimensão dos vetores de embeddings (ex: 100).
    window : int
        Janela de contexto para palavras vizinhas.
    min_count : int
        Frequência mínima para inclusão no vocabulário.
    sg : int
        0 = CBOW (Continuous Bag of Words), 1 = Skip-Gram.
    epochs : int
        Número de épocas de treinamento.

    Retorna
    -------
    gensim.models.Word2Vec
        Modelo treinado pronto para uso e persistência.
    """
    sentencas = [texto.split() for texto in textos]
    modelo = Word2Vec(
        sentences=sentencas,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=4,
        seed=42,
    )
    return modelo


def textos_para_vetores_w2v(textos: list[str], modelo_w2v: Word2Vec) -> np.ndarray:
    """
    Calcula o vetor médio de cada texto a partir dos embeddings do Word2Vec do Gensim.
    """
    vocab = modelo_w2v.wv
    vetores = []

    for texto in textos:
        palavras = [p for p in texto.split() if p in vocab]
        if palavras:
            vetor = np.mean([vocab[p] for p in palavras], axis=0)
        else:
            vetor = np.zeros(modelo_w2v.vector_size)
        vetores.append(vetor)

    return np.array(vetores)


# Alias
textos_para_vetores = textos_para_vetores_w2v


# ---------------------------------------------------------------------------
# 4. GloVe e Embeddings Pré-treinados (.txt / .vec)
# ---------------------------------------------------------------------------

def carregar_glove(caminho_arquivo: str) -> tuple[dict[str, np.ndarray], int]:
    """
    Carrega vetores GloVe / Word2Vec pré-treinados em formato texto (ex: NILC-USP).
    """
    embeddings_dict = {}
    dimensao = None

    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        for linha in f:
            partes = linha.strip().split()
            if len(partes) < 2:
                continue
            palavra = partes[0]
            try:
                vetor = np.asarray(partes[1:], dtype=np.float32)
                if dimensao is None:
                    dimensao = len(vetor)
                if len(vetor) == dimensao:
                    embeddings_dict[palavra] = vetor
            except ValueError:
                continue

    print(f"Embeddings carregados: {len(embeddings_dict)} palavras com dimensão {dimensao}.")
    return embeddings_dict, dimensao or 100


def textos_para_vetores_glove(
    textos: list[str],
    glove_dict: dict[str, np.ndarray],
    vector_size: int = 100
) -> np.ndarray:
    """
    Calcula a média dos vetores GloVe de cada texto.
    """
    vetores = []

    for texto in textos:
        palavras = [p for p in texto.split() if p in glove_dict]
        if palavras:
            vetor = np.mean([glove_dict[p] for p in palavras], axis=0)
        else:
            vetor = np.zeros(vector_size)
        vetores.append(vetor)

    return np.array(vetores)


# ---------------------------------------------------------------------------
# Funções de persistência e split
# ---------------------------------------------------------------------------

def ajustar_e_transformar(vetorizador, textos_treino, textos_teste=None):
    """
    Ajusta o vetorizador (fit_transform) no treino e apenas transforma o teste.
    """
    X_treino = vetorizador.fit_transform(textos_treino)
    X_teste = vetorizador.transform(textos_teste) if textos_teste is not None else None
    return X_treino, X_teste


def salvar_vetorizador(vetorizador, caminho: str):
    joblib.dump(vetorizador, caminho)
    print(f"Vetorizador salvo em: {caminho}")


def carregar_vetorizador(caminho: str):
    return joblib.load(caminho)

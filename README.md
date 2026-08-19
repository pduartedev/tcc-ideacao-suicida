# TCC — Detecção de Ideação Suicida com Conjunto de Classificadores

Projeto experimental do Trabalho de Conclusão de Curso (TCC) cujo objetivo é detectar
ideação suicida em textos em português utilizando conjunto de classificadores.

## Pré-processamento

Realizado com **NLTK**, incluindo:
- Limpeza de dados
- Tokenização
- Remoção de stopwords em português (com preservação de negações: *não*, *nunca*, *jamais*, etc.)
- Stemming (RSLPStemmer)

## Vetorização / Representação Textual

Cada modelo é treinado com as quatro representações abaixo:

| Representação | Tipo | Arquivo |
|---|---|---|
| **Bag of Words (BoW)** | Esparsa (contagem) | `data/processed/splits_bow.joblib` |
| **TF-IDF** | Esparsa (ponderada) | `data/processed/splits_tfidf.joblib` |
| **Word2Vec** | Densa (treinada no dataset) | `data/processed/splits_w2v.joblib` |
| **GloVe** | Densa (pré-treinada NILC-USP) | `data/processed/splits_glove.joblib` |

## Matriz de Experimentos: Modelos × Representações

Cada modelo é avaliado nas 4 representações textuais. A célula `✅` indica compatibilidade total e `⚠️` indica que uma variante alternativa é usada.

| Modelo | BoW | TF-IDF | Word2Vec | GloVe |
|---|:---:|:---:|:---:|:---:|
| **Support Vector Machine (SVM)** | ✅ | ✅ | ✅ | ✅ |
| **Naive Bayes** *(MultinomialNB / GaussianNB)* | ✅ MultinomialNB | ✅ MultinomialNB | ⚠️ GaussianNB | ⚠️ GaussianNB |
| **Decision Tree** | ✅ | ✅ | ✅ | ✅ |
| **Random Forest** | ✅ | ✅ | ✅ | ✅ |
| **AdaBoost** | ✅ | ✅ | ✅ | ✅ |
| **Gradient Boosting** | ✅ | ✅ | ✅ | ✅ |
| **XGBoost** | ✅ | ✅ | ✅ | ✅ |
| **Voting Classifier** | ✅ | ✅ | ✅ | ✅ |
| **Stacking Classifier** | ✅ | ✅ | ✅ | ✅ |

> ⚠️ **Naive Bayes**: O `MultinomialNB` exige valores ≥ 0 (compatível com BoW e TF-IDF).
> Para Word2Vec e GloVe (vetores com valores negativos), é utilizado o `GaussianNB`.

Otimização de hiperparâmetros via **GridSearchCV** (scikit-learn) para todos os modelos.

## Estrutura

```
tcc-ideacao-suicida/
├── data/
│   ├── raw/               # Dataset original (Boamente_Atualizado_Janeiro_2026.csv)
│   ├── processed/         # Splits vetorizados (.joblib) e CSV processado
│   └── embeddings/        # Embeddings GloVe NILC-USP (não versionado no git)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # Análise Exploratória dos Dados (EDA)
│   ├── 02_preprocessing.ipynb           # Pré-processamento e limpeza de texto
│   ├── 03_vectorization.ipynb           # Vetorização: BoW, TF-IDF, Word2Vec, GloVe
│   ├── 04_model_training.ipynb          # Classificadores individuais: Naive Bayes, SVM, Decision Tree
│   ├── 05_ensembles.ipynb               # Ensembles base: Random Forest, AdaBoost, Gradient Boosting, XGBoost
│   ├── 06_ensemble_combinados.ipynb     # Ensembles combinados: Voting Classifier, Stacking Classifier
│   └── 07_evaluation.ipynb             # Avaliação e comparação geral de todos os modelos
├── src/
│   ├── preprocessing.py   # Funções de limpeza, tokenização e stemming
│   ├── vectorization.py   # BoW, TF-IDF, Word2Vec, GloVe
│   └── models.py          # Definição e treinamento dos classificadores
├── results/
│   ├── figures/           # Gráficos e visualizações geradas
│   └── metrics/           # Métricas e modelos salvos (.joblib)
├── docs/                  # PDF e documentação do TCC
├── requirements.txt
└── README.md
```

## Configuração do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/pduartedev/tcc-ideacao-suicida.git
cd tcc-ideacao-suicida

# Criar e ativar o ambiente virtual (Python 3.12)
python -m venv .venv
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Registrar o kernel Jupyter
python -m ipykernel install --user --name=tcc-venv --display-name "Python (TCC)"
```

> **Embeddings GloVe:** Baixe o arquivo `glove_s100.zip` do [NILC-USP (Hugging Face)](https://huggingface.co/nilc-nlp/glove-100d)
> e extraia em `data/embeddings/glove_s100.txt`. Este arquivo não está versionado no Git por exceder 100 MB.

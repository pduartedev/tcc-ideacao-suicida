# TCC — Detecção de Ideação Suicida com Conjunto de Classificadores

Projeto experimental do Trabalho de Conclusão de Curso (TCC) cujo objetivo é detectar
ideação suicida em textos em português utilizando um ensemble de classificadores.

## Classificadores

- **Random Forest** (scikit-learn)
- **AdaBoost** (scikit-learn)
- **XGBoost** (xgboost)

Otimização de hiperparâmetros via **GridSearchCV** (scikit-learn).

## Pré-processamento

Realizado com **NLTK**, incluindo:
- Tokenização
- Remoção de stopwords em português
- Stemming (RSLPStemmer)
- Vetorização via TF-IDF

## Estrutura

```
tcc-ideacao-suicida/
├── data/
│   ├── raw/            # Dataset original
│   └── processed/      # Dados pós-processados
├── notebooks/          # Análise exploratória e experimentos
├── src/                # Módulos Python reutilizáveis
├── results/            # Figuras e métricas geradas
├── docs/               # PDF do TCC
├── requirements.txt
└── README.md
```

## Configuração do Ambiente

```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar o ambiente (Windows)
venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar recursos NLTK
python src/setup_nltk.py
```

## Execução

Abra os notebooks na ordem numérica:

1. `01_exploratory_analysis.ipynb` — Análise exploratória do dataset
2. `02_preprocessing.ipynb` — Pré-processamento e vetorização
3. `03_model_training.ipynb` — Treinamento e GridSearch
4. `04_evaluation.ipynb` — Avaliação e comparação dos modelos

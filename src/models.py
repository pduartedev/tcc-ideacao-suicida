"""
models.py
---------
Definição e treinamento dos classificadores do TCC:

Classificadores Individuais:
    - Naive Bayes (MultinomialNB para BoW/TF-IDF | GaussianNB para W2V/GloVe)
    - Support Vector Machine (SVM)
    - Decision Tree

Ensembles:
    - Random Forest
    - AdaBoost
    - Gradient Boosting
    - XGBoost
    - Voting Classifier
    - Stacking Classifier

Inclui função para otimização de hiperparâmetros via GridSearchCV.
"""

import joblib
import numpy as np
from scipy.sparse import issparse

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Classificadores Individuais
# ---------------------------------------------------------------------------

def criar_naive_bayes(representacao: str = "bow"):
    """
    Retorna o Naive Bayes adequado para a representação textual:
      - 'bow' ou 'tfidf' → MultinomialNB (exige valores ≥ 0)
      - 'w2v' ou 'glove'  → GaussianNB  (aceita valores negativos)

    Parâmetros
    ----------
    representacao : str
        Uma das strings: 'bow', 'tfidf', 'w2v', 'glove'.
    """
    representacao = representacao.lower()
    if representacao in ("bow", "tfidf"):
        return MultinomialNB(alpha=1.0)
    else:
        return GaussianNB()


def criar_svm(random_state: int = 42) -> SVC:
    """Retorna um SVC com configuração base (kernel RBF)."""
    return SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        random_state=random_state,
    )


def criar_decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """Retorna um DecisionTreeClassifier com configuração base."""
    return DecisionTreeClassifier(
        max_depth=None,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def criar_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Retorna um RandomForestClassifier com configuração base."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )


def criar_adaboost(random_state: int = 42) -> AdaBoostClassifier:
    """Retorna um AdaBoostClassifier com configuração base."""
    return AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=random_state,
    )


def criar_gradient_boosting(random_state: int = 42) -> GradientBoostingClassifier:
    """Retorna um GradientBoostingClassifier com configuração base."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )


def criar_xgboost(random_state: int = 42) -> XGBClassifier:
    """Retorna um XGBClassifier com configuração base."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def criar_voting_classifier(estimadores: list) -> VotingClassifier:
    """
    Retorna um VotingClassifier (soft voting) com os estimadores fornecidos.

    Parâmetros
    ----------
    estimadores : list of (str, estimador)
        Ex: [('svm', svc), ('rf', rf), ('xgb', xgb)]
    """
    return VotingClassifier(estimators=estimadores, voting="soft", n_jobs=-1)


def criar_stacking_classifier(estimadores: list, meta_modelo=None) -> StackingClassifier:
    """
    Retorna um StackingClassifier com os estimadores e meta-modelo fornecidos.

    Parâmetros
    ----------
    estimadores : list of (str, estimador)
        Classificadores base (nível 0).
    meta_modelo : estimador scikit-learn, opcional
        Meta-classificador (nível 1). Padrão: LogisticRegression.
    """
    if meta_modelo is None:
        meta_modelo = LogisticRegression(max_iter=1000)
    return StackingClassifier(
        estimators=estimadores,
        final_estimator=meta_modelo,
        cv=5,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Grids de hiperparâmetros para GridSearchCV
# ---------------------------------------------------------------------------

PARAMS_NAIVE_BAYES_MULTINOMIAL = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
}

PARAMS_NAIVE_BAYES_GAUSSIAN = {
    "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
}

PARAMS_SVM = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

PARAMS_DECISION_TREE = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}

PARAMS_RANDOM_FOREST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}

PARAMS_ADABOOST = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.5, 1.0, 1.5],
}

PARAMS_GRADIENT_BOOSTING = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5],
}

PARAMS_XGBOOST = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def para_denso(X):
    """Converte matriz esparsa para densa, se necessário (para GaussianNB)."""
    if issparse(X):
        return X.toarray()
    return np.array(X)


def otimizar_modelo(
    modelo,
    param_grid: dict,
    X_treino,
    y_treino,
    cv: int = 5,
    scoring: str = "f1_weighted",
    verbose: int = 1,
) -> GridSearchCV:
    """
    Executa GridSearchCV para encontrar os melhores hiperparâmetros.

    Parâmetros
    ----------
    modelo : estimador scikit-learn compatível
    param_grid : dict
        Grade de hiperparâmetros a testar.
    X_treino : array-like
        Features de treino.
    y_treino : array-like
        Labels de treino.
    cv : int
        Número de folds na validação cruzada (StratifiedKFold).
    scoring : str
        Métrica de avaliação.

    Retorna
    -------
    GridSearchCV ajustado com best_estimator_ disponível.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
    )
    grid.fit(X_treino, y_treino)
    print(f"\nMelhores parâmetros : {grid.best_params_}")
    print(f"Melhor score ({scoring}): {grid.best_score_:.4f}")
    return grid


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def salvar_modelo(modelo, caminho: str):
    """Salva o modelo treinado em disco."""
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")


def carregar_modelo(caminho: str):
    """Carrega um modelo previamente salvo."""
    return joblib.load(caminho)

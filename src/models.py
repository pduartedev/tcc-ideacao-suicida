"""
models.py
---------
Definição e treinamento dos classificadores do ensemble:
    - Random Forest
    - AdaBoost
    - XGBoost

Inclui função para otimização de hiperparâmetros via GridSearchCV.
"""

import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Definição dos modelos com hiperparâmetros padrão iniciais
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
        algorithm="SAMME",
    )


def criar_xgboost(random_state: int = 42) -> XGBClassifier:
    """Retorna um XGBClassifier com configuração base."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Grid de hiperparâmetros para GridSearchCV
# ---------------------------------------------------------------------------

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

PARAMS_XGBOOST = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------

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
    print(f"\nMelhores parâmetros: {grid.best_params_}")
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

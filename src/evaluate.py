"""
evaluate.py
-----------
Funções para avaliação e comparação dos classificadores.

Métricas: Acurácia, Precisão, Recall, F1-Score, ROC-AUC
Visualizações: Matriz de confusão, curva ROC, comparação de modelos
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Diretório padrão para salvar resultados
RESULTS_DIR = Path(__file__).parent.parent / "results"


def calcular_metricas(y_verdadeiro, y_predito, y_proba=None) -> dict:
    """
    Calcula as principais métricas de classificação.

    Retorna um dicionário com acurácia, precisão, recall, f1 e roc_auc.
    """
    metricas = {
        "acuracia": accuracy_score(y_verdadeiro, y_predito),
        "precisao": precision_score(y_verdadeiro, y_predito, average="weighted", zero_division=0),
        "recall": recall_score(y_verdadeiro, y_predito, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_verdadeiro, y_predito, average="weighted", zero_division=0),
    }
    if y_proba is not None:
        try:
            metricas["roc_auc"] = roc_auc_score(y_verdadeiro, y_proba)
        except ValueError:
            metricas["roc_auc"] = None
    return metricas


def relatorio_completo(y_verdadeiro, y_predito, nome_modelo: str = ""):
    """Imprime o classification_report completo do sklearn."""
    print(f"\n{'='*50}")
    print(f"Relatório: {nome_modelo}")
    print("="*50)
    print(classification_report(y_verdadeiro, y_predito, zero_division=0))


def plotar_matriz_confusao(y_verdadeiro, y_predito, nome_modelo: str = "", salvar: bool = True):
    """Plota e opcionalmente salva a matriz de confusão."""
    cm = confusion_matrix(y_verdadeiro, y_predito)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusão — {nome_modelo}")
    plt.tight_layout()
    if salvar:
        path = RESULTS_DIR / "figures" / f"confusion_{nome_modelo.lower().replace(' ', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Figura salva: {path}")
    plt.show()


def plotar_curva_roc(y_verdadeiro, y_proba, nome_modelo: str = "", salvar: bool = True):
    """Plota e opcionalmente salva a curva ROC."""
    fpr, tpr, _ = roc_curve(y_verdadeiro, y_proba)
    auc = roc_auc_score(y_verdadeiro, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("Taxa de Falso Positivo")
    ax.set_ylabel("Taxa de Verdadeiro Positivo")
    ax.set_title(f"Curva ROC — {nome_modelo}")
    ax.legend()
    plt.tight_layout()
    if salvar:
        path = RESULTS_DIR / "figures" / f"roc_{nome_modelo.lower().replace(' ', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Figura salva: {path}")
    plt.show()


def comparar_modelos(resultados: dict, salvar: bool = True):
    """
    Plota um gráfico de barras comparando métricas entre modelos.

    Parâmetros
    ----------
    resultados : dict
        Formato: {"NomeModelo": {"acuracia": 0.9, "f1_weighted": 0.89, ...}, ...}
    """
    modelos = list(resultados.keys())
    metricas_plot = ["acuracia", "precisao", "recall", "f1_weighted"]
    x = np.arange(len(modelos))
    largura = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metrica in enumerate(metricas_plot):
        valores = [resultados[m].get(metrica, 0) for m in modelos]
        ax.bar(x + i * largura, valores, largura, label=metrica.replace("_", " ").title())

    ax.set_xticks(x + largura * 1.5)
    ax.set_xticklabels(modelos)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos")
    ax.legend()
    plt.tight_layout()
    if salvar:
        path = RESULTS_DIR / "figures" / "comparacao_modelos.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Figura salva: {path}")
    plt.show()


def salvar_metricas(resultados: dict, nome_arquivo: str = "metricas.json"):
    """Salva o dicionário de métricas em JSON."""
    path = RESULTS_DIR / "metrics" / nome_arquivo
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"Métricas salvas em: {path}")

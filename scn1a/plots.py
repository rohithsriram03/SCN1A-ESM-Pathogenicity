from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

PATHOGENIC = "#c1121f"
BENIGN = "#0353a4"


def set_style():
    """Apply the project's publication figure style. Call once at script start."""
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def save(fig, path) -> Path:
    """Write `fig` to `path` at 300 dpi and close it."""
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def roc(y_true, y_prob, label: str = "model", ax=None):
    """ROC curve with AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.plot(fpr, tpr, color=PATHOGENIC, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    return ax.figure


def confusion(y_true, y_pred, labels=("Benign", "Pathogenic")):
    """Annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
                cmap="rocket_r", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True")
    return fig


def boxplot_by_group(df, value: str, group: str, order=None, ax=None):
    """Distribution of `value` per `group`, with points overlaid."""
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    sns.boxplot(data=df, x=group, y=value, order=order, ax=ax, color="#cdd5df")
    sns.stripplot(data=df, x=group, y=value, order=order, ax=ax,
                  color=PATHOGENIC, size=3, alpha=0.4)
    ax.set(xlabel="", ylabel=value.replace("_", " "))
    return ax.figure


def metric_bars(metrics_df, index: str = "model", ax=None):
    """Grouped bar chart of a metrics table (one row per model)."""
    data = metrics_df.set_index(index)
    ax = data.plot.bar(rot=0, colormap="rocket", ax=ax, figsize=(8, 5))
    ax.set(ylim=(0, 1), ylabel="score")
    ax.legend(loc="lower right", ncol=len(data.columns))
    return ax.figure


def embedding_scatter(X, labels, ax=None):
    """2-D PCA projection of embeddings coloured by binary label."""
    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    labels = np.asarray(labels)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    for value, color, name in [(0, BENIGN, "Benign"), (1, PATHOGENIC, "Pathogenic")]:
        m = labels == value
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.6, color=color, label=name)
    ax.set(xlabel="PC1", ylabel="PC2")
    ax.legend()
    return ax.figure

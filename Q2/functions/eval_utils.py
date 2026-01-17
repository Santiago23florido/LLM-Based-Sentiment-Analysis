"""Evaluation helper reused across Q1/Q2 notebooks.

This function generates a 3-row report:
1) Metrics overview
2) Confusion matrix
3) Multiclass ROC curves (OvR)

All comments are in English as requested.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    log_loss,
    cohen_kappa_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(y_true, y_pred, y_probs, model_name, output_path="reports/"):
    """Evaluate a multi-class classifier and export a visual report.

    Parameters
    ----------
    y_true: array-like
        Ground-truth labels.
    y_pred: array-like
        Predicted labels.
    y_probs: ndarray
        Predicted probabilities with shape (n_samples, n_classes).
    model_name: str
        Model label used in plot title and exported filename.
    output_path: str
        Directory where the PNG report will be saved.
    """

    os.makedirs(output_path, exist_ok=True)

    classes = np.unique(y_true)
    n_classes = len(classes)

    # 1) Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "ROC-AUC (Weighted)": roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted'),
        "Log Loss": log_loss(y_true, y_probs),
    }

    report = classification_report(y_true, y_pred, output_dict=True)
    avg_metrics = {
        "Precision (macro)": report["macro avg"]["precision"],
        "Recall (macro)": report["macro avg"]["recall"],
        "F1 (macro)": report["macro avg"]["f1-score"],
        "Precision (weighted)": report["weighted avg"]["precision"],
        "Recall (weighted)": report["weighted avg"]["recall"],
        "F1 (weighted)": report["weighted avg"]["f1-score"],
    }

    # --- Visualization (3 Rows) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 22))
    fig.suptitle(f"Model Performance Report: {model_name}", fontsize=18, fontweight='bold')

    # Row 1: Metric overview
    all_stats = {**metrics, **avg_metrics}
    plot_data = {k: v for k, v in all_stats.items() if k != "Log Loss"}

    sns.barplot(x=list(plot_data.values()), y=list(plot_data.keys()), ax=axes[0])
    axes[0].set_xlim(0, 1.1)
    axes[0].set_title('Metric Overview (Higher is Better)', fontsize=14)
    for i, v in enumerate(plot_data.values()):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

    # Row 2: Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 14})
    axes[1].set_title('Confusion Matrix', fontsize=14)
    axes[1].set_ylabel('Actual Label')
    axes[1].set_xlabel('Predicted Label')

    # Row 3: Multiclass ROC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        axes[2].plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc_val:.2f})')

    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('Receiver Operating Characteristic (ROC) - Multiclass', fontsize=14)
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    save_path = f"{output_path}/{model_name.replace(' ', '_')}_Full_Report.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Full report exported successfully: {save_path}")

    plt.show()

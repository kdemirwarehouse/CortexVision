# evaluation.py

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, "/content/drive/MyDrive/CortexVision")

from cortex_config import (
    DEVICE,
    CKPT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    CLASSES,
    NUM_CLASSES,
)

from dataset import test_loader
from model import CortexVisionModel


def load_model(checkpoint_path, device):
    """En iyi checkpoint'i yukler ve modeli eval moduna alir."""
    model = CortexVisionModel().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint yuklendi: epoch {checkpoint['epoch']}")
    print(f"En iyi val accuracy: {checkpoint['best_val_acc']:.4f}")

    return model


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Test seti uzerinden tahminleri ve olasiliklari toplar."""
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Confusion matrix grafigini kaydeder."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gercek")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return cm


def plot_roc_curves(y_true, y_probs, classes, save_path):
    """Her sinif icin ROC egrisi cizer."""
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))

    plt.figure(figsize=(9, 7))

    colors = ["#534AB7", "#0F6E56", "#993C1D", "#854F0B"]
    auc_scores = {}

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc

        plt.plot(
            fpr, tpr,
            color=color, linewidth=2.2,
            label=f"{cls}  (AUC = {roc_auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Test Set (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return auc_scores


def save_metrics_report(y_true, y_pred, y_probs, classes, auc_scores, save_path):
    """Tum metrikleri JSON olarak kaydeder."""
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(classes)))
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    report = {
        "overall": {
            "accuracy": round(float(accuracy), 6),
            "macro_precision": round(float(macro_p), 6),
            "macro_recall": round(float(macro_r), 6),
            "macro_f1": round(float(macro_f1), 6),
        },
        "per_class": {},
    }

    for i, cls in enumerate(classes):
        report["per_class"][cls] = {
            "precision": round(float(precision[i]), 6),
            "recall":    round(float(recall[i]), 6),
            "f1":        round(float(f1[i]), 6),
            "auc":       round(float(auc_scores[cls]), 6),
            "support":   int(support[i]),
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return report


def print_metrics_summary(report, classes):
    """Sonuclari ekrana yazdirir."""
    print("\n" + "=" * 70)
    print("GENEL SONUCLAR")
    print("=" * 70)
    print(f"Test Accuracy        : {report['overall']['accuracy']:.4f}")
    print(f"Macro Precision      : {report['overall']['macro_precision']:.4f}")
    print(f"Macro Recall         : {report['overall']['macro_recall']:.4f}")
    print(f"Macro F1             : {report['overall']['macro_f1']:.4f}")

    print("\n" + "=" * 70)
    print("SINIF BAZINDA SONUCLAR")
    print("=" * 70)
    print(f"{'Sinif':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Support':>10}")
    print("-" * 70)
    for cls in classes:
        m = report["per_class"][cls]
        print(f"{cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['auc']:>10.4f} {m['support']:>10}")


def main():
    device = torch.device(DEVICE)
    use_amp = device.type == "cuda"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt_path  = CKPT_DIR / "best.pt"
    cm_path         = FIGURE_DIR / "confusion_matrix.png"
    roc_path        = FIGURE_DIR / "roc_curves.png"
    metrics_path    = RESULT_DIR / "test_metrics.json"

    print(f"Device: {device}")
    print(f"Test batches: {len(test_loader)}")

    model = load_model(best_ckpt_path, device)

    y_true, y_pred, y_probs = evaluate(model, test_loader, device, use_amp)

    cm = plot_confusion_matrix(y_true, y_pred, CLASSES, cm_path)
    auc_scores = plot_roc_curves(y_true, y_probs, CLASSES, roc_path)

    report = save_metrics_report(
        y_true, y_pred, y_probs, CLASSES, auc_scores, metrics_path
    )

    print_metrics_summary(report, CLASSES)

    print(f"\nFigurler:")
    print(f"  Confusion matrix : {cm_path}")
    print(f"  ROC curves       : {roc_path}")
    print(f"\nMetrikler        : {metrics_path}")


if __name__ == "__main__":
    main()



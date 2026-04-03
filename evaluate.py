import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
from typing import Dict

from config import CFG
from data.dataset import generate_synthetic_dataset, split_records, get_dataloaders, NewsDataset, get_image_transform
from models.detector import FakeNewsDetector
from xai.explainer import UnifiedExplainer
from train import evaluate_detailed, get_device


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6366F1", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: Dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#6366F1", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#f59e0b", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(epochs, history["train_f1"], label="Train", color="#6366F1", lw=2)
    axes[1].plot(epochs, history["val_f1"], label="Val", color="#f59e0b", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation(ablation_results: Dict, save_path: str):
    configs = list(ablation_results.keys())
    acc = [ablation_results[c]["accuracy"] for c in configs]
    f1 = [ablation_results[c]["f1_macro"] for c in configs]
    auc_vals = [ablation_results[c]["auc_roc"] for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, acc, width, label="Accuracy", color="#6366F1")
    ax.bar(x, f1, width, label="F1 Macro", color="#22c55e")
    ax.bar(x + width, auc_vals, width, label="AUC-ROC", color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Contribution of Each Modality")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate():
    device = get_device()
    checkpoint_path = CFG.app.model_checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint at {checkpoint_path}. Run train.py first.")
        return

    output_dir = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_saved = ckpt.get("config", {})

    tokenizer = AutoTokenizer.from_pretrained(CFG.model.text_encoder)
    model = FakeNewsDetector(
        use_text=cfg_saved.get("use_text", True),
        use_image=cfg_saved.get("use_image", True),
        use_social=cfg_saved.get("use_social", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']} (best val F1: {ckpt['val_f1']:.4f})")

    from data.dataset import load_fakenews_hf
    records = load_fakenews_hf()
    records_train, records_val, records_test = split_records(records)
    _, _, test_loader = get_dataloaders(records_train, records_val, records_test, tokenizer)

    print("Running evaluation...")
    metrics = evaluate_detailed(model, test_loader, device)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro : {metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

    plot_confusion_matrix(
        metrics["ground_truth"], metrics["predictions"],
        os.path.join(output_dir, "confusion_matrix.png")
    )
    print("Saved confusion matrix")

    log_path = os.path.join(CFG.train.log_dir, "training_results.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)
        plot_training_curves(
            training_log["history"],
            os.path.join(output_dir, "training_curves.png")
        )
        print("Saved training curves")

    ablation_path = os.path.join(CFG.train.log_dir, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)
        plot_ablation(ablation, os.path.join(output_dir, "ablation_study.png"))
        print("Saved ablation chart")

    print("\nGenerating XAI explanations for 3 test samples...")
    explainer = UnifiedExplainer(model, tokenizer, device)
    test_ds = NewsDataset(records_test[:5], tokenizer, get_image_transform(train=False))

    for i in range(min(3, len(test_ds))):
        sample = test_ds[i]
        batch = {
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
            "image": sample["image"].unsqueeze(0),
            "graph_x": [sample["graph_x"]],
            "graph_edge_index": [sample["graph_edge_index"]],
            "graph_num_nodes": [sample["graph_num_nodes"]],
        }
        explanation = explainer.explain_prediction(batch, news_id=f"sample_{i}")
        report_path = explainer.generate_report(explanation, news_id=f"sample_{i}", save_dir=output_dir)
        print(f"  Sample {i}: {explanation['prediction']} ({explanation['confidence']:.1%}) → {report_path}")

    print(f"\nAll evaluation outputs saved to: {output_dir}/")


if __name__ == "__main__":
    evaluate()

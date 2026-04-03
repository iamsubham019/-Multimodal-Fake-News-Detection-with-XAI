import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
from typing import Dict, Tuple, List
import time

from config import CFG
from data.dataset import (
    load_fakenewsnet,
    generate_synthetic_dataset,
    load_fakenews_hf,
    split_records,
    get_dataloaders,
)
from models.detector import FakeNewsDetector


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: FakeNewsDetector,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    is_train: bool,
) -> Tuple[float, float]:
    model.train(is_train)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(loader, desc="Train" if is_train else "Eval", leave=False)

    with context:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            image = batch["image"].to(device) if "image" in batch else None
            graph_x = batch.get("graph_x")
            graph_edge_index = batch.get("graph_edge_index")
            graph_num_nodes = batch.get("graph_num_nodes")

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                graph_num_nodes=graph_num_nodes,
            )

            loss = model.get_loss(
                out["logits"], labels, out.get("inconsistency_score")
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG.train.gradient_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            preds = out["probabilities"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


def evaluate_detailed(
    model: FakeNewsDetector,
    loader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            image = batch["image"].to(device) if "image" in batch else None

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                graph_x=batch.get("graph_x"),
                graph_edge_index=batch.get("graph_edge_index"),
                graph_num_nodes=batch.get("graph_num_nodes"),
            )

            preds = out["probabilities"].argmax(dim=-1).cpu().numpy()
            probs = out["probabilities"][:, 1].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_fake": f1_score(all_labels, all_preds, pos_label=1, average="binary", zero_division=0),
        "f1_real": f1_score(all_labels, all_preds, pos_label=0, average="binary", zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "report": classification_report(all_labels, all_preds, target_names=["Real", "Fake"], zero_division=0),
        "predictions": all_preds,
        "ground_truth": all_labels,
    }


def run_ablation(
    records_train, records_val, records_test,
    tokenizer, device
) -> Dict:
    configs = [
        ("Text only", True, False, False),
        ("Text + Social", True, False, True),
        ("Full (Text+Social)", True, False, True),
    ]

    results = {}
    for name, use_text, use_image, use_social in configs:
        print(f"\n  Running ablation: {name}")
        train_loader, val_loader, test_loader = get_dataloaders(
            records_train, records_val, records_test, tokenizer
        )
        model = FakeNewsDetector(use_text=use_text, use_image=use_image, use_social=use_social).to(device)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=CFG.train.learning_rate, weight_decay=CFG.train.weight_decay,
        )
        for epoch in range(min(3, CFG.train.num_epochs)):
            run_epoch(model, train_loader, optimizer, None, device, is_train=True)

        metrics = evaluate_detailed(model, test_loader, device)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "auc_roc": metrics["auc_roc"],
        }
        print(f"  {name}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f}")

    return results


def train():
    set_seed(CFG.train.seed)
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(CFG.train.save_dir, exist_ok=True)
    os.makedirs(CFG.train.log_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.text_encoder)

    print("Loading dataset...")
    from data.dataset import load_fakenews_hf
    records = load_fakenews_hf()

    records_train, records_val, records_test = split_records(records)
    print(f"Train: {len(records_train)} | Val: {len(records_val)} | Test: {len(records_test)}")

    train_loader, val_loader, test_loader = get_dataloaders(
        records_train, records_val, records_test, tokenizer
    )

    print("Building model...")
    model = FakeNewsDetector(
        use_text=CFG.train.use_text,
        use_image=CFG.train.use_image,
        use_social=CFG.train.use_social,
    ).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=CFG.train.learning_rate,
        weight_decay=CFG.train.weight_decay,
    )

    total_steps = len(train_loader) * CFG.train.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CFG.train.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
    )

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = 0.0
    patience_counter = 0

    print("\nStarting training...\n")
    for epoch in range(1, CFG.train.num_epochs + 1):
        t_start = time.time()

        train_loss, train_f1 = run_epoch(
            model, train_loader, optimizer, scheduler, device, is_train=True
        )
        val_loss, val_f1 = run_epoch(
            model, val_loader, optimizer, None, device, is_train=False
        )

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:02d}/{CFG.train.num_epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint_path = os.path.join(CFG.train.save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "config": {
                    "use_text": CFG.train.use_text,
                    "use_image": CFG.train.use_image,
                    "use_social": CFG.train.use_social,
                },
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG.train.early_stopping_patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(os.path.join(CFG.train.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nFinal test evaluation:")
    test_metrics = evaluate_detailed(model, test_loader, device)
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro : {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Fake  : {test_metrics['f1_fake']:.4f}")
    print(f"  AUC-ROC  : {test_metrics['auc_roc']:.4f}")
    print("\nClassification Report:")
    print(test_metrics["report"])

    log_path = os.path.join(CFG.train.log_dir, "training_results.json")
    with open(log_path, "w") as f:
        json.dump({
            "history": history,
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "report"},
            "best_val_f1": best_val_f1,
        }, f, indent=2)
    print(f"\nResults saved to {log_path}")

    print("\nRunning ablation study...")
    ablation_results = run_ablation(records_train, records_val, records_test, tokenizer, device)
    ablation_path = os.path.join(CFG.train.log_dir, "ablation_results.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Ablation results saved to {ablation_path}")


if __name__ == "__main__":
    train()

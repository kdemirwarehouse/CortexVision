# train.py

import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# Colab Drive proje yolu
sys.path.insert(0, "/content/drive/MyDrive/CortexVision")

from cortex_config import (
    DEVICE,
    CKPT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    PATIENCE,
    RANDOM_SEED,
    CLASS_WEIGHTS,
)

from dataset import train_loader, val_loader
from model import CortexVisionModel


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik için seed ayarları."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs, targets):
    """Batch accuracy hesaplar."""
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()

    running_loss = 0.0
    correct_total = 0
    sample_total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        correct, total = calculate_accuracy(outputs, labels)
        correct_total += correct
        sample_total += total

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct_total / sample_total:.4f}"
        })

    epoch_loss = running_loss / sample_total
    epoch_acc = correct_total / sample_total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()

    running_loss = 0.0
    correct_total = 0
    sample_total = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        correct, total = calculate_accuracy(outputs, labels)
        correct_total += correct
        sample_total += total

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct_total / sample_total:.4f}"
        })

    epoch_loss = running_loss / sample_total
    epoch_acc = correct_total / sample_total

    return epoch_loss, epoch_acc


def save_checkpoint(path, model, optimizer, epoch, best_val_acc):
    """Model checkpoint kaydeder."""
    checkpoint = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def save_history(history, save_path):
    """Eğitim geçmişini JSON olarak kaydeder."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def plot_training_curves(history, save_path):
    """Loss ve accuracy eğrilerini kaydeder."""
    epochs = [item["epoch"] for item in history]

    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    train_accs = [item["train_acc"] for item in history]
    val_accs = [item["val_acc"] for item in history]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    set_seed(RANDOM_SEED)

    device = torch.device(DEVICE)
    use_amp = device.type == "cuda"

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    last_ckpt_path = CKPT_DIR / "last.pt"
    best_ckpt_path = CKPT_DIR / "best.pt"
    history_path = RESULT_DIR / "train_history.json"
    curve_path = FIGURE_DIR / "training_curves.png"

    print(f"Device: {device}")
    print(f"Mixed Precision: {use_amp}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model = CortexVisionModel().to(device)

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    scaler = GradScaler(enabled=use_amp)

    history = []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")
        print(f"Learning Rate: {current_lr:.8f}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        scheduler.step()

        epoch_time = time.time() - start_time

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "learning_rate": current_lr,
            "epoch_time_sec": round(epoch_time, 2),
        }

        history.append(epoch_log)
        save_history(history, history_path)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Her epoch sonunda son model kaydedilir
        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc
        )

        # En iyi model sadece val_acc artarsa kaydedilir
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc
            )

            print(f"Best checkpoint saved. Best Val Acc: {best_val_acc:.4f}")

        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    plot_training_curves(history, curve_path)

    print("\nTraining completed.")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Training history: {history_path}")
    print(f"Training curves: {curve_path}")


if __name__ == "__main__":
    main()
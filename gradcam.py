# gradcam.py

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, "/content/drive/MyDrive/CortexVision")

from cortex_config import (
    DEVICE,
    CKPT_DIR,
    FIGURE_DIR,
    CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    DATA_DIR,
)

from model import CortexVisionModel


def load_model(device):
    """best.pt yukler."""
    model = CortexVisionModel().to(device)
    checkpoint = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model yuklendi: epoch {checkpoint['epoch']}, val_acc {checkpoint['best_val_acc']:.4f}")
    return model


def get_target_layer(model):
    """GradCAM++ icin hedef katmani dondurur."""
    # EfficientNet-B0'in son conv blogu
    return [model.efficientnet.conv_head]


def get_transforms():
    """Goruntu donusturuculeri dondurur."""
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    return preprocess


def load_and_preprocess(img_path, preprocess, device):
    """Tek bir goruntuyu yukler ve modele hazirlar."""
    img_pil = Image.open(img_path).convert("RGB")

    # Normalize edilmemis versiyon (gorsellestrme icin)
    img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img_resized) / 255.0

    # Model icin normalize edilmis versiyon
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    return input_tensor, img_np


def find_samples(model, device, preprocess, n_per_class=3):
    """Her sinif icin dogru ve yanlis tahmin edilen ornekler bulur."""
    import os

    test_dir = DATA_DIR / "Testing"
    correct_samples = {cls: [] for cls in CLASSES}
    wrong_samples = {cls: [] for cls in CLASSES}

    for cls_idx, cls_name in enumerate(CLASSES):
        folder = test_dir / cls_name
        if not folder.exists():
            # Kaggle'da 'notumor' olarak isimlendirilmis olabilir
            alt_name = cls_name.replace("_", "")
            folder = test_dir / alt_name
            if not folder.exists():
                print(f"Klasor bulunamadi: {cls_name}")
                continue

        files = sorted(os.listdir(folder))

        for fname in files:
            if len(correct_samples[cls_name]) >= n_per_class and len(wrong_samples[cls_name]) >= n_per_class:
                break

            fpath = folder / fname
            try:
                input_tensor, img_np = load_and_preprocess(str(fpath), preprocess, device)
            except:
                continue

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            if pred == cls_idx and len(correct_samples[cls_name]) < n_per_class:
                correct_samples[cls_name].append({
                    "path": str(fpath),
                    "img_np": img_np,
                    "input_tensor": input_tensor,
                    "pred": pred,
                    "true": cls_idx,
                    "fname": fname,
                })
            elif pred != cls_idx and len(wrong_samples[cls_name]) < n_per_class:
                wrong_samples[cls_name].append({
                    "path": str(fpath),
                    "img_np": img_np,
                    "input_tensor": input_tensor,
                    "pred": pred,
                    "true": cls_idx,
                    "fname": fname,
                })

    return correct_samples, wrong_samples


def generate_gradcam(cam, input_tensor, img_np, target_class=None):
    """Tek bir goruntu icin GradCAM++ isi haritasi uretir."""
    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam


def plot_correct_samples(correct_samples, cam, save_path):
    """Dogru tahmin edilen orneklerin GradCAM++ gorsellerini kaydeder."""
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(12, 16))
    fig.suptitle("GradCAM++ - Dogru Tahminler", fontsize=16, fontweight="bold", y=0.98)

    for row, cls_name in enumerate(CLASSES):
        samples = correct_samples[cls_name]
        for col in range(3):
            ax = axes[row][col]
            if col < len(samples):
                s = samples[col]
                vis, _ = generate_gradcam(cam, s["input_tensor"], s["img_np"], s["true"])
                ax.imshow(vis)
                ax.set_title(f"Gercek: {cls_name}\nTahmin: {CLASSES[s['pred']]}", fontsize=9)
            ax.axis("off")

        # Sol tarafa sinif etiketi
        axes[row][0].set_ylabel(cls_name, fontsize=12, fontweight="bold", rotation=90, labelpad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dogru tahmin figurleri kaydedildi: {save_path}")


def plot_wrong_samples(wrong_samples, cam, save_path):
    """Yanlis tahmin edilen orneklerin GradCAM++ gorsellerini kaydeder."""
    # Kac sinifta yanlis ornek var?
    active_classes = [cls for cls in CLASSES if len(wrong_samples[cls]) > 0]

    if not active_classes:
        print("Hicbir sinifta yanlis tahmin bulunamadi.")
        return

    n_rows = len(active_classes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    fig.suptitle("GradCAM++ - Yanlis Tahminler", fontsize=16, fontweight="bold", y=0.98)

    if n_rows == 1:
        axes = [axes]

    for row, cls_name in enumerate(active_classes):
        samples = wrong_samples[cls_name]
        for col in range(3):
            ax = axes[row][col]
            if col < len(samples):
                s = samples[col]
                vis, _ = generate_gradcam(cam, s["input_tensor"], s["img_np"], s["pred"])
                ax.imshow(vis)
                ax.set_title(
                    f"Gercek: {cls_name}\nTahmin: {CLASSES[s['pred']]}",
                    fontsize=9, color="red"
                )
            ax.axis("off")

        axes[row][0].set_ylabel(cls_name, fontsize=12, fontweight="bold", rotation=90, labelpad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Yanlis tahmin figurleri kaydedildi: {save_path}")


def plot_comparison_grid(correct_samples, wrong_samples, cam, save_path):
    """Glioma icin dogru vs yanlis karsilastirma gridi."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    fig.suptitle("GradCAM++ Karsilastirma: Glioma (Dogru vs Yanlis)",
                 fontsize=14, fontweight="bold", y=0.98)

    # Ust satir: dogru tahminler
    for col in range(3):
        ax = axes[0][col]
        if col < len(correct_samples["glioma"]):
            s = correct_samples["glioma"][col]
            vis, _ = generate_gradcam(cam, s["input_tensor"], s["img_np"], s["true"])
            ax.imshow(vis)
            ax.set_title(f"Dogru: glioma → glioma", fontsize=10, color="green")
        ax.axis("off")
    axes[0][0].set_ylabel("Dogru\nTahmin", fontsize=12, fontweight="bold", rotation=90, labelpad=15)

    # Alt satir: yanlis tahminler
    for col in range(3):
        ax = axes[1][col]
        if col < len(wrong_samples["glioma"]):
            s = wrong_samples["glioma"][col]
            vis, _ = generate_gradcam(cam, s["input_tensor"], s["img_np"], s["pred"])
            ax.imshow(vis)
            ax.set_title(
                f"Yanlis: glioma → {CLASSES[s['pred']]}",
                fontsize=10, color="red"
            )
        ax.axis("off")
    axes[1][0].set_ylabel("Yanlis\nTahmin", fontsize=12, fontweight="bold", rotation=90, labelpad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Glioma karsilastirma figurleri kaydedildi: {save_path}")


def main():
    device = torch.device(DEVICE)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Model yukleniyor...")
    model = load_model(device)

    print("Hedef katman belirleniyor...")
    target_layers = get_target_layer(model)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    preprocess = get_transforms()

    print("Ornekler seciliyor...")
    correct_samples, wrong_samples = find_samples(model, device, preprocess, n_per_class=3)

    # Bulunan ornek sayilari
    for cls in CLASSES:
        print(f"  {cls}: {len(correct_samples[cls])} dogru, {len(wrong_samples[cls])} yanlis")

    print("\nGradCAM++ gorselleri uretiliyor...")

    # 1. Dogru tahminler gridi
    plot_correct_samples(
        correct_samples, cam,
        FIGURE_DIR / "gradcam_correct.png"
    )

    # 2. Yanlis tahminler gridi
    plot_wrong_samples(
        wrong_samples, cam,
        FIGURE_DIR / "gradcam_wrong.png"
    )

    # 3. Glioma karsilastirma
    plot_comparison_grid(
        correct_samples, wrong_samples, cam,
        FIGURE_DIR / "gradcam_glioma_comparison.png"
    )

    print("\nTum GradCAM++ gorselleri kaydedildi!")
    print(f"Klasor: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
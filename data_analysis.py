from google.colab import drive
drive.mount("/content/drive")

import os
import matplotlib.pyplot as plt

base = "/content/drive/MyDrive/CortexVision/data/brain-tumor-mri"

splits = {"Training": {}, "Testing": {}}
for split in splits:
  for cls in os.listdir(f"{base}/{split}"):
    count = len(os.listdir(f"{base}/{split}/{cls}"))
    splits[split][cls] = count
    print(f"{split} / {cls}: {count} goruntu")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, data) in zip(axes, splits.items()):
    ax.bar(data.keys(), data.values(), color=["#534AB7","#0F6E56","#993C1D","#854F0B"])
    ax.set_title(split)
    ax.set_ylabel("Görüntü sayısı")
    for i, (k, v) in enumerate(data.items()):
        ax.text(i, v + 10, str(v), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/CortexVision/figures/class_distribution.png", dpi=150)
plt.show()
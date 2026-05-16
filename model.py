import torch
import torch.nn as nn
import timm
import sys
sys.path.insert(0, '/content/drive/MyDrive/CortexVision')
from cortex_config import (
    NUM_CLASSES, EFF_MODEL, SWIN_MODEL,
    FUSION_DIM, NUM_HEADS, DROPOUT
)


class CortexVisionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.efficientnet = timm.create_model(
            EFF_MODEL, pretrained=True, num_classes=0
        )
        eff_dim = self.efficientnet.num_features

        self.swin = timm.create_model(
            SWIN_MODEL, pretrained=True, num_classes=0
        )
        swin_dim = self.swin.num_features

        self.eff_proj  = nn.Linear(eff_dim, FUSION_DIM)
        self.swin_proj = nn.Linear(swin_dim, FUSION_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        eff  = self.eff_proj(self.efficientnet(x))
        swin = self.swin_proj(self.swin(x))

        combined = torch.stack([eff, swin], dim=1)
        fused, _ = self.attention(combined, combined, combined)
        fused = fused.mean(dim=1)

        logits = self.classifier(fused)
        return logits
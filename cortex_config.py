import torch
from pathlib import Path

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DRIVE_ROOT    = Path("/content/drive/MyDrive/CortexVision")
DATA_DIR      = DRIVE_ROOT / "data"/"brain-tumor-mri"
FIGSHARE_DIR  = DRIVE_ROOT / "data"
DATA_DIR      = DRIVE_ROOT / "data" / "brain-tumor-mri"
FIGSHARE_DIR  = DRIVE_ROOT / "data" / "figshare"
CKPT_DIR      = DRIVE_ROOT / "checkpoints"
RESULT_DIR    = DRIVE_ROOT / "results"
FIGURE_DIR    = DRIVE_ROOT / "figures"

CLASSES       = ["glioma", "meningioma", "pituitary", "no_tumor"]
NUM_CLASSES   = len(CLASSES)
CLASS_TO_IDX  = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS  = {i: cls for i, cls in enumerate(CLASSES)}

EFF_MODEL     = "efficientnet_b0"
SWIN_MODEL    = "swin_tiny_patch4_window7_224"
FUSION_DIM    = 512
NUM_HEADS     = 8
DROPOUT       = 0.3

GRADCAM_LAYER = "efficientnet.blocks"

IMAGE_SIZE    = 224
BATCH_SIZE    = 64
NUM_EPOCHS    = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-2
PATIENCE      = 4
RANDOM_SEED   = 42

CLASS_WEIGHTS = [1.2, 1.5, 1.0, 1.0]

TRAIN_RATIO   = 0.85
VAL_RATIO     = 0.15

NORM_MEAN     = [0.485, 0.456, 0.406]
NORM_STD      = [0.229, 0.224, 0.225]
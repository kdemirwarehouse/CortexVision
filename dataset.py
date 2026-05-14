from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import sys
sys.path.insert(0,'/content/drive/MyDrive/CortexVision')
from cortex_config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE,
    TRAIN_RATIO, RANDOM_SEED, NORM_MEAN, NORM_STD
)

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

train_val_dataset = datasets.ImageFolder(
    root=str(DATA_DIR / "Training"),
    transform=train_transforms
)

test_dataset = datasets.ImageFolder(
    root=str(DATA_DIR / "Testing"),
    transform=val_test_transforms
)

train_size = int(TRAIN_RATIO * len(train_val_dataset))
val_size   = len(train_val_dataset) - train_size
generator  = torch.Generator().manual_seed(RANDOM_SEED)

train_dataset, val_dataset = random_split(
    train_val_dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

images, labels = next(iter(train_loader))
print(f"Batch boyutu : {images.shape}")
print(f"Etiketler    : {labels}")
print(f"Train size   : {len(train_dataset)}")
print(f"Val size     : {len(val_dataset)}")
print(f"Test size    : {len(test_dataset)}")
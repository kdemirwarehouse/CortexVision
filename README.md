# CortexVision

Hybrid deep learning model for brain tumor detection and classification from MRI images.
EfficientNet-B4 and Swin Transformer with Attention-Based Intermediate Fusion.
Explainability via GradCAM++ and SHAP.

## Classes

| Index | Class      |
|-------|------------|
| 0     | glioma     |
| 1     | meningioma |
| 2     | pituitary  |
| 3     | no_tumor   |

## Dataset

- Training: Kaggle Brain Tumor MRI Dataset (7024 images, 4 classes)
- Cross-validation: Figshare CE-MRI Dataset (3064 images, 3 classes)

## Architecture

```
MRI Input
    |
    +---> [EfficientNet-B4]  -->  local features  (512 dim)
    |
    +---> [Swin Transformer] -->  global context  (512 dim)
                                        |
                              [Attention Fusion]
                                        |
                              [Classifier Head]  -->  4 classes
                                        |
                         [GradCAM++] + [SHAP]   -->  explainability
```

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
CortexVision/
├── config.py
├── dataset.py
├── model.py
├── train.py
├── gradcam.py
├── shap_analysis.py
├── app.py
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## Drive Structure

```
CortexVision/
├── data/
│   ├── kaggle/
│   └── figshare/
├── checkpoints/
├── results/
└── figures/
```

## References

- Cheng J. et al. PLoS ONE, 2015. (Figshare dataset)
- Masoud Nickparvar. Kaggle Brain Tumor MRI Dataset, 2021.

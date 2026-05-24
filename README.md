# 🦌 WildEye — Robust Wildlife Classifier

> **Mini Hackathon #1: How Can Machines See What Matters?**  
> Transfer Learning + Domain Augmentation for Camera-Trap Animal Classification

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/zkmine/wildeye)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)

---

## 🎯 Problem Statement

Camera-trap models trained on clean data fail in the real world. Wildlife cameras operate in conditions that standard training data never captures:

- 🌙 **Night vision** — cameras auto-switch to infrared after dark
- 💨 **Motion blur** — animals move when the shutter fires
- 🔅 **Low light** — dusk, dawn, and moonlight shots
- 🌫️ **Weather** — fog, rain, mist
- 🌿 **Occlusion** — leaves and branches blocking the lens
- 📐 **Framing** — animals often appear in only a small corner of the frame

A model that scores 96% on clean data can collapse to 25% in real field conditions. **Robustness matters more than raw accuracy.**

---

## 💡 Approach

### Transfer Learning
- Pre-trained **EfficientNet-B0** (ImageNet weights)
- Frozen backbone → only classifier head trained
- Only **2.3%** of parameters are trainable
- Enables strong generalization from limited data

### Domain Augmentation
Each augmentation directly encodes a real-world failure mode:

| Augmentation | Simulates |
|---|---|
| `ToGray` / `ToSepia` | IR night vision |
| `MotionBlur` | Fast-moving animal |
| `RandomBrightnessContrast` | Low light conditions |
| `GaussNoise` | High-ISO sensor noise |
| `RandomFog` / `RandomRain` | Weather variation |
| `CoarseDropout` | Leaf/branch occlusion |
| `RandomResizedCrop(scale=0.3)` | Animal in corner of frame |

---

## 📊 Results

### Clean Test Set

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (no aug) | 96.8% | 95.7% |
| Domain Aug (ours) | 95.7% | 94.8% |

*On clean data, both models perform similarly — the baseline has a slight edge.*

### Robustness Under Real-World Conditions

| Condition | Baseline (severe) | Domain Aug (severe) | Gain |
|---|---|---|---|
| Night / IR | — | — | — |
| Motion Blur | — | — | — |
| Low Light | — | — | — |
| Occlusion | — | — | — |
| **Combined (worst-case)** | **24.9%** | **44.2%** | **+19.3%** |

**Average accuracy at severe perturbation:**
- Baseline: **54.7%** (drop: -42.1%)
- Domain Aug: **67.4%** (drop: -28.3%)

### Key Finding

> *"The cost of domain augmentation is -1.1% on clean data. The benefit is +19.3% under worst-case field conditions."*

---

## 🚀 Live Demo

**[→ Try WildEye on Hugging Face Spaces](https://huggingface.co/spaces/zkmine/wildeye)**

Upload any animal image and use the sidebar sliders to simulate real camera-trap conditions:
- Adjust Night IR, Motion Blur, Low Light, Noise, and Occlusion in real time
- Hit **"Worst-case scenario"** to see how the model holds up under maximum stress
- Compare confidence scores as conditions degrade

---

## 🗂️ Repository Structure

```
wildeye/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── 01_baseline_pipeline.ipynb      # Baseline training (no augmentation)
├── 02_augmented_training.ipynb     # Domain augmentation training
├── 03_robustness_evaluation.ipynb  # Robustness evaluation across conditions
├── models/
│   ├── baseline_no_aug_metrics.json
│   └── domain_aug_metrics.json
└── results/
    └── figures/
        ├── augmentation_visualization.png
        ├── comparison_bar_chart.png
        ├── robustness_degradation.png
        └── robustness_summary_bar.png
```

---

## ⚙️ Setup & Running

### Local
```bash
git clone https://github.com/Kattine/wildeye.git
cd wildeye
pip install -r requirements.txt
streamlit run app.py
```

### Google Colab
Open the notebooks in order:
1. `01_baseline_pipeline.ipynb` — train baseline
2. `02_augmented_training.ipynb` — train with domain augmentations
3. `03_robustness_evaluation.ipynb` — evaluate robustness

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Model | EfficientNet-B0 (torchvision) |
| Augmentations | albumentations |
| Training | PyTorch |
| Evaluation | scikit-learn |
| Web App | Streamlit |
| Deployment | Hugging Face Spaces |
| Dataset | Animals-10 (Kaggle) — 8,695 images, 10 classes |

---

## 📁 Dataset

**Animals-10** from Kaggle (`alessiocorrado99/animals10`)
- 10 animal classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel
- 8,695 images total
- Train / Val / Test split: 70% / 15% / 15% (stratified)

---

*Mini Hackathon #1 — MENG 570 · WildEye · EfficientNet-B0 + Domain Augmentation*

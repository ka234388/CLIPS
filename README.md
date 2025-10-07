# CLIP, SgLIP, and A-CLIP Fine-Tuning on Human Action Recognition (HAR) Dataset

This project explores **vision-language model fine-tuning** for the **Human Action Recognition (HAR)** dataset

---

## 📘 Project Overview

In this assignment, three models were fine-tuned and evaluated on the **HAR dataset**:

- **CLIP (Contrastive Language–Image Pre-training)**  
- **SgLIP (Scalable Language–Image Pre-training)**  
- **A-CLIP (Attention-based CLIP variant)**  

The goal was to compare their **accuracy**, **training speed**, and **memory efficiency** when fine-tuned for downstream human action recognition tasks.

---

## 🧠 Dataset

- **Dataset:** Human Action Recognition (HAR)
- **Classes:** 15 (e.g., calling, cycling, eating, running)
- **Images:** ~12.6k total
- **Preprocessing:**
  - Images resized to 224 × 224
  - Random crop and horizontal flip augmentations
- **Loss Function:** Cross-Entropy  
- **Optimizer:** AdamW with cosine learning rate schedule  
- **Precision:** Mixed Precision (AMP)  
- **Epochs:** 10  

---

## ⚙️ Training Configuration

| Parameter | CLIP | SgLIP | A-CLIP |
|------------|------|-------|--------|
| Epochs | 10 | 10 | 10 |
| Batch Size | 256 | 512 | 4096 (64 × 64 accum.) |
| Optimizer | AdamW | AdamW | AdamW |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Loss | Cross-Entropy | Cross-Entropy | Contrastive + Classification |
| Image Size | 224 × 224 | 224 × 224 | 224 × 224 |

---

## 📊 Results Summary

### 🔹 CLIP Fine-Tuning
| Epoch | Train Acc | Val Acc | Time (s) | Peak Mem (MB) |
|:------|:-----------|:---------|:----------|:----------------|
| 1 | 0.118 | 0.173 | 444.7 | 1940 |
| 5 | 0.250 | 0.264 | 389.9 | 1940 |
| 10 | 0.377 | 0.322 | 385.0 | 1940 |

➡️ *Shows gradual improvement; demonstrates adaptability of CLIP’s vision encoder.*

---

### 🔹 SgLIP Fine-Tuning
| Epoch | Train Acc | Val Acc | Time (s) | Peak Mem (MB) |
|:------|:-----------|:---------|:----------|:----------------|
| 1 | 0.615 | 0.827 | 284.5 | 3582.7 |
| 2 | 0.881 | 0.848 | 283.9 | 3581.7 |
| 10 | 0.974 | 0.817 | 284.7 | 3581.7 |

➡️ *Achieved best validation accuracy (84.8%) at Epoch 2 — fast convergence and stable performance.*

---

### 🔹 A-CLIP Fine-Tuning
| Epoch | Train Acc | Val Acc | Time (s) | Peak Mem (MB) |
|:------|:-----------|:---------|:----------|:----------------|
| 1 | 0.130 | 0.248 | 69.0 | 4859 |
| 5 | 0.628 | 0.594 | 69.0 | 3531 |
| 10 | 0.850 | 0.793 | 69.1 | 3531 |

➡️ *A-CLIP reached 79.3% validation accuracy after 10 epochs — significant improvement over standard CLIP.*

---

## 🧩 Key Takeaways

- **SgLIP** achieves faster and higher accuracy due to efficient pretraining with large batch sizes.  
- **A-CLIP** improves fine-grained recognition using saliency-based masking and dual-loss optimization.  
- **CLIP** baseline demonstrates robustness and stable adaptability to downstream tasks.

---

## 🖼️ Visual Results

The repository includes:
- Confusion Matrices  
- Accuracy and Loss Plots  
- Sample Training Logs  

---

## 👩‍💻 Author

**Karthika Ramasamy**  
University of Central Florida  
📧 [ka234388@ucf.edu](mailto:ka234388@ucf.edu)

---

## 📚 References

- OpenAI, “CLIP: Connecting Text and Images”
- Li et al., “Scaling Language–Image Pretraining via Masking”
- Human Action Recognition Dataset (HAR)

---

## 🪄 How to Run

You can reproduce fine-tuning by running the respective notebooks:
```bash
CLIP.ipynb
SgLIP.ipynb
A-CLIP.ipynb

## 🧩 Installation

Make sure to install the required dependencies:

```bash
pip install torch torchvision transformers
| Model      | Validation Accuracy | Best Epoch | Time per Epoch | Peak GPU Memory |
| :--------- | :-----------------: | :--------: | :------------: | :-------------: |
| **CLIP**   |        32.1%        |     10     |      385s      |      1.9GB      |
| **SgLIP**  |        84.8%        |      2     |      284s      |      3.5GB      |
| **A-CLIP** |        79.3%        |     10     |       69s      |      3.5GB      |


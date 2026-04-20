# Crop vs. Weed Classification using Deep Learning

## 1. Project Overview

This project was developed as part of a university group project on *Drones & Machine Learning in Agriculture*.  
The goal is to investigate how well deep learning models can classify crops and weeds from aerial imagery, and how performance changes as task complexity increases.

We design a three-stage classification pipeline, progressively increasing the difficulty of the task:

- **Stage 1:** Binary classification (Crop vs. Weed)  
- **Stage 2:** Crop vs. Weed + Growth Stage (4 classes)  
- **Stage 3:** Fine-grained classification including weed species (24 classes)

---

## 2. Dataset

We use the **Crop vs Weed Dataset**, which contains:

- 60,000+ RGB images  
- Captured from drone perspective  
- Focus on early growth stages  
- Includes:
  - Crops (maize, tomato)
  - Various weed species

### Data Preparation

- Custom stratified train/validation/test split  
- Ensures balanced class distribution across splits

---

## 3. Methodology

### Model

We use a pretrained **ResNet-50** implemented in PyTorch:

- Pretrained on ImageNet  
- Fine-tuned on the agricultural dataset  
- Trained using GPU acceleration (CUDA) in Google Colab

### Training Setup

- Epochs: **3** (consistent across all experiments for comparability)  
- Training time: ~1700 seconds per experiment  
- Same setup used across all stages to isolate the effect of task complexity

---

## 4. Experiments

### Stage 1 — Binary Classification

- Classes: Crop vs. Weed  
- **Accuracy: 0.99**

→ Task is highly separable at a coarse level.

---

### Stage 2 — Crop + Growth Stage

- Classes: 4 (Crop/Weed × Growth Stage)  
- **Accuracy: 0.93**

→ Performance drops as intra-class variation increases.

---

### Stage 3 — Fine-Grained Classification

- Classes: 24 (Crop/Weed + Growth Stage + Weed Species)  
- **Accuracy: 0.92**

→ Further increase in complexity leads to additional performance degradation.

---

## 5. Error Analysis

To better understand model limitations, we analyzed prediction errors using confusion matrices and manual inspection.

### Key Findings

#### 1. Class Imbalance Effects
Classes with fewer training samples show significantly lower classification performance.  
This indicates that the model struggles to generalize in low-data regimes.

#### 2. Visual Similarity Between Classes
Misclassifications frequently occur between visually similar categories, e.g.:

- Tomato (young) vs. Tomato (old)  
- Similar weed species with overlapping textures and color patterns  

This is especially relevant in early growth stages where distinguishing features are less pronounced.

#### 3. Label Noise in Dataset
Manual inspection of selected misclassified samples revealed cases where:

- Ground truth labels do not match the visual content  

This suggests the presence of label noise, which likely impacts model performance.

---

### Interpretation

While binary classification achieves very high accuracy (>0.99), performance decreases in more complex scenarios (~0.92) due to:

- Increased intra-class similarity  
- Class imbalance  
- Label inconsistencies  

This highlights a key limitation:

> High accuracy in simple classification tasks does not necessarily translate to robust performance in fine-grained, real-world scenarios.

---

## 6. Project Structure
```
├── notebooks/
│   ├── ResNet50_stage1.ipynb
│   ├── ResNet50_stage2.ipynb
│   └── ResNet50_stage3.ipynb
│
├── requirements.txt
└── README.md
```

---

## 7. How to Run

### 1. Clone the repository
```
git clone https://github.com/JUS1807/crop-vs-weed-classification.git
cd crop-vs-weed-classification
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Download dataset 

Download the dataset from:
https://digital.csic.es/handle/10261/368094

Adjust the dataset path in the notebooks accordingly.

### 4. Run Notebooks

Open and run the notebooks in order:

1. **Stage 1**
2. **Stage 2**
3. **Stage 3**

---

## 8. Key Takeaways

- Deep learning models perform extremely well on coarse agricultural classification tasks  
- Performance degrades with increasing task granularity  
- Data quality (label noise) and class imbalance significantly impact results  
- Fine-grained plant classification remains a challenging real-world problem

---

## 9. Future Work

- Improve robustness via data augmentation  
- Address class imbalance (e.g., resampling or weighted loss)  
- Clean or relabel noisy samples  
- Experiment with alternative architectures or longer training

---

## 10. Notes

- All experiments were intentionally limited to **3 epochs** to ensure comparability  
- Results already show strong performance without extensive hyperparameter tuning



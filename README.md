# Machine-Learning-Project  
# Adaptive Teaching Regularization with Class-Aware Knowledge Weighting
DSL501 Machine Learning Project | Rohit Raghuwanshi (Roll No: 12341820)

Colab Link - https://colab.research.google.com/drive/1GykqxrbBa9X4I7WfF0-SCq8sJrxKG7NA#scrollTo=c42OcEgabDcB

## 📌 Project Overview
This project introduces Class-Aware Adaptive Teaching Regularization (CATR), a novel framework that addresses two critical limitations of Liu et al. (NeurIPS 2024):
1. Uniform knowledge transfer that ignores class imbalance
2. Static regularization strength that cannot adapt to sample difficulty

### ✨ Key Innovations
- Class-Aware Weighting**: Dynamically adjusts teaching intensity using `α(x) = w_c × (1 - max(p_teacher))`
- Lightweight Gating**: Learnable gate `g(x) = σ(W · [h_t; h_s] + b)` decides when to use teacher knowledge
- <1% computational overhead** while delivering **+5.36% Macro-F1 gain**

## 📊 Datasets Used
- IMDB Movie Reviews**:  
  25,000 samples** (12,500 positive + 12,500 negative) — balanced binary sentiment classification.

- Davidson Hate Speech:  
  - 4,783 samples (~25K) with severe class imbalance:
  - Hate speech: 1,430 samples (5.77%)
  - Offensive language: 19,190 samples (77.43%)
  - Neither: 4,163 samples (16.80%)

> 💡 Both datasets are used exactly as specified in the SoP (Sections 4.1–4.2).

## 📂 Project Structure

Machine-Learning-Project/
├── data/
│   ├── raw/                    # Raw datasets (Davidson)
│   └── processed/              # Cleaned + preprocessed CSVs
│
├── models/
│   ├── baseline_tiny_davidson/      # Baseline model (part 2)
│   ├── teaching_reg_davidson/       # Liu et al. Teaching Reg (part 3)
│   └── catr_weighting_davidson/     # Best model (Class-Aware only)
│
├── results/
│   ├── part6_metrics.csv            # Ablation & low-resource results
│   └── part7_efficiency.csv         # Training time, memory, latency
│
├── docs/
│   └── final_report.pdf             # Full LaTeX report
│   └── prokect_sop.pdf
|
├── requirements.txt                 # All dependencies
└── README.md                        # Main project overview


This structure ensures full reproducibility and clear navigation.

# ⚽ FIFA Unified Scouting System
### Advanced ML Pipeline — Player Valuation & Performance Tier Prediction

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Tasks Breakdown](#-tasks-breakdown)
- [Models & Results](#-models--results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Group Members](#-group-members)

---

## 🎯 Overview

The **FIFA Unified Scouting System** is an end-to-end machine learning pipeline built on FIFA player data. It tackles two simultaneous prediction problems:

- 💰 **Regression** — Predict a player's market value (`Value Per M$`)
- 🏅 **Classification** — Predict a player's performance tier (`Low / Mid / High / Elite`)

The system progresses from exploratory analysis through preprocessing, baseline models, hyperparameter tuning, ensemble methods, and a unified inference pipeline — with statistical stability proof via cross-validation.

---

## 📊 Dataset

**Source:** `Fifa.csv` — FIFA player dataset with attributes including:

| Feature | Description |
|---|---|
| `Name` | Player name (identifier, dropped from features) |
| `Age` | Player age |
| `Overall_Rating` | Current skill rating |
| `Future Potential` | Projected future rating |
| `Total_Stats Score` | Composite performance score |
| `Value Per M$` | Market value in millions (regression target) |
| `Position` | Playing position (OHE encoded) |
| `Country` | Nationality (OHE encoded) |
| `Team` | Club team (OHE encoded) |

> **Note:** No missing values found — no imputation required. `Value Per M$` is heavily right-skewed (skewness ≈ 8); IQR-based winsorization is applied on training data only.

---

## 🏗️ Pipeline Architecture

```
Raw Data (Fifa.csv)
       │
       ▼
┌─────────────────────┐
│  Task 1: EDA        │  Distribution analysis, correlation matrix, position ratings
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Task 2: Preprocessing │  Train/Test split → OHE encoding → Outlier handling → Scaling
└────────┬────────────┘    (all transformations fit on TRAIN only — no leakage)
         │
         ▼
┌─────────────────────┐
│  Task 3: Target     │  Performance Tier created from TRAIN quartiles of Overall_Rating
│  Engineering        │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
REGRESSION  CLASSIFICATION
    │              │
    ▼              ▼
Polynomial     Logistic Reg.
Regression     Naïve Bayes
+ Ridge/Lasso  (Gaussian/Bernoulli/Complement)
    │              │
    └────┬─────────┘
         ▼
┌────────────────────────────┐
│  Assignment 3 Extensions   │
│  KNN · SVM · Random Forest │
│  GridSearchCV Tuning        │
│  Learning Curve Diagnosis   │
│  Voting · GradBoost · Stack │
│  Unified sklearn Pipeline   │
│  5-Fold CV Stability        │
└────────────────────────────┘
         │
         ▼
  Scouting Report + results.json
```

---

## 🧩 Tasks Breakdown

### Task 1 — Exploratory Data Analysis
- Distribution of `Value Per M$` (raw vs. log-transformed)
- Correlation heatmap for numerical features
- Average `Overall_Rating` per playing position
- Pairplot of key numerical features

### Task 2 — Data Preprocessing
- **Split first** (80/20), then encode and scale to prevent leakage
- One-Hot Encoding for `Country`, `Position`, `Team`
- IQR-based outlier capping fitted on training set only
- `StandardScaler` fitted on training set, applied to test

### Task 3 — Classification Target Engineering
- `Performance_Tier` defined from training-set quartiles of `Overall_Rating`:
  - **Low** → below Q1
  - **Mid** → Q1 to Median
  - **High** → Median to Q3
  - **Elite** → above Q3
- Balanced class distribution ensured by percentile-based thresholds

### Task 4 — Polynomial Regression (Assignment 2)
- Degrees 1–4 tested; best degree selected by test R²
- **Ridge (L2)** and **Lasso (L1)** regularisation sweeps over log-spaced alphas
- Lasso zero-out analysis: automatic feature selection
- Ridge outperforms Lasso due to many collectively useful OHE features

### Task 5 — Logistic Regression (Assignment 2)
- C hyperparameter sweep over log-spaced values
- L1 vs L2 penalty comparison
- Confusion matrix for tier misclassification analysis

### Task 6 — Naïve Bayes Classification (Assignment 2)
- **GaussianNB** on continuous numerical features
- **BernoulliNB** on binarized OHE features
- **ComplementNB** on shifted non-negative OHE features
- Scaling sensitivity analysis for GaussianNB

### Task 7 — Cross-Validation (Assignment 2)
- 5-Fold KFold CV for best regression model (Ridge + Polynomial)
- 5-Fold Stratified KFold for Logistic Regression and GaussianNB

### Task 1 (A3) — Diverse Model Selection
Three orthogonal algorithmic approaches:

| Algorithm | Inductive Bias | Why FIFA? |
|---|---|---|
| **KNN** | Instance-based / lazy | Exploits local similarity in dense numerical space |
| **SVM (RBF kernel)** | Margin maximisation | Maps high-dimensional OHE space to linearly separable classes |
| **Random Forest** | Tree bagging + subsampling | Handles non-linear interactions; robust to outliers |

### Task 2 (A3) — Hyperparameter Tuning & Bias-Variance Diagnosis
- `GridSearchCV` with 5-fold CV for all 3 models × 2 tasks
- Learning curves plotted for each model to diagnose underfitting vs. overfitting

### Task 3 (A3) — Ensemble Methods

| Strategy | Implementation | Rationale |
|---|---|---|
| **Voting** | `VotingClassifier` / `VotingRegressor` | Reduces variance through diversity |
| **Gradient Boosting** | Sequential error correction | Reduces bias and variance jointly |
| **Stacking** | Meta-learner (LR / Ridge) | Learns optimal combination of base predictions |

### Task 4 (A3) — Unified Inference Pipeline
- Single `sklearn.Pipeline` for both regression and classification
- Accepts raw feature arrays, applies scaling internally
- `scouting_report()` function outputs human-readable player valuation + tier

### Task 5 (A3) — Stability Assessment
- 5-Fold CV mean ± std for best regression and classification models
- Statistical proof of consistent generalisation across data subsets

### Task 6 (A3) — System Comparison
- Assignment 2 (Ridge + Poly / Logistic Reg.) vs. Assignment 3 (best ensemble)
- Side-by-side R² and Accuracy comparison charts

---

## 📈 Models & Results

### Regression (`Value Per M$`)

| Model | R² | RMSE |
|---|---|---|
| Linear Regression (baseline) | — | — |
| Polynomial + Ridge (A2 best) | see notebook | see notebook |
| KNN (tuned) | see notebook | see notebook |
| SVM (tuned) | see notebook | see notebook |
| Random Forest (tuned) | see notebook | see notebook |
| Voting Regressor | see notebook | see notebook |
| Gradient Boosting | see notebook | see notebook |
| **Stacking Regressor** | **best** | **lowest** |

### Classification (`Performance_Tier`)

| Model | Accuracy |
|---|---|
| Logistic Regression (A2 best) | see notebook |
| GaussianNB | see notebook |
| KNN (tuned) | see notebook |
| SVM (tuned) | see notebook |
| Random Forest (tuned) | see notebook |
| Voting Classifier | see notebook |
| Gradient Boosting | see notebook |
| **Stacking Classifier** | **best** |

> Exact metrics depend on your dataset and random seed. Run the notebook to reproduce.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | All ML models, preprocessing, evaluation |
| `scipy` | Statistical utilities |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/fifa-scouting-system.git
cd fifa-scouting-system

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

### Run the Notebook
```bash
jupyter notebook FIFA_ML_Assignment3.ipynb
```

Place `Fifa.csv` in the same directory as the notebook before running.

---

## 📁 Project Structure

```
fifa-scouting-system/
│
├── FIFA_ML_Assignment3.ipynb   # Main notebook (A2 + A3 full pipeline)
├── Fifa.csv                    # Dataset (add locally)
├── results.json                # Generated deliverable (CV stability + best params)
├── requirements.txt
└── README.md
```

---

## 📄 results.json

The notebook automatically generates `results.json` containing:

```json
{
  "group_members": [ ... ],
  "best_hyperparameters": {
    "classification": { "KNN": {...}, "SVM": {...}, "RandomForest": {...} },
    "regression":     { "KNN": {...}, "SVM": {...}, "RandomForest": {...} }
  },
  "cv_stability": {
    "regression":     { "metric": "r2",       "mean": ..., "std": ..., "per_fold": [...] },
    "classification": { "metric": "accuracy", "mean": ..., "std": ..., "per_fold": [...] }
  }
}
```

---



<div align="center">
FIFA Unified Scouting System · Machine Learning Assignment 3
</div>

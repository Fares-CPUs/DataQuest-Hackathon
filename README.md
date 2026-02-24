# DataQuest Hackathon — Intelligent Insurance Bundle Recommendation

This repository contains the final Submittion Files for the **DataQuest Hackathon** which was made in Collaboration by the **INCEPTION Team**

## INCEPTION Team Members 
- Oueslati Ouala Eddine (Team Leader)
- Nasri Mohamed Amine
- Berrehouma Fares
- Gharbi Wieme


## Project Overview

This project implements a **machine learning-based recommender system** that predicts which insurance coverage bundle a prospective customer will purchase. The model uses structured tabular data capturing demographic, financial, behavioral, and temporal signals to make accurate predictions across 10 classes (bundles 0–9).

The solution was designed to optimize the **competition-adjusted Macro F1 score**, taking into account:

- Model size constraints  
- Inference latency penalties  
- Memory and CPU limitations  


## Project Structure

```bash

DataQuest-Brief-Document.pdf
│  └─ Official hackathon brief containing problem statement, constraints, and evaluation rules.

Dataset/
│
├─ solution.py
│    └─ Competition-ready inference pipeline (preprocess, load_model, predict).
│
├─ test.csv
│    └─ Test dataset (features only) used for final predictions.
│
├─ train.csv
|    └─ Training dataset (features + Purchased_Coverage_Bundle target).
|
└─ README.md
     └─ Dataset description, feature definitions, and usage instructions.

Deliverables/
│
├─ Phase 1 : Model Development/
│   │
│   ├─ modeltraining.py
│   │    └─ Script used for feature engineering, model training, and model serialization.
│   │
│   └─ submission/
│        │
│        ├─ solution.py
│        │    └─ Final judge-compliant solution file.
│        │
│        ├─ model.joblib
│        │    └─ Serialized LightGBM model bundle (compressed).
│        │
│        └─ requirements.txt
│             └─ Python dependency list required for execution in the judge environment.
│
└─ Phase 2 : Productization & Deployment/
    │
    ├─ Web & API Integration/
    │    │
    │    ├─ api.py
    │    │    └─ Backend API exposing prediction endpoints for real-time inference.
    │    │
    │    └─ UI.py
    │         └─ Frontend user interface enabling bundle prediction through a simple interactive form.
    │
    ├─ Inception AI - Presentation Slides.pdf
    │    └─ Final pitch deck presented during Phase II.
    │
    ├─ INCEPTION Team - Technical Report.pdf
    │    └─ Complete technical documentation including architecture, features, and justification.
    │
    └─ Video Demonstration.mp4
         └─ Recorded demo showcasing the deployed solution and inference workflow.

```


## Dataset

- **Training Set:** 60,868 rows with features + target (`Purchased_Coverage_Bundle`)  
- **Test Set:** 15,218 rows with features only  

### Key Features

- **Demographics & Household:** Adult/Child/Infant Dependents  
- **Financial:** Estimated Annual Income  
- **Behavioral & Risk:** Previous Claims, Policy Amendments, Claim-free Years  
- **Temporal:** Policy Start Month/Week/Day  
- **Broker & Acquisition:** Broker_ID, Employer_ID, Acquisition Channel  

**Target:** `Purchased_Coverage_Bundle` (10 classes: 0–9)


## Solution Architecture

High-level pipeline:

1. **Preprocessing**
   - Feature engineering: `Total_Dependents`, `Income_Log`, `Risk_Score`, `Month_sin/cos`  
   - Encoding: ordinal + frequency encoding for high-cardinality features  
   - Numeric imputation: median replacement  

2. **Model Loading**
   - Pre-trained LightGBM classifier (`LGBMClassifier`) stored in `model.joblib`  

3. **Prediction**
   - `predict()` returns `User_ID` + `Purchased_Coverage_Bundle`  
   - Predictions are clipped to valid class range `[0–9]`  


## Preprocessing & Feature Engineering

- **Household features:** Total dependents, vehicles per dependent  
- **Financial features:** Log-transformed income  
- **Behavioral risk score:** Combines claims, claim-free years, and amendments  
- **Temporal encoding:** Month as cyclical features (`sin` and `cos`)  
- **Broker influence:** Frequency encoding  
- **Categorical encoding:** Ordinal mapping  
- **Numeric imputation:** Median fill + float32 conversion  


## Model Training

- **Model:** LightGBM classifier (`LGBMClassifier`)  
- **Trees:** 45 trees, 63 leaves  
- **Learning rate:** 0.08  
- **Class weights:** Adjusted to optimize Macro F1  
- **Objective:** Multiclass (10 classes)  

**Rationale for LightGBM:**

| Model       | Advantages                         | Notes                                      |
|------------ |-----------------------------------|-------------------------------------------|
| **LightGBM** | Fast inference, small model size, memory efficient | Selected for competition constraints |
| CatBoost    | Great categorical handling, high accuracy | Larger model, slower inference            |
| XGBoost     | Strong baseline, stable            | Slower training, slightly larger          |


## Explainability

- **Feature Importance:** Top 15 features influence predictions significantly  
- **SHAP Analysis:** Highlights contribution of each feature and interactions  
- **Prediction Confidence:** Histogram of max probabilities shows model certainty  
- **Tree Visualization:** Provides insight into first tree structure  


## Installation

```bash
# Clone repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Licence 

This code is for hackathon evaluation purposes only.
Do not redistribute or use for commercial purposes without permission.

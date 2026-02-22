# INCEPTION Team - Intelligent Insurance Bundle Recommender

This repository contains the final solution for the *DataQuest Hackathon*.  
The system predicts which insurance bundle a customer is likely to purchase based on profile and policy data.

---

## Project Structure : 
submission.zip
│
├─ solution.py          # Preprocessing, model loading, and predict interface
├─ model.joblib         # Trained LightGBM model bundle
├─ requirements.txt     # Python dependencies
├─ demo.mp4              # Web Integration Video Capture
├─ README.md            # Project documentation
### 1. solution.py

Contains the required functions for submission:

1. *Preprocessing*
   - Feature engineering: Total_Dependents, Income_Log, Risk_Score, Month_sin/cos  
   - Encoding: ordinal + frequency encoding for high-cardinality features  
   - Numeric imputation: median replacement  

2. *Model Loading*
   - Pre-trained LightGBM classifier (LGBMClassifier) stored in model.joblib  

3. *Prediction*
   - predict() returns User_ID + Purchased_Coverage_Bundle  
   - Predictions are clipped to valid class range [0–9] ---

### 2. modelTraining.py

Used to:

- Load the training dataset (train.csv)  
- Engineer features: Total_Dependents, Income_Log, Risk_Score, Month_sin, Month_cos  
- Encode categorical features and handle missing values  
- Apply class weighting for Macro F1 optimization  
- Train a *LightGBM* classifier (45 trees)  
- Save the model bundle as model.joblib

---

### 3. model.joblib

Serialized model bundle containing:

- Trained LightGBM model  
- Feature order and columns info  
- Categorical encoders  
- Numeric median values  
- Frequency maps and month mapping

---

### 4. requirements.txt
Lists Python dependencies


##Author
INCEPTION Team during the DataQuest 2026

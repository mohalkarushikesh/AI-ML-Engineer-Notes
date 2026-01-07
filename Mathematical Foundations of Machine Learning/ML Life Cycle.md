# 📘 Machine Learning Life Cycle

## 🔟 Steps

1. **Problem Formulation**  
   - Define the task clearly → **classification** (categorical output) or **regression** (continuous output).  

2. **Data Acquisition**  
   - Collect datasets from sources like **Kaggle, UCI**, or generate your own.  

3. **Data Cleaning & Labelling**  
   - Remove irrelevant features, fix noisy/missing data, and label if needed.  

4. **Exploratory Data Analysis (EDA)**  
   - Use statistics and **visualizations** to uncover patterns, correlations, and distributions.  

5. **Data Pre-processing**  
   - Handle **missing values**, outliers, imbalance.  
   - **Scale** numerical features, **encode** categorical features.  

6. **Dataset Splitting**  
   - Split into **train, validation, test sets** (e.g., 70/15/15).  

7. **Model Training & Performance Evaluation**  
   - Train chosen models.  
   - Evaluate with metrics:  
     - **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC.  
     - **Regression:** MSE, RMSE, MAE, $R^2$.  

8. **Hyperparameter Tuning**  
   - Optimize with **GridSearchCV**, **RandomizedSearchCV**, **Optuna**, or Bayesian optimization.  

9. **Final Evaluation**  
   - Test the best model on the **test set** to estimate real-world performance.  

10. **Model Deployment**  
    - Serve via **Flask/Django APIs**, **FastAPI**, or integrate into apps (Android Studio, Flutter).  
    - Monitor performance post-deployment.  

---

## ⚡ Pocket Version (super short)
- **Formulate → Acquire → Clean → EDA → Preprocess → Split → Train → Tune → Evaluate → Deploy**

---

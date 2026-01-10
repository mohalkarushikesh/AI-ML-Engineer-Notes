# 🧠 Stacking (Stacked Generalization)

## 📖 Definition
- **Stacking** is an ensemble learning technique that combines predictions from multiple diverse models (base learners) using a **meta‑model** (also called a blender).  
- Unlike Bagging and Boosting, which usually combine models of the same type, Stacking often uses **different algorithms** (e.g., decision trees, SVMs, logistic regression).  
- The meta‑model learns how to best combine the outputs of base learners to improve overall performance.

---

## ⚙️ How Stacking Works
1. **Train Base Learners**  
   - Train multiple models (e.g., decision tree, SVM, k‑NN, logistic regression) on the training dataset.  

2. **Generate Predictions**  
   - Each base learner makes predictions on the dataset.  
   - These predictions become **new features** for the next stage.  

3. **Train Meta‑Model (Blender)**  
   - A meta‑model (often logistic regression or linear regression) is trained on the predictions of base learners.  
   - It learns how to optimally combine them.  

4. **Final Prediction**  
   - The meta‑model outputs the final prediction, leveraging the strengths of all base learners.  

<img width="1235" height="539" alt="stacking" src="https://github.com/user-attachments/assets/1f28b2ca-9680-4659-9f20-b6ba2c37ed51" />

---

## 🔹 Key Characteristics
- **Diversity of models**: Uses different algorithms as base learners.  
- **Meta‑learning**: A second‑level model learns how to combine predictions.  
- **Flexibility**: Can be applied to both classification and regression tasks.  
- **Bias–variance balance**: By combining diverse models, Stacking reduces both bias and variance.  

---

## ✅ Advantages
- Often achieves **higher accuracy** than individual models.  
- Leverages strengths of different algorithms.  
- Flexible and adaptable to many problem types.  
- Works well in competitions (e.g., Kaggle).  

---

## ⚠️ Limitations
- **Complexity**: More difficult to implement and tune compared to Bagging/Boosting.  
- **Computational cost**: Requires training multiple models plus a meta‑model.  
- **Risk of overfitting** if meta‑model is too complex or base learners are not diverse.  
- **Interpretability**: Harder to explain compared to single models.  

---

## 🧪 Applications
- **Finance**: Credit scoring, fraud detection.  
- **Healthcare**: Disease prediction combining multiple models.  
- **Marketing**: Customer churn prediction.  
- **Competitions**: Widely used in Kaggle for top‑performing solutions.  

---

## 📝 Comparison: Bagging vs Boosting vs Stacking

| Feature | Bagging | Boosting | Stacking |
|---------|---------|----------|----------|
| Training | Parallel | Sequential | Parallel (then meta‑model) |
| Focus | Variance reduction | Bias reduction | Combine diverse models |
| Combination | Average/Vote | Weighted sum | Meta‑model learns combination |
| Example | Random Forest | AdaBoost, XGBoost | Stacked ensembles |

---

# ✅ Summary
- **Stacking = Stacked Generalization**.  
- Combines multiple diverse base learners using a meta‑model.  
- More flexible than Bagging and Boosting, but also more complex.  
- Best suited for tasks where combining different algorithms can capture complementary strengths.  

---

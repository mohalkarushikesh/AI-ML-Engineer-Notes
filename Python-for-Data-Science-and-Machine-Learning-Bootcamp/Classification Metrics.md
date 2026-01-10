**Accuracy, Precision, Recall, F1, and ROC‑AUC are the most widely used metrics for evaluating classification models. Each highlights a different aspect of performance: Accuracy measures overall correctness, Precision focuses on false positives, Recall emphasizes false negatives, F1 balances both, and ROC‑AUC evaluates ranking ability across thresholds. Choosing the right metric depends on the problem context, especially in imbalanced datasets.**

---

# 📊 In‑Depth Classification Metrics

## 🔹 Accuracy
- **Definition**: Proportion of correctly classified samples out of all samples.  
- Formula:  
  $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$  
- **Strengths**: Simple, intuitive, good for balanced datasets.  
- **Limitations**: Misleading in imbalanced datasets (e.g., 95% accuracy when predicting all negatives in a 95:5 dataset).  

---

## 🔹 Precision
- **Definition**: Fraction of positive predictions that are actually correct.  
- Formula:  
  $Precision = \frac{TP}{TP + FP}$  
- **Interpretation**: “Of all predicted positives, how many were true?”  
- **Use Case**: Important when **false positives are costly** (e.g., spam detection, medical diagnosis where false alarms are problematic).  

---

## 🔹 Recall (Sensitivity / True Positive Rate)
- **Definition**: Fraction of actual positives that were correctly identified.  
- Formula:  
  $Recall = \frac{TP}{TP + FN}$  
- **Interpretation**: “Of all actual positives, how many did we catch?”  
- **Use Case**: Important when **false negatives are costly** (e.g., cancer detection, fraud detection).  

---

## 🔹 F1 Score
- **Definition**: Harmonic mean of Precision and Recall.  
- Formula:  
  $F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$  
- **Interpretation**: Balances Precision and Recall into a single metric.  
- **Use Case**: Best for **imbalanced datasets** where both false positives and false negatives matter.  

---

## 🔹 ROC‑AUC (Receiver Operating Characteristic – Area Under Curve)
- **ROC Curve**: Plots True Positive Rate (Recall) vs. False Positive Rate across thresholds.  
- **AUC**: Area under the ROC curve, ranging from 0.5 (random guessing) to 1.0 (perfect classifier).  
- **Interpretation**: Measures the model’s ability to distinguish between classes across thresholds.  
- **Use Case**: Useful for **ranking problems** and when threshold choice is flexible (e.g., credit scoring, risk prediction).  

---

## 📝 Comparison Table

| Metric      | Formula | Focus | Best Use Case |
|-------------|---------|-------|---------------|
| Accuracy    | $(TP+TN)/(TP+TN+FP+FN)$ | Overall correctness | Balanced datasets |
| Precision   | $TP/(TP+FP)$ | False positives | Spam filters, medical tests |
| Recall      | $TP/(TP+FN)$ | False negatives | Fraud/cancer detection |
| F1 Score    | Harmonic mean of Precision & Recall | Balance | Imbalanced datasets |
| ROC‑AUC     | Area under ROC curve | Ranking ability | Threshold‑independent evaluation |

---

## ⚠️ Key Insights
- **Accuracy is not enough** for imbalanced datasets.  
- **Precision vs Recall trade‑off**: Increasing one often decreases the other.  
- **F1 Score** is a balanced metric when both errors matter.  
- **ROC‑AUC** is threshold‑independent and widely used for model comparison.  

---

## ✅ Summary
- **Accuracy**: Good for balanced data.  
- **Precision**: Focuses on false positives.  
- **Recall**: Focuses on false negatives.  
- **F1 Score**: Balances Precision and Recall.  
- **ROC‑AUC**: Evaluates ranking ability across thresholds.  

---

Sources: [Neptune.ai](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc), [Deepchecks](https://www.deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/), [Sanfoundry](https://www.sanfoundry.com/machine-learning-model-evaluation-metrics-accuracy-precision-recall-f1/)

---

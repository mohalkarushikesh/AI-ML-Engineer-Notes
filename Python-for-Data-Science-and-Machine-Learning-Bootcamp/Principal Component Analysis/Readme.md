# **🔹 Principal Component Analysis (PCA)**

## 📖 Definition
- Principal Component Analysis (PCA) is a **statistical technique** used to examine the interrelations among a set of variables in order to identify the **underlying structure** of those variables.  
- It is also known as **general factor analysis**.  
- While regression determines a **single line of best fit** to a dataset, factor analysis (and PCA) determines **several orthogonal lines of best fit**.  

---

## 🔸 Orthogonality
- **Orthogonal** means *at right angles*.  
- In PCA, the principal components are **perpendicular to each other** in n‑dimensional space.  
- This ensures that the components are **uncorrelated**.  

---

## 🔸 n‑Dimensional Space
- The **sample space** is defined by the number of variables.  
- Example: If a dataset has 4 variables, the sample space is **4‑dimensional**.  

<img width='700' height='400' src="https://github.com/user-attachments/assets/8b7b0dd5-0043-414f-be57-90ac96dca72d" /> 

---

## 🔸 Components
- PCA performs a **linear transformation** that redefines the variable system such that:  
  - The **first principal component** captures the greatest variance in the dataset.  
  - The **second principal component** captures the next greatest variance, and so on.  
- This process allows us to **reduce the number of variables** used in analysis while retaining most of the information.  

<img width='700' height='400' src="https://github.com/user-attachments/assets/f6497092-5c6b-43a1-b719-6e0acc596452" />

---

## 🔸 Properties
- Principal components are **uncorrelated** because they are orthogonal in the sample space.  
- PCA can be extended to **higher dimensions**, where each new component explains progressively less variance.  

<img <img width='700' height='400' src="https://github.com/user-attachments/assets/363e978c-a6ee-4e48-ba81-89fe4ccb7efd" /> 

---

## 🔸 Compression of Variation
- For datasets with a **large number of variables**, PCA can compress the explained variation into just a few components.  
- This reduces dimensionality while preserving most of the dataset’s variability.  
- The most challenging part of PCA is **interpreting the components** — understanding what each principal component represents in terms of the original variables.  

---

## ✅ Summary
- **PCA** is a dimensionality reduction technique that identifies orthogonal directions (principal components) capturing maximum variance.  
- **Key ideas**: orthogonality, variance maximization, dimensionality reduction.  
- **Advantages**: reduces complexity, removes redundancy, improves visualization.  
- **Challenge**: interpreting the meaning of components in real-world terms.  

---

# 🔹 PCA in Short

- **Principal Component Analysis (PCA)** is a **dimensionality reduction algorithm**.  
- It transforms data into a new coordinate system where the axes (principal components) capture maximum variance.  
- **Key property**: Components are orthogonal (uncorrelated).

---

## 📊 Where PCA is Applied
- **Supervised Learning** → Preprocessing step to reduce features before classification/regression (e.g., logistic regression, SVM).  
- **Unsupervised Learning** → Used to visualize and compress high‑dimensional data (e.g., clustering with K‑Means).  
- **Semi‑Supervised Learning** → Helps reduce dimensionality when labels are partially available, improving efficiency and generalization.

---

## ✅ In Short
- **PCA = dimensionality reduction tool**.  
- **Not a classifier/regressor itself** → but a preprocessing technique.  
- **Applicable across supervised, unsupervised, and semi‑supervised tasks** → whenever feature reduction or variance capture is useful.  

---


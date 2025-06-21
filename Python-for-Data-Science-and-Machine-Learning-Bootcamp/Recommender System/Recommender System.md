## 📚 Recommender System

Recommender systems suggest items to users based on various signals such as preferences, behavior, or item attributes. The two most common types are:

### 🔹 Content-Based Filtering (CB)
- Focuses on the **attributes of items**.
- Recommends items that are similar to those the user liked in the past.
- Relies heavily on metadata, such as genre, tags, or features.
- Example: Recommending similar books based on topics or author style.

### 🔹 Collaborative Filtering (CF)
- Makes recommendations using the **"wisdom of the crowd"**.
- Learns from the **interactions between users and items** (e.g., ratings, likes, purchases).
- Does **feature learning automatically**, reducing manual effort in feature engineering.
- Generally more effective and scalable than content-based systems.

---

## 🧠 Types of Collaborative Filtering

### 🔸 Memory-Based Collaborative Filtering
- Uses similarity metrics (e.g., **cosine similarity**, Pearson correlation) to find like-minded users or similar items.
- Often implemented using **User-User** or **Item-Item** similarity matrices.

### 🔸 Model-Based Collaborative Filtering
- Uses machine learning techniques to **learn latent features** from user-item interactions.
- Commonly implemented with **SVD (Singular Value Decomposition)**, Matrix Factorization, or deep learning models.

---

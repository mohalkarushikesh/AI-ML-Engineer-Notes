## 📘 Apriori Algorithm

### 🔹 Definition
- Apriori is an algorithm used to **mine frequent itemsets** and generate **association rules** from transactional datasets.
- It is widely applied in **market basket analysis** (e.g., “if a customer buys bread, they are likely to buy butter”).

---

### 🔹 Key Concepts
1. **Itemset:** A collection of one or more items.  
2. **Support:** Frequency of occurrence of an itemset in the dataset.  
   \[
   \text{Support}(A) = \frac{\text{Number of transactions containing A}}{\text{Total transactions}}
   \]
3. **Confidence:** Likelihood that item $B$ is purchased when item $A$ is purchased.  
   \[
   \text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}
   \]
4. **Lift:** Strength of a rule compared to random chance.  
   \[
   \text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}
   \]

---

### 🔹 Working Steps
1. **Set minimum thresholds** for support and confidence.  
2. **Generate candidate itemsets** of length $k$.  
3. **Prune itemsets** that do not meet minimum support.  
4. **Repeat** until no more frequent itemsets can be found.  
5. **Generate rules** from frequent itemsets that meet confidence and lift thresholds.

---

### 🔹 Applications
- **Retail/Market Basket Analysis:** Identify product bundles frequently bought together.  
- **Recommendation Systems:** Suggest items based on purchase history.  
- **Healthcare:** Discover co-occurrence of symptoms or treatments.  
- **Web Usage Mining:** Find patterns in user navigation.  

---

### 🔹 Example
Suppose we have transactions:
- T1: {Milk, Bread}  
- T2: {Milk, Diaper, Beer, Bread}  
- T3: {Milk, Diaper, Beer, Cola}  
- T4: {Diaper, Beer}  
- T5: {Milk, Diaper, Bread, Beer}  

Apriori might discover:
- Frequent itemset: {Milk, Diaper, Beer}  
- Rule: {Diaper, Beer} → {Milk} with high confidence.  

---

### 🔹 Advantages
- Simple and easy to understand.  
- Works well for small to medium datasets.  

### 🔹 Limitations
- Computationally expensive for large datasets (many candidate itemsets).  
- Requires multiple database scans.  

---

✅ **In short:** Apriori is a foundational algorithm for discovering frequent itemsets and association rules, especially useful in market basket analysis and recommendation systems.

---

Here’s a **Python implementation of the Apriori algorithm** using the `mlxtend.frequent_patterns` library. This example demonstrates how to find frequent itemsets and generate association rules from a small dataset:

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Create a sample dataset (transactions)
dataset = [
    ['Milk', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Bread', 'Beer']
]

# Step 2: Convert dataset into a one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("Transaction DataFrame:")
print(df)

# Step 3: Apply Apriori to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

### 🔹 What happens here:
1. **Dataset:** A list of transactions (like market basket data).  
2. **Encoding:** Convert transactions into a binary matrix (items present = 1, absent = 0).  
3. **Apriori:** Finds frequent itemsets that meet the minimum support threshold.  
4. **Association Rules:** Generates rules with confidence and lift values.  

---

### 🔹 Example Output (simplified)
- Frequent itemset: `{Milk, Diaper}` with support ≈ 0.6  
- Rule: `{Diaper, Beer} → {Milk}` with confidence ≈ 0.75 and lift > 1  

---

✅ This shows how Apriori can uncover hidden relationships in transactional data — the foundation of **market basket analysis** and **recommendation systems**.

---

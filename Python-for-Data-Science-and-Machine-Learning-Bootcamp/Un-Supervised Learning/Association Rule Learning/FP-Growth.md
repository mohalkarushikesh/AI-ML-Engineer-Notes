## 📘 FP-Growth Algorithm

### 🔹 Definition
- **FP-Growth (Frequent Pattern Growth)** is an efficient algorithm for mining frequent itemsets without candidate generation.  
- It uses a **compact data structure called FP-tree (Frequent Pattern Tree)** to represent transactions.  
- It avoids the costly candidate generation step of Apriori and the repeated intersections of Eclat.

---

### 🔹 How It Works
1. **Build FP-Tree:**  
   - Scan the dataset once to find frequent items.  
   - Order items by frequency and insert transactions into a tree structure.  
   - Shared prefixes are merged, making the tree compact.  

2. **Mine Patterns:**  
   - Recursively extract frequent itemsets from the FP-tree using conditional pattern bases.  
   - Grow longer itemsets by combining smaller frequent patterns.  

---

### 🔹 Key Features
- **Efficiency:** Faster than Apriori and Eclat for large datasets.  
- **Compact Representation:** FP-tree compresses transactions into a smaller structure.  
- **No Candidate Generation:** Directly mines frequent itemsets.  

---

### 🔹 Applications
- **Market Basket Analysis:** Discover product bundles frequently bought together.  
- **Recommendation Systems:** Suggest items based on frequent co-occurrence.  
- **Healthcare:** Identify frequent symptom-treatment combinations.  
- **Web Mining:** Find common navigation paths or clickstream patterns.  

---

### 🔹 Comparison with Apriori & Eclat
| Feature | Apriori | Eclat | FP-Growth |
|---------|---------|-------|-----------|
| Strategy | Candidate generation | TID list intersections | FP-tree compression |
| Efficiency | Slow for large data | Faster for dense data | Fastest overall |
| Memory | Multiple scans | TID lists | Compact FP-tree |
| Best For | Small datasets | Dense datasets | Large-scale datasets |

---

### 🔹 Advantages
- Very fast and scalable.  
- Requires fewer database scans (usually 2).  
- Handles large datasets efficiently.  

### 🔹 Limitations
- FP-tree can be large if dataset has many unique items.  
- More complex to implement compared to Apriori.  

---

✅ **In short:** FP-Growth is a highly efficient algorithm for frequent pattern mining, using FP-trees to compress data and recursively grow itemsets, making it ideal for large-scale association rule learning.

---

Here’s a **Python implementation of FP-Growth** using the `mlxtend.frequent_patterns` library. This shows how to mine frequent itemsets and generate association rules efficiently:

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Sample dataset (transactions)
dataset = [
    ['Milk', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Bread', 'Beer']
]

# Step 2: Convert dataset into one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("Transaction DataFrame:")
print(df)

# Step 3: Apply FP-Growth to find frequent itemsets
frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

### 🔹 What happens here:
1. **Dataset:** Transactions are defined (like market basket data).  
2. **Encoding:** Convert transactions into a binary matrix (items present = 1, absent = 0).  
3. **FP-Growth:** Finds frequent itemsets that meet the minimum support threshold using FP-tree compression.  
4. **Association Rules:** Generates rules with confidence and lift values.  

---

### 🔹 Example Output (simplified)
- Frequent itemset: `{Milk, Diaper}` with support ≈ 0.6  
- Rule: `{Diaper, Beer} → {Milk}` with confidence ≈ 0.75 and lift > 1  

---

✅ **In short:** FP-Growth is faster and more scalable than Apriori or Eclat, especially for large datasets, because it compresses transactions into an FP-tree and mines patterns without generating candidates.

---

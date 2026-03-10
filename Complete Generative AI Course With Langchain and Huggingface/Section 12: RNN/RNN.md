### 📝 One‑Hot Encoding Example
Sentence: **“The food is good”**  
- Remove stop word **“is”**  
- Vocabulary = {The, food, good}  

Encoding:
- **The** → `[1 0 0]`  
- **food** → `[0 1 0]`  
- **good** → `[0 0 1]`  

---

👉 **In short:**  
One‑hot encoding turns each word into a vector where **only one position is “1”** (the word’s slot in the vocabulary), and all others ar

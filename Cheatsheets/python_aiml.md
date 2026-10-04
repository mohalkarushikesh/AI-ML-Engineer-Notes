# Python Cheat Sheet for AI/ML Engineers (In-Depth)

## 1. Python Basics

### Variables
```python
name = "Rushikesh"
age = 25
salary = 50000.0
is_active = True
```

### Input/Output
```python
name = input("Enter name: ")
print(f"Hello {name}")
```

### Type Checking
```python
type(age)
isinstance(age, int)
```

## 2. Data Structures

### List
```python
nums = [1, 2, 3, 4]
nums.append(5)
nums.pop()
nums.remove(2)
nums[0]
nums[-1]
nums[1:3]
squared = [x**2 for x in nums]
```

### Tuple (Immutable)
```python
point = (10, 20)
x, y = point
```

### Set
```python
s = {1, 2, 3}
s.add(4)
s.remove(1)
a = {1,2,3}
b = {3,4,5}
a.union(b)
a.intersection(b)
```

### Dictionary
```python
person = {
    "name": "Rushikesh",
    "age": 25
}

person["name"]
person.get("salary", 0)

for k, v in person.items():
    print(k, v)
```

## 3. Control Statements

### If Else
```python
if score > 90:
    print("A")
elif score > 75:
    print("B")
else:
    print("C")
```

### Loops
#### For Loop
```python
for i in range(5):
    print(i)
```

#### Enumerate
```python
for idx, value in enumerate(nums):
    print(idx, value)
```

#### Zip
```python
names = ["A", "B"]
ages = [20, 30]
for n, a in zip(names, ages):
    print(n, a)
```

## 4. Functions
```python
def add(a, b):
    return a + b

result = add(10, 20)
```

### Lambda
```python
square = lambda x: x**2
```

### *args and **kwargs
```python
def func(*args, **kwargs):
    print(args)
    print(kwargs)

func(1,2,3,name="AI")
```

## 5. OOP for ML Projects
### Class
```python
class Employee:
    def __init__(self, name):
        self.name = name

    def display(self):
        print(self.name)

obj = Employee("Rushikesh")
obj.display()
```

### Inheritance
```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        print("Bark")
```

## 6. Exception Handling
```python
try:
    result = 10/0
except ZeroDivisionError:
    print("Cannot divide")
finally:
    print("Executed")
```

## 7. File Handling
### Read
```python
with open("data.txt", "r") as f:
    content = f.read()
```

### Write
```python
with open("data.txt", "w") as f:
    f.write("Hello")
```

### JSON
```python
import json
with open("sample.json") as f:
    data = json.load(f)
```

## 8. List, Map, Filter, Reduce
### Map
```python
nums = [1,2,3]
list(map(lambda x:x*2, nums))
```
### Filter
```python
list(filter(lambda x:x%2==0, nums))
```
### Reduce
```python
from functools import reduce
reduce(lambda a,b:a+b, nums)
```

## 9. NumPy (Most Important)
```python
import numpy as np
```
### Array Creation
```python
a = np.array([1,2,3])
np.zeros((3,3))
np.ones((2,2))
np.arange(0,10,2)
np.linspace(0,1,5)
np.random.rand(3,3)
```
### Shape Operations
```python
a.shape
a.reshape(3,1)
a.flatten()
a.T
```
### Statistics
```python
np.mean(a)
np.median(a)
np.std(a)
np.var(a)
np.max(a)
np.min(a)
np.sum(a)
```
### Broadcasting
```python
a + 10
```

## 10. Pandas
```python
import pandas as pd
```
### Read Data
```python
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")
```
### Exploration
```python
df.head()
df.tail()
df.shape
df.info()
df.describe()
```
### Selection
```python
df["salary"]
df[["name", "salary"]]
df.iloc[0]
df.loc[0]
```
### Missing Values
```python
df.isnull().sum()
df.dropna()
df.fillna(0)
df.fillna(df.mean())
```
### Group By
```python
df.groupby("department")["salary"].mean()
```
### Sorting
```python
df.sort_values("salary", ascending=False)
```

# 11. Matplotlib

```python
import matplotlib.pyplot as plt
```

## Line Plot

```python
plt.plot(x, y)

plt.xlabel("x")
plt.ylabel("y")

plt.show()
```

## Histogram

```python
plt.hist(data)
```

## Scatter Plot

```python
plt.scatter(x, y)
```

---

# 12. Seaborn

```python
import seaborn as sns
```

## Heatmap

```python
sns.heatmap(df.corr())
```

## Distribution

```python
sns.histplot(data)
```

## Pairplot

```python
sns.pairplot(df)
```

---

# 13. Scikit-Learn

## Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)
```

## Classification

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)
```

## Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy_score(y_test, pred)

precision_score(y_test, pred)

recall_score(y_test, pred)

f1_score(y_test, pred)
```

---

# 14. Deep Learning (PyTorch)

## Tensor

```python
import torch

x = torch.tensor([1,2,3])

x.shape
```

## GPU

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

## Neural Network

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
```

---

# 15. Deep Learning (TensorFlow/Keras)

## Sequential Model

```python
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)
])
```

## Compile

```python
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

## Train

```python
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)
```

---

# 16. NLP Essentials

## Tokenization

```python
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
```

## Stopwords

```python
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
```

## Lemmatization

```python
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

lemma.lemmatize("running")
```

---

# 17. Hugging Face Transformers

## Load Model

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)

classifier(
    "AI is amazing"
)
```

## Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding = model.encode(
    "What is GenAI?"
)
```

---

# 18. Generative AI & LLMs

## OpenAI Style

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role":"user",
            "content":"Hello"
        }
    ]
)
```

## Prompt Template

```python
template = """
Answer based on context:
{context}

Question:
{question}
"""
```

---

# 19. RAG Essentials

## Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

## Embedding

```python
embedding_model.embed_query(text)
```

## Vector Search

```python
vectorstore.similarity_search(
    query,
    k=5
)
```

---

# 20. Agentic AI Essentials (LangGraph)

## State

```python
class State(TypedDict):
    query: str
    response: str
```

## Node

```python
def generate(state):
    return {
        "response": llm.invoke(
            state["query"]
        )
    }
```

## Graph

```python
builder.add_node(
    "generate",
    generate
)

builder.set_entry_point(
    "generate"
)
```

---

# Top 25 Python Functions Every AI/ML Engineer Uses

```python
len()
range()
enumerate()
zip()
map()
filter()
sum()
min()
max()
sorted()
round()
abs()
any()
all()
isinstance()
type()
print()
input()
open()
list()
dict()
set()
tuple()
int()
float()
str()
```

---

# AI/ML Interview Quick Revision

**Python:** List, Tuple, Set, Dict, OOP, Generators, Decorators, Context Manager

**NumPy:** Array, Broadcasting, Vectorization, Shape Manipulation

**Pandas:** Data Cleaning, GroupBy, Merge, Missing Values

**ML:** Train/Test Split, Overfitting, Bias-Variance, Precision, Recall, F1

**DL:** Epoch, Batch, Iteration, ReLU, Backpropagation, Adam

**NLP:** Tokenization, Embeddings, Transformers, Attention

**GenAI:** Prompting, RAG, Fine-Tuning, Vector DB, Agentic AI

**MLOps:** Docker, FastAPI, MLflow, Prometheus, Grafana, CI/CD, AWS EC2/S3/Bedrock

## Interview Focus Areas for 2+ Years AI/ML + GenAI Experience

**Python → NumPy → Pandas → Scikit-Learn → Deep Learning → Transformers → RAG → LangGraph → FastAPI → Docker → MLOps**

These are the highest-frequency topics asked in AI/ML, GenAI, Python Developer, and Applied AI Engineer interviews.

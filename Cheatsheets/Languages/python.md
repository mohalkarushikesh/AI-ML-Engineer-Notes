# 🐍 Python Cheatsheet

## 📌 Basics
```python
# Variables
x = 10          # integer
y = 3.14        # float
name = "Alice"  # string
flag = True     # boolean

# Data types
type(x)         # int
type(y)         # float
type(name)      # str
type(flag)      # bool
```

---

## 🔄 Control Flow
```python
# If-else
if x > 5:
    print("Big")
else:
    print("Small")

# Loops
for i in range(5):
    print(i)

while x > 0:
    x -= 1
```

---

## 📦 Data Structures
```python
# List
nums = [1, 2, 3]
nums.append(4)
nums[0]   # 1

# Tuple (immutable)
point = (3, 4)

# Set (unique values)
s = {1, 2, 2, 3}   # {1, 2, 3}

# Dictionary
person = {"name": "Alice", "age": 25}
person["age"]      # 25
```

---

## 🛠️ Functions
```python
def greet(name, msg="Hello"):
    return f"{msg}, {name}!"

greet("Bob")               # Hello, Bob!
greet("Bob", "Welcome")    # Welcome, Bob!
```

---

## 🎭 Classes & OOP
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return f"{self.name} makes a sound"

dog = Animal("Dog")
print(dog.speak())   # Dog makes a sound
```

---

## 📚 Modules & Imports
```python
import math
math.sqrt(16)   # 4.0

from random import randint
randint(1, 10)  # random int between 1 and 10
```

---

## 🧮 Useful Built-ins
```python
len([1,2,3])          # 3
sum([1,2,3])          # 6
max([1,2,3])          # 3
sorted([3,1,2])       # [1,2,3]
list(range(5))        # [0,1,2,3,4]
```

---

## ⚡ Advanced
```python
# List comprehension
squares = [i**2 for i in range(5)]   # [0,1,4,9,16]

# Lambda
add = lambda a, b: a + b
add(2, 3)   # 5

# Generators
def gen():
    for i in range(3):
        yield i
list(gen())   # [0,1,2]

# Decorators
def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@decorator
def hello():
    print("Hello")
```

---

## 🧵 Error Handling
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
finally:
    print("Done")
```

---

## 📂 File I/O
```python
# Write
with open("file.txt", "w") as f:
    f.write("Hello World")

# Read
with open("file.txt", "r") as f:
    data = f.read()
```

---

## 🌐 Popular Libraries
- **NumPy** → numerical computing  
- **Pandas** → data analysis  
- **Matplotlib/Seaborn** → visualization  
- **Scikit-learn** → machine learning  
- **TensorFlow/PyTorch** → deep learning  

---

👉 This cheatsheet covers **basics → advanced → practical tools**.  

## 📌 Keras Cheat Sheet

### 🔹 Imports
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

---

### 🔹 Core Concepts
- **Sequential API** → simple linear stack of layers  
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])
```

- **Functional API** → flexible graph of layers (multi‑input/output, shared layers)  
```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

---

### 🔹 Common Layers
- `Dense(units, activation)` → fully connected layer  
- `Conv2D(filters, kernel_size, activation)` → convolutional layer  
- `MaxPooling2D(pool_size)` → downsampling  
- `Dropout(rate)` → regularization  
- `Flatten()` → reshape for dense layers  
- `Embedding(input_dim, output_dim)` → word embeddings  

---

### 🔹 Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### 🔹 Training & Evaluation
```python
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.evaluate(x_test, y_test)
predictions = model.predict(x_new)
```

---

### 🔹 Optimizers
- `keras.optimizers.SGD(lr=0.01, momentum=0.9)`  
- `keras.optimizers.Adam(lr=0.001)`  
- `keras.optimizers.RMSprop(lr=0.001)`  

---

### 🔹 Callbacks
```python
callbacks = [
    keras.callbacks.EarlyStopping(patience=3),
    keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
]
```

---

### 🔹 Saving & Loading
```python
model.save("my_model.h5")
loaded_model = keras.models.load_model("my_model.h5")
```

---

### 🔹 Dataset Utilities
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

---

## 📚 Sources
- [GeeksforGeeks Keras Cheatsheet (2025)](https://www.geeksforgeeks.org/blogs/keras-cheatsheet/)  
- [Cheat Sheets Hero – Keras](https://cheatsheetshero.com/user/all/599-keras-cheat-sheet)

---

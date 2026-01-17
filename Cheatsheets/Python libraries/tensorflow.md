## 📌 TensorFlow Cheat Sheet

### 🔹 Imports
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

---

### 🔹 Tensors
- **Create tensors**
```python
x = tf.constant([1, 2, 3])          # constant
x = tf.zeros([2, 3])                # 2x3 zeros
x = tf.ones([3, 2])                 # 3x2 ones
x = tf.random.normal([2, 2])        # random normal
```

- **Properties**
```python
x.shape      # dimensions
x.dtype      # data type
tf.rank(x)   # number of dimensions
```

- **Operations**
```python
y = tf.add(x, x)                    # addition
z = tf.matmul(x, tf.transpose(x))   # matrix multiplication
r = tf.reshape(x, [3, 2])           # reshape
```

---

### 🔹 Neural Networks (Keras API)
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])
```

---

### 🔹 Compile & Train
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10, batch_size=32)
```

---

### 🔹 Evaluation & Prediction
```python
model.evaluate(test_data, test_labels)
predictions = model.predict(new_data)
```

---

### 🔹 Optimizers
```python
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
keras.optimizers.Adam(learning_rate=0.001)
```

---

### 🔹 Saving & Loading Models
```python
model.save("my_model.h5")                 # save
loaded_model = keras.models.load_model("my_model.h5")  # load
```

---

### 🔹 GPU Utilization
```python
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

---

### 🔹 TensorFlow Datasets
```python
import tensorflow_datasets as tfds
dataset, info = tfds.load('mnist', as_supervised=True, with_info=True)
```

---

## 📚 Sources
- [GeeksforGeeks TensorFlow Cheat Sheet (2025)](https://www.geeksforgeeks.org/blogs/tensorflow-cheat-sheet)  
- [Cheat Sheets Hero – TensorFlow](https://cheatsheetshero.com/user/all/670-tensorflow-cheat-sheet)  
- [GitHub TensorFlow Cheat Sheet](https://github.com/Mayumiwandi/TensorFlow-Cheat-Sheet)

---

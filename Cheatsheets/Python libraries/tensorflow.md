# TensorFlow / Keras Cheatsheet

---

## Installation

```bash
# Latest stable
pip install tensorflow

# GPU support (Linux/Windows — CUDA handled automatically in TF 2.x)
pip install tensorflow[and-cuda]

# Lightweight (CPU only, smaller package)
pip install tensorflow-cpu

# Check version & GPU
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## Tensors

### Creating Tensors
```python
import tensorflow as tf

tf.constant([1, 2, 3])                    # From list
tf.constant([[1.0, 2.0], [3.0, 4.0]])    # 2D tensor

tf.zeros([3, 4])                          # All zeros
tf.ones([3, 4])                           # All ones
tf.fill([3, 4], 7.0)                      # Fill with value
tf.eye(4)                                 # Identity matrix

tf.random.uniform([3, 4], minval=0, maxval=1)   # Uniform [0, 1)
tf.random.normal([3, 4], mean=0, stddev=1)      # Standard normal
tf.random.set_seed(42)                           # Reproducibility

tf.range(0, 10, delta=2)                  # [0, 2, 4, 6, 8]
tf.linspace(0.0, 1.0, 5)                  # [0, .25, .5, .75, 1]

tf.zeros_like(x)                          # Same shape/dtype as x
tf.ones_like(x)
```

### Tensor Properties
```python
x = tf.random.normal([3, 4])

x.shape          # TensorShape([3, 4])
x.ndim           # 2
x.dtype          # tf.float32
x.device         # /job:localhost/.../device:CPU:0
x.numpy()        # Convert to NumPy array
len(x)           # 3 (first dimension)
tf.size(x)       # 12 (total elements)
```

### Data Types
| dtype | Description |
|-------|-------------|
| `tf.float32` | Default float |
| `tf.float16` | Half precision |
| `tf.bfloat16` | Brain float |
| `tf.float64` | Double precision |
| `tf.int32` | 32-bit integer |
| `tf.int64` | 64-bit integer |
| `tf.bool` | Boolean |
| `tf.string` | String tensor |

```python
tf.cast(x, tf.float16)    # Cast to dtype
tf.cast(x, tf.int32)
```

---

## Tensor Operations

### Shape Manipulation
```python
x = tf.random.normal([2, 3, 4])

tf.reshape(x, [6, 4])           # Reshape
tf.squeeze(x)                   # Remove dims of size 1
tf.squeeze(x, axis=0)           # Remove specific dim
tf.expand_dims(x, axis=0)       # Add dim → (1, 2, 3, 4)
tf.transpose(x, perm=[2, 0, 1]) # Reorder dimensions
tf.transpose(x)                 # Reverse all dims
```

### Indexing & Slicing
```python
x = tf.random.normal([4, 5])

x[0]               # First row
x[:, 1]            # Second column
x[1:3, 2:4]        # Slice
x[x > 0]           # Boolean mask → 1D

tf.where(x > 0, x, tf.zeros_like(x))   # Conditional select
tf.gather(x, indices=[0, 2], axis=0)   # Gather rows
tf.gather_nd(x, indices=[[0,1],[2,3]]) # Gather by index pairs
```

### Math Operations
```python
# Element-wise
x + y,  x - y,  x * y,  x / y
x ** 2,  tf.sqrt(x)
tf.abs(x),  tf.exp(x),  tf.math.log(x)
tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)

# Matrix ops
x @ y                         # Matrix multiply
tf.matmul(x, y)               # Equivalent
tf.linalg.matmul(x, y, transpose_b=True)

# Reductions
tf.reduce_sum(x)              # Sum all
tf.reduce_sum(x, axis=0)      # Along axis
tf.reduce_mean(x),  tf.reduce_max(x),  tf.reduce_min(x)
tf.argmax(x, axis=1)
tf.argmin(x, axis=1)
tf.math.top_k(x, k=3)        # Top-k values & indices

# Comparison
tf.equal(x, y)
tf.greater(x, y),  tf.less(x, y)
tf.reduce_all(tf.equal(x, y)) # All elements equal?
```

### Concatenation & Stacking
```python
tf.concat([a, b, c], axis=0)      # Concatenate along existing axis
tf.stack([a, b, c], axis=0)       # Stack along NEW axis
tf.split(x, num_or_size_splits=3, axis=0)  # Split into chunks
tf.unstack(x, axis=0)             # Unstack to list of tensors
```

---

## Variables

```python
# tf.Variable — mutable, tracked by autograd
w = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.zeros([3]))

# Update
w.assign(new_value)
w.assign_add(delta)
w.assign_sub(delta)

# Properties
w.trainable        # True by default
w.numpy()          # Convert to NumPy
```

---

## GPU Management

```python
# List devices
tf.config.list_physical_devices('GPU')
tf.config.list_physical_devices('CPU')

# Limit GPU memory growth (prevents OOM)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory to fixed amount
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # MB
)

# Place ops on specific device
with tf.device('/GPU:0'):
    result = tf.matmul(a, b)

with tf.device('/CPU:0'):
    result = tf.matmul(a, b)

# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()  # Sync multi-GPU
with strategy.scope():
    model = build_model()
```

---

## Automatic Differentiation

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)   # dy/dx = 6.0

# Multiple gradients
with tf.GradientTape() as tape:
    y = model(x)
    loss = loss_fn(y, targets)

grads = tape.gradient(loss, model.trainable_variables)

# Persistent tape (multiple gradient calls)
with tf.GradientTape(persistent=True) as tape:
    y = x ** 2 + x ** 3

tape.gradient(y, x)            # First call
tape.gradient(y, x)            # Second call (works with persistent=True)
del tape                       # Free resources

# Higher-order gradients (nested tapes)
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = x ** 3
    dy_dx = t1.gradient(y, x)
d2y_dx2 = t2.gradient(dy_dx, x)
```

---

## Building Models

### Sequential API
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
```

### Functional API (multi-input/output, skip connections)
```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Subclassing API (most flexible)
```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense2(x)

model = MyModel()
```

### Common Layers
```python
# Core
layers.Dense(units, activation=None, use_bias=True)
layers.Activation('relu')
layers.Lambda(lambda x: x * 2)

# Convolutional
layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation='relu')
layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')  # Upsample
layers.DepthwiseConv2D(kernel_size, strides=1, padding='same')
layers.SeparableConv2D(filters, kernel_size)

# Pooling
layers.MaxPooling2D(pool_size=2, strides=2)
layers.AveragePooling2D(pool_size=2)
layers.GlobalAveragePooling2D()             # Spatial → vector
layers.GlobalMaxPooling2D()

# Normalization
layers.BatchNormalization()
layers.LayerNormalization()
layers.GroupNormalization(groups=8)

# Recurrent
layers.LSTM(units, return_sequences=True, return_state=False)
layers.GRU(units, return_sequences=True)
layers.Bidirectional(layers.LSTM(units))
layers.SimpleRNN(units)

# Attention
layers.MultiHeadAttention(num_heads=8, key_dim=64)
layers.Attention()

# Embedding
layers.Embedding(input_dim, output_dim, mask_zero=False)

# Regularization
layers.Dropout(rate=0.5)
layers.SpatialDropout2D(rate=0.5)          # Drops entire feature maps
layers.GaussianNoise(stddev=0.1)

# Shape
layers.Flatten()
layers.Reshape(target_shape)
layers.Concatenate(axis=-1)
layers.Add()
layers.Multiply()
layers.UpSampling2D(size=2)
layers.ZeroPadding2D(padding=1)
```

---

## Compiling & Training

### compile()
```python
model.compile(
    optimizer='adam',                              # or optimizer object
    loss='sparse_categorical_crossentropy',        # or loss object
    metrics=['accuracy', 'AUC'],
)

# With objects for more control
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

### fit()
```python
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),    # or validation_split=0.2
    callbacks=callbacks_list,
    class_weight={0: 1.0, 1: 5.0},    # For imbalanced classes
    sample_weight=sample_weights,
    shuffle=True,
    verbose=1,
)

# Access history
history.history['loss']
history.history['val_accuracy']
```

### evaluate() & predict()
```python
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

predictions = model.predict(x_test, batch_size=64)   # Returns numpy array
probabilities = predictions                           # After softmax
classes = predictions.argmax(axis=-1)                # Class indices
```

---

## Loss Functions

```python
# Regression
keras.losses.MeanSquaredError()
keras.losses.MeanAbsoluteError()
keras.losses.Huber(delta=1.0)
keras.losses.LogCosh()

# Classification
keras.losses.BinaryCrossentropy(from_logits=False)
keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
keras.losses.SparseCategoricalCrossentropy(from_logits=True)
keras.losses.KLDivergence()
keras.losses.CosineSimilarity()

# String shortcuts for compile()
'mse', 'mae', 'binary_crossentropy',
'categorical_crossentropy', 'sparse_categorical_crossentropy'
```

---

## Optimizers

```python
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01)
keras.optimizers.RMSprop(learning_rate=1e-3)
keras.optimizers.Adagrad(learning_rate=0.01)

# String shortcuts
'sgd', 'adam', 'adamw', 'rmsprop', 'adagrad'
```

### LR Schedules
```python
# Exponential decay
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.96
)

# Cosine decay
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=10000
)

# Piecewise constant
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[3000, 6000], values=[1e-3, 1e-4, 1e-5]
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

---

## Callbacks

```python
callbacks = [
    # Save best model
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
    ),

    # Stop early
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
    ),

    # Reduce LR on plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    ),

    # TensorBoard logging
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        update_freq='epoch',
    ),

    # CSV logging
    keras.callbacks.CSVLogger('training_log.csv'),

    # LR warmup (custom)
    keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
            model.optimizer.learning_rate.assign(min(1e-3, epoch * 1e-4))
    ),
]
```

### Custom Callback
```python
class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.99:
            print("\nTarget accuracy reached, stopping.")
            self.model.stop_training = True
```

---

## Data Pipelines (tf.data)

```python
# From tensors
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# From generator
dataset = tf.data.Dataset.from_generator(
    generator_fn, output_signature=(
        tf.TensorSpec(shape=(784,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# From files
dataset = tf.data.TFRecordDataset(filenames)

# Pipeline
dataset = (
    dataset
    .shuffle(buffer_size=10000, seed=42)
    .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)    # Overlap CPU prep & GPU training
    .cache()                        # Cache in memory or to disk
)

# Cache to disk (large datasets)
dataset = dataset.cache('./cache_dir')

# Iterate
for x_batch, y_batch in dataset:
    ...
```

### tf.data Performance Tips
| Tip | Method |
|-----|--------|
| Prefetch batches | `.prefetch(tf.data.AUTOTUNE)` |
| Parallel map | `.map(fn, num_parallel_calls=tf.data.AUTOTUNE)` |
| Cache dataset | `.cache()` |
| Vectorize map | Apply ops to full batch, not single sample |
| Interleave files | `.interleave(tf.data.TFRecordDataset, cycle_length=4)` |

---

## Custom Training Loop

```python
optimizer = keras.optimizers.Adam(1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = keras.metrics.SparseCategoricalAccuracy()
val_acc = keras.metrics.SparseCategoricalAccuracy()

@tf.function                          # Compile to graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(y, logits)
    return loss

@tf.function
def val_step(x, y):
    logits = model(x, training=False)
    val_acc.update_state(y, logits)
    return loss_fn(y, logits)

for epoch in range(epochs):
    train_acc.reset_state()
    val_acc.reset_state()

    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)

    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)

    print(f"Epoch {epoch+1} | "
          f"Train Acc: {train_acc.result():.4f} | "
          f"Val Acc: {val_acc.result():.4f}")
```

---

## Saving & Loading

```python
# Keras native format (recommended)
model.save('my_model.keras')
model = keras.models.load_model('my_model.keras')

# SavedModel format (TF serving compatible)
model.save('saved_model_dir/')
model = keras.models.load_model('saved_model_dir/')

# Weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')

# JSON architecture + weights separately
json_config = model.to_json()
model = keras.models.model_from_json(json_config)
model.load_weights('weights.h5')
```

---

## Mixed Precision

```python
from tensorflow.keras import mixed_precision

# Enable globally (fp16 compute, fp32 storage)
mixed_precision.set_global_policy('mixed_float16')

# Build model normally — layers auto use fp16
model = build_model()

# Ensure output layer is float32
outputs = layers.Dense(10, dtype='float32')(x)  # Softmax needs fp32

# Loss scaling (handled automatically in model.fit)
# For custom loops:
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
with tf.GradientTape() as tape:
    loss = model(x)
    scaled_loss = optimizer.get_scaled_loss(loss)
grads = tape.gradient(scaled_loss, model.trainable_variables)
grads = optimizer.get_unscaled_gradients(grads)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## Transfer Learning

```python
# Load pretrained base
base = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,     # Remove classification head
    input_shape=(224, 224, 3)
)

# Freeze base
base.trainable = False

# Build new model
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.resnet50.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Fine-tune: unfreeze top layers
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='...', metrics=['accuracy'])
```

### Available Keras Applications
```python
keras.applications.ResNet50 / ResNet101 / ResNet152
keras.applications.EfficientNetB0 / EfficientNetV2S
keras.applications.VGG16 / VGG19
keras.applications.InceptionV3
keras.applications.MobileNetV2 / MobileNetV3Large
keras.applications.Xception
keras.applications.DenseNet121 / DenseNet201
keras.applications.NASNetLarge
```

---

## Metrics

```python
# Classification
keras.metrics.Accuracy()
keras.metrics.BinaryAccuracy()
keras.metrics.CategoricalAccuracy()
keras.metrics.SparseCategoricalAccuracy()
keras.metrics.AUC(curve='ROC')
keras.metrics.Precision()
keras.metrics.Recall()
keras.metrics.F1Score(average='macro')

# Regression
keras.metrics.MeanSquaredError()
keras.metrics.MeanAbsoluteError()
keras.metrics.RootMeanSquaredError()

# Usage in custom loops
metric = keras.metrics.SparseCategoricalAccuracy()
metric.update_state(y_true, y_pred)
print(metric.result().numpy())
metric.reset_state()
```

---

## Regularization

```python
# Kernel regularizers
layers.Dense(64,
    kernel_regularizer=keras.regularizers.L2(0.01),
    activity_regularizer=keras.regularizers.L1(0.001)
)
keras.regularizers.L1L2(l1=0.01, l2=0.01)

# Dropout
layers.Dropout(0.5)

# Batch Normalization
layers.BatchNormalization(momentum=0.99, epsilon=0.001)

# Weight constraints
layers.Dense(64, kernel_constraint=keras.constraints.MaxNorm(3.0))
```

---

## TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir ./logs
```

```python
# Log custom scalars
writer = tf.summary.create_file_writer('./logs/custom')
with writer.as_default():
    for step, value in enumerate(my_values):
        tf.summary.scalar('my_metric', value, step=step)
        tf.summary.histogram('weights', model.layers[1].weights[0], step=step)
        tf.summary.image('sample', img_tensor, step=step)
        writer.flush()
```

---

## Model Deployment

```python
# TensorFlow Serving (SavedModel)
model.save('serving_model/1/')  # Versioned directory

# TFLite (mobile/edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]    # Quantize
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# TFLite inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_idx, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_idx)

# ONNX export
pip install tf2onnx
python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx
```

---

## @tf.function — Graph Mode

```python
# Compile Python function to TF graph (faster, exportable)
@tf.function
def compute(x, y):
    return tf.matmul(x, y) + tf.reduce_sum(x)

# Trace with input signature (avoids retracing)
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    tf.TensorSpec(shape=[None], dtype=tf.int32),
])
def train_step(x, y):
    ...

# Avoid Python side effects inside @tf.function
# Use tf.print instead of print
# Use tf.TensorArray instead of Python lists
```

---

## Useful Utilities

```python
# Reproducibility
tf.random.set_seed(42)
import numpy as np; np.random.seed(42)

# Tensor ↔ NumPy
arr = tensor.numpy()                     # Tensor → NumPy
t = tf.constant(arr)                     # NumPy → Tensor

# Model summary
model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file='model.png')

# Count parameters
model.count_params()

# Layer outputs (feature extraction)
layer_model = keras.Model(inputs=model.input,
                          outputs=model.get_layer('dense').output)
features = layer_model.predict(x)

# Gradient clipping
optimizer = keras.optimizers.Adam(clipnorm=1.0)    # Clip by norm
optimizer = keras.optimizers.Adam(clipvalue=0.5)   # Clip by value

# Check GPU memory
tf.config.experimental.get_memory_info('GPU:0')
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `InvalidArgumentError: logits and labels must have same shape` | Shape mismatch | Check output layer units vs num classes |
| `ResourceExhaustedError: OOM` | GPU out of memory | Reduce batch size, enable memory growth |
| `ValueError: Input 0 is incompatible with layer` | Wrong input shape | Check `Input(shape=...)` matches your data |
| `UnimplementedError: Cast string to float` | Data type mismatch | `tf.cast()` your tensors |
| `tf.function retracing` | Different input shapes | Use `input_signature` or `tf.TensorSpec` |
| Loss is `nan` | Exploding gradients / bad lr | Use `clipnorm`, lower lr, check data normalization |
| Slow training | Not using `@tf.function` or `.prefetch` | Decorate train step, add `.prefetch(AUTOTUNE)` |
| `AttributeError` on `model.fit` | Using `training=True/False` wrong | Pass `training` arg in subclass `call()` |

---

## Keras 3 (Multi-Backend)

```python
# Keras 3 supports TensorFlow, JAX, and PyTorch backends
pip install keras

import os
os.environ['KERAS_BACKEND'] = 'jax'  # or 'torch' or 'tensorflow'

import keras
# All keras.layers, keras.Model, keras.optimizers work the same across backends
```

---

*Last updated: 2025 · TensorFlow 2.x / Keras 3 · Always pin versions in production.*

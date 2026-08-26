Here are hands-on deep learning practice exercises organized by level. Each is a mini-project you can build, train, and evaluate end-to-end (PyTorch or TensorFlow/Keras both work).

## Beginner

1. **Perceptron from scratch** — Implement a single neuron with NumPy to solve a linearly separable problem (e.g., AND/OR gates); code the forward pass and weight updates by hand.
2. **MNIST digit classifier** — Build a simple feedforward network (a few dense layers) to classify handwritten digits; report accuracy and plot the loss curve.
3. **Activation function comparison** — Train the same small network with sigmoid, tanh, and ReLU; compare convergence speed and final accuracy.
4. **Manual gradient descent** — Fit a linear/logistic model using a hand-written training loop so you understand forward pass → loss → backward pass → update.
5. **Fashion-MNIST with a small MLP** — Classify clothing images and experiment with the number of layers and neurons.
6. **Overfitting & regularization demo** — Show overfitting on a small dataset, then add dropout and L2 regularization and observe the effect on validation loss.
7. **Learning rate exploration** — Train one model with several learning rates (too high, too low, just right) and plot how each affects the loss curve.

## Medium

1. **CNN for image classification** — Build a convolutional network for CIFAR-10; add pooling, batch normalization, and dropout, then compare against your MLP baseline.
2. **Transfer learning** — Fine-tune a pre-trained model (ResNet, VGG, MobileNet) on a small custom image dataset; compare against training from scratch.
3. **Data augmentation study** — Apply flips, rotations, crops, and color jitter; quantify how augmentation affects generalization.
4. **RNN/LSTM for sequences** — Train an LSTM for a task like sentiment classification or time-series prediction; compare LSTM vs. GRU vs. vanilla RNN.
5. **Autoencoder** — Build an autoencoder for image denoising or dimensionality reduction; visualize reconstructions and the latent space.
6. **Custom training loop** — Rewrite a Keras `.fit()` workflow as an explicit loop (or use PyTorch) with manual gradient computation, metrics, and checkpointing.
7. **Optimizer comparison** — Train the same model with SGD, SGD+momentum, RMSProp, and Adam; compare convergence and final performance.
8. **Character-level text generation** — Train an RNN to generate text character by character (e.g., names, poetry) and sample from it.
9. **Hyperparameter tuning** — Use Keras Tuner or Optuna to search over layers, units, learning rate, and batch size.

## Advanced

1. **Build a transformer from scratch** — Implement self-attention, multi-head attention, and positional encoding; train a small transformer on a sequence task.
2. **Fine-tune a pre-trained transformer** — Use Hugging Face to fine-tune BERT/DistilBERT for classification or GPT-style models for generation; report proper metrics.
3. **GAN** — Train a DCGAN to generate images (e.g., faces or digits); handle training instability and visualize samples across epochs.
4. **Variational Autoencoder (VAE)** — Build a VAE, sample from the latent distribution, and interpolate between points in latent space.
5. **Object detection / segmentation** — Fine-tune YOLO or a Faster R-CNN / U-Net model on a custom dataset with bounding boxes or masks.
6. **Seq2seq with attention** — Build an encoder-decoder model with attention for translation or summarization.
7. **Self-supervised / contrastive learning** — Implement a simplified SimCLR-style setup: learn representations without labels, then evaluate via a linear probe.
8. **Neural style transfer** — Combine the content of one image with the style of another using a pre-trained CNN's feature maps.
9. **Distributed / mixed-precision training** — Scale training across GPUs (or use mixed precision) and measure the speedup and any accuracy trade-offs.
10. **Model deployment & optimization** — Export a trained model (ONNX / TorchScript / TF Lite), apply quantization or pruning, and serve it via an API; measure size and latency reductions.
11. **Reinforcement learning intro** — Train a DQN or policy-gradient agent on a Gym environment (e.g., CartPole) and plot the reward curve.

A good approach is to complete one project per level fully (data → architecture → training loop → evaluation → short write-up) before advancing, since debugging training dynamics is where most of the real learning happens.

Want me to expand any single exercise into a full step-by-step project with starter code, architecture details, and a dataset suggestion?

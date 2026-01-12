# 🔗 Multilayer Perceptron (MLP) — Full Notes

## 📌 Definition
- A **Multilayer Perceptron (MLP)** is a type of **feedforward artificial neural network (ANN)**.  
- It consists of **three or more layers**: input, hidden, and output.  
- Each neuron computes a **weighted sum + bias**, passes it through a **nonlinear activation function**, and forwards the result.  
- MLPs are often called the **vanilla neural network** because they are the simplest deep learning architecture.

---

## 🧬 Architecture
1. **Input Layer**  
   - Accepts feature vectors ($x_1, x_2, ..., x_n$).  

2. **Hidden Layers**  
   - One or more fully connected layers.  
   - Each neuron computes:  

     $$z = \sum w_i x_i + b$$

     $$a = f(z)$$

     where $f$ is an activation function (ReLU, Sigmoid, Tanh).  

3. **Output Layer**  
   - Produces final prediction (classification probabilities or regression values).  

<img width='700' height='500' src="https://github.com/user-attachments/assets/aba18c16-742d-461d-aa92-506a6339114a" /> 

---

## 🔧 Working
1. **Forward Propagation**  
   - Inputs → weighted sums → activation → outputs.  

2. **Loss Function**  
   - Measures prediction error (e.g., MSE, Cross‑Entropy).  

3. **Backpropagation**  
   - Gradients of loss w.r.t weights are computed.  
   - Weights updated using optimizers (SGD, Adam).  

---

## 📐 Key Formulas
- **Neuron Output**:  

  $$a = f\left(\sum w_i x_i + b\right)$$  

- **Error (MSE example)**:  

  $$L = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$  

- **Weight Update (Gradient Descent)**:  

  $$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$  

---

## ✅ Advantages
- Learns **nonlinear relationships**.  
- Flexible for regression and classification tasks.  
- Universal Approximation Theorem → can approximate any continuous function.  
- Foundation for deeper architectures (CNNs, RNNs, Transformers).  

---

## ⚠️ Limitations
- Requires large datasets.  
- Computationally expensive.  
- Prone to overfitting (needs regularization like dropout, L2).  
- Harder to interpret compared to simpler models.  

---

## 📌 Applications
- Image recognition (basic tasks).  
- Natural Language Processing (before RNNs/Transformers).  
- Financial forecasting.  
- Medical diagnosis.  
- Recommendation systems.  

---

## 📝 Summary
- **MLP = Vanilla feedforward ANN**.  
- Structure: Input → Hidden → Output layers.  
- Training: **Forward propagation + Backpropagation + Optimizer**.  
- Activation functions introduce nonlinearity.  
- Widely used as the **foundation of deep learning**.  

---

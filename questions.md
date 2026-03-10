### why we add bias in neural network ?

Think of bias in a neural network like the **intercept in a straight‑line equation** (`y = mx + c`).  

- Without bias, the neuron’s output always passes through the origin (0,0).  
- With bias, you can **shift the activation function up or down**, which lets the neuron fit data better.  
- In simple terms: **bias gives flexibility**. It allows the model to learn patterns that don’t start at zero.  

👉 Example:  
- Imagine trying to fit a line to points that don’t cross the origin.  
- If you only have slope (`m`), you can’t shift the line.  
- Adding bias (`c`) lets you move the line up or down so it matches the data.  

So, **bias helps the network learn more general and accurate mappings** instead of being stuck passing through zero.  

---

### Explain y = mx + b ?

Here’s the **simplest way** to understand \(y = mx + b\):

- \(m\) = slope → tells how steep the line is (how much \(y\) changes when \(x\) changes).  
- \(b\) = intercept → tells where the line crosses the \(y\)-axis (shifts the line up or down).  
- \(x\) = input, \(y\) = output.

### ✨ Example
Equation: \(y = 2x + 3\)  
- Slope = 2 → for every step in \(x\), \(y\) goes up by 2.  
- Intercept = 3 → when \(x = 0\), \(y = 3\).  

So the line starts at 3 on the \(y\)-axis and rises steeply.

👉 In short:  
**\(y = mx + b\) is just a straight line where \(m\) controls tilt and \(b\) controls vertical shift.**  

<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/9f9e3b76-2e92-420d-88d3-e41966f2cc0f" />


---

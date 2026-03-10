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

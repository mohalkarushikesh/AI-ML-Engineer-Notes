1. review intro to calculus 
2. ML gradients 
3. Integrals

- The **gradient** is a mathematical concept that represents the rate of change or slope of a function at a particular point. It indicates how much the function's output changes with respect to changes in its input variables. Gradients are crucial in optimization and machine learning, especially for training models.

- In calculus terms:
  - For a single-variable function, the gradient is equivalent to the derivative.
  - For a multi-variable function, the gradient is a vector containing partial derivatives with respect to each input variable.

In the context of machine learning:
  - Gradients are used to compute how the parameters (like m and b in your case) should be updated to minimize the loss function.
  - Optimizers like SGD rely on gradients to move parameters in the direction that reduces the loss.
- For example: If a loss function is f(x, y), its gradient at a point (x, y) is: $$ \nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) $$

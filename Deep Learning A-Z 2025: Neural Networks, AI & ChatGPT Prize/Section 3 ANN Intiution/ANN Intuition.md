- The Neuron 
    - Dendrites: function of dendrites is to receive information from other neurons, called pre-synaptic neurons
    - Axon: that carries nerve impulses away from the cell body 

    - weights : strength of connections between the nodes (neurons)

- Efficient Backdrop

- Activation Functions   
  - Threshold Function 
  - Sigmoid Function 
  - Rectifier Function 
  - Hyperbolic Tanget Function (tanh)

  - Additional Learning: Deep Sparce rectifier neural netoworks by Xavier Glorot (2011)
  - Additional Learning for cost functions: A list of cost functions used in neural networks, alongside applications by Cross Validated (2015) 

- NN Working ?
Composed of interconnected nodes called neurons, neural networks arrange these units in layers. Each neuron receives input from others, processes it, and transmits an output to other neurons. Connections between neurons have associated weights, signifying the connection strength.

- how NN learns ?
Neural networks learn through a process of iterative adjustments to their internal connections, called weights, based on the difference between their predictions and the actual values in a training dataset. This process, known as training, involves feeding the network with data, calculating the error in its predictions, and then adjusting the weights to minimize this error. 

- Gradient descent is an optimization algorithm used in machine learning to minimize a function, often a cost or loss function, by iteratively adjusting the function's parameters. 
It works by taking steps in the direction of the steepest descent (negative gradient) until a minimum value is reached. 

- Stochastic Gradient Descent (SGD) is an optimization algorithm used in machine learning, particularly when dealing with large datasets. 
It's a variation of Gradient Descent that updates model parameters using the gradient calculated from a single data point or a small batch of data points at each iteration. 
This approach makes it computationally efficient and suitable for large-scale datasets. 

- The primary difference between Gradient Descent (GD) and Stochastic Gradient Descent (SGD) lies in how they use the training data to update model parameters. GD uses the entire training dataset to calculate the gradient, while SGD uses only a single randomly selected data point or a small batch of data points in each iteration. This difference in data usage impacts their convergence behavior and computational efficiency.

  - Additional Learning A Neural network in 13 lines of python (part 2 gradient descent) by Andrew Trask 2015 
  - Additional Learning A Neural Network and deep learning by Michael Neilsen 2015 
  
- Backpropogation is a supervised learning algorithm used to train artificial neural networks. It's the process of adjusting the weights of connections in a neural network to minimize the difference between the network's predicted output and the desired output. Essentially, it propagates the error backward through the network, layer by layer, to calculate how much each weight contributed to the error and then adjusts those weights accordingly. 


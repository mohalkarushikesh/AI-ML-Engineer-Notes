- binary classification problem : sigmoid function 
- Multiclass classification : Softmax function 
- classification problem : cross entropy function in order to compare two probability ditributions this function is based on kullback leibler divergence 
	- DKL((P||Q)) divergence between two distributions P and Q 

**CNN : Convolution neural network** 
	**Convolution** is mathematical operation which transforms an input matrix by means of another matrix (kernel)
 		- In particular convolution is symmetric, associative, distributive 
		- layers are defined as:
			- kernel size: which is the dimention of linear filter 
			- stride: which no of input elements to be crossed in consecutive operations of the kernel
			- padding: which is the artificial enlargement of the input to allow the application of the filters on the borders of the input 
			- depth: which is the no of the different filters that one wish to synthesize (corresponds to the depth of the next layer)
      			- polling layers: In CNN it's common practice to alternate convolutional layers with pooling layers, where each neuron takes the mean and max in it's receptive field. This practice reduces the dimention of the output.
	**Correlation** is mathematical operation similar to covolution which does not flip the given kernal 		
   	
    	- if one does not flip the kernel then one computes the correlation between the two function 


**Deteriorated information** refers to information that has become less accurate, reliable, or useful over time. This can happen due to various factors, including the passage of time, changes in context, or errors in data collection or storage. 
The term implies a decline in the quality of information, making it potentially misleading or invalid. 

### ✅ Refined Definition:
**Deteriorated information** refers to data or knowledge that has lost its accuracy, reliability, or relevance over time. This degradation can result from factors such as:

- 📆 **Passage of time** — information becomes outdated or obsolete.
- 🔄 **Contextual changes** — shifts in circumstances render the information less applicable.
- 🧪 **Data corruption or loss** — due to technical issues, poor storage, or transmission errors.
- 🧍 **Human error** — mistakes in data entry, interpretation, or reporting.

As a result, deteriorated information may become misleading, incomplete, or invalid for decision-making or analysis.

**Question** : if the information is less accurate how then it is useful over time ?

Excellent question—and you're absolutely right to challenge that.

If information becomes **less accurate**, it typically becomes **less useful**, not more. The phrase “deteriorated information” implies that its **usefulness declines** over time, precisely because its **accuracy, relevance, or reliability** has degraded.

So to clarify:

> Information that is **less accurate** is generally **less useful**, especially for tasks that depend on precision, like decision-making, forecasting, or analysis.

---

### 🧠 But Here’s a Nuance:
Sometimes, even deteriorated information can retain **some** value—just not in the way it was originally intended.

#### 🔍 Example 1: Historical Context
Old weather records may be inaccurate by modern standards, but they still offer insight into climate trends over time.

#### 🧪 Example 2: Machine Learning
Noisy or partially incorrect data might still help train a model if the overall patterns are preserved.

#### 📰 Example 3: Journalism
An outdated news article might no longer reflect current events, but it can still be useful for understanding public sentiment or media framing at the time.

---

So you're right: **less accurate = less useful** in most cases. But in some contexts, even flawed or outdated information can offer **secondary value**—just not for its original purpose.

Classification task with tensorflow : 

Indentify and deal with overfitting through early stopping callbacks and Dropout Layers 

Early stopping : Keras can automatically stop tranning based on loss condition on validation data passed during the model.fit() call.

Dropout layers: can be added to layers to turn off neurons during tranning to prevent overfitting.

The rectified linear unit activation function (or ReLU, for short) transforms output using the following algorithm: 
If the input value is less than 0, return 0. If the input value is greater than or equal to 0, return the input value.


In a neural network, the batch size refers to the number of training examples used in one iteration to update the model's parameters. 
It essentially defines how many data samples are processed simultaneously before the model's weights are adjusted. 
A batch can be a single sample, or it can be a larger chunk of the training data. 

The .h5 format, also known as HDF5 (Hierarchical Data Format)

Precision: The proportion of correctly predicted positive instances out of all instances predicted as positive.

Recall: The proportion of correctly predicted positive instances out of all actual positive instances.

F1-Score: The harmonic mean of precision and recall, balancing both metrics in a single score.

Support: The number of actual occurrences of each class in the dataset.

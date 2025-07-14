
**Artificial Intelligence (AI)**  
- The broad field focused on creating machines that can perform tasks requiring human-like intelligence—such as reasoning, learning, and problem-solving.

**Machine Learning (ML)**  
- A subset of AI that enables systems to learn from data and improve performance over time without being explicitly programmed.

**Deep Learning**  
- A specialized branch of ML that uses neural networks with multiple layers to learn complex patterns, especially in large datasets like images, audio, or text.

---


**List**  
- Ordered, mutable collection of items.  
- Defined with square brackets: `[]`  
- Allows duplicates.  
- Example: `my_list = [1, 2, 3]`

---

**Dictionary**  
- Unordered, mutable collection of key-value pairs.  
- Defined with curly braces and colons: `{}`  
- Keys must be unique.  
- Example: `my_dict = {'a': 1, 'b': 2}`

---

**Set**  
- Unordered, mutable collection of unique items.  
- Defined with curly braces: `{}`  
- No duplicates allowed.  
- Example: `my_set = {1, 2, 3}`

---

**Tuple**  
- Ordered, immutable collection of items.  
- Defined with parentheses: `()`  
- Allows duplicates.  
- Example: `my_tuple = (1, 2, 3)`

```
NumPy
  •	Handles arrays and numerical operations efficiently.
  •	Example: vector math, matrix manipulation.
Pandas
  •	Manages tabular data using DataFrames.
  •	Ideal for data cleaning and analysis.
Matplotlib
  •	Basic plotting library for charts and graphs.
  •	Great for line plots, bar charts, histograms.
Seaborn
  •	Built on Matplotlib; adds prettier, statistical plots.
  •	Example: heatmaps, violin plots, regression plots.
Scikit-learn
  •	Core ML library for models and evaluation.
  •	Includes regression, classification, clustering, and more.
Deep Learning Libraries
  •	TensorFlow – Google's framework for building and training neural networks.
  •	Keras – User-friendly wrapper around TensorFlow for fast prototyping.
  •	PyTorch – Facebook’s flexible deep learning library with dynamic graphs.

📚 NLTK (Natural Language Toolkit)
  •	A classic library for working with human language data.
  •	Supports tokenization, stemming, tagging, parsing, and corpus access.
  •	Great for educational and research use.
🗣️ spaCy
  •	Industrial-strength NLP library built for speed and efficiency.
  •	Handles tagging, parsing, named entity recognition (NER), and more.
  •	Comes with pre-trained models for many languages.
🧠 TextBlob
  •	Simple NLP tool built on NLTK and Pattern.
  •	Easy interface for sentiment analysis, translation, and more.
💬 Gensim
  •	Specializes in topic modeling and document similarity.
  •	Widely used for word embeddings like Word2Vec.
🔠 Transformers (by Hugging Face)
  •	Deep learning library focused on state-of-the-art models (BERT, GPT, etc.).
  •	Handles text classification, translation, summarization, Q&A, and more.
🧮 Tesseract
  •	Optical Character Recognition (OCR) engine.
  •	Useful for extracting text from images or scanned documents.
📊 Beautiful Soup & Scrapy
  •	Not NLP libraries per se, but great for collecting text data from websites via web scraping.

```

---

- **Linear Algebra**:

**Scalar**  
- A single numerical value (just a number).  
- Example: `5`, `-3.14`

**Vector**  
- An ordered list of numbers representing direction and magnitude.  
- Example: `[2, -1, 3]` — often visualized as an arrow in space.

**Matrix**  
- A rectangular grid of numbers arranged in rows and columns.  
- Example: `[[1, 2], [3, 4]]` — used for linear transformations and storing data.

- **Calculus**: 
- Function: A rule that maps inputs to outputs.
- Derivative: Measures the rate of change of a function.
- Gradient: A vector showing direction and rate of steepest increase.

- **Statistics**:
________________________________________
- Mean – Average value of a dataset.
- Median – Middle value when data is sorted.
- Variance – Measures spread of data from the mean.
- Standard Deviation – Square root of variance; shows how much data varies.
________________________________________
- Basic Probability – Likelihood of an event occurring, between 0 and 1.
________________________________________
**Supervised Learning**

*Regression*  
- **Linear Regression** – Models the relationship between dependent and independent variables using a straight line.  
- **Polynomial Regression** – Extends linear regression by fitting a polynomial equation.  
- **Support Vector Regression (SVR)** – Uses support vectors to find a best-fit hyperplane for regression problems.  
- **Decision Tree Regression** – Splits data into branches to model continuous outputs.  
- **Random Forest Regression** – A collection of decision trees that improve predictive accuracy.  
- **Gradient Boosting Regression** – Sequentially builds models that correct previous errors.

*Classification*  
- **Logistic Regression** – Predicts binary outcomes using a logistic function.  
- **Support Vector Machines (SVM)** – Finds the hyperplane that best separates data into classes.  
- **Decision Trees** – Tree-like models for splitting data based on features.  
- **Random Forest** – Combines multiple decision trees to reduce variance.  
- **Naive Bayes** – Probabilistic model based on Bayes’ theorem with strong independence assumptions.  
- **K-Nearest Neighbors (KNN)** – Classifies data points based on their nearest neighbors.  
- **Neural Networks** – Computational models that mimic human brain function for pattern recognition.  
- **Gradient Boosting** – Builds classifiers sequentially to minimize error.  
- **Linear Discriminant Analysis** – Projects data to maximize class separability.  
- **XGBoost** – An efficient, scalable implementation of gradient boosting.

---

**Unsupervised Learning**

*Clustering*  
- **K-Means Clustering** – Divides data into K clusters by minimizing intra-cluster variance.  
- **Hierarchical Clustering** – Builds nested clusters by merging or splitting groups.  
- **DBSCAN** – Density-based clustering that groups closely packed points.  
- **Gaussian Mixture Models** – Probabilistic model representing data as mixtures of Gaussians.

*Dimensionality Reduction*  
- **Principal Component Analysis (PCA)** – Transforms data to lower dimensions while preserving variance.  
- **Linear Discriminant Analysis (LDA)** – Used for class-based dimension reduction.  
- **t-SNE** – Visualizes high-dimensional data in lower dimensions by preserving local structure.  
- **Independent Component Analysis (ICA)** – Separates a multivariate signal into independent non-Gaussian signals.  
- **UMAP** – Preserves both local and global data structure during reduction.

*Association Rule Learning*  
- **Apriori** – Identifies frequent itemsets in transactional data and derives rules.

---

**Reinforcement Learning**

- **Q-learning** – Learns optimal policies by maximizing expected rewards.  
- **SARSA** – Similar to Q-learning but updates based on the action actually taken.

---

**Ensemble Methods**

Combine multiple models to improve predictions:  
- Random Forest  
- Gradient Boosting  
- Adaboost  
- Bagging  
- Stacking  
________________________________________
📊 Data Preprocessing
•	Feature Scaling – Adjusts numeric features to a common scale.
o	Standardization: Rescales to mean 0, std. dev. 1.
o	Normalization: Scales values to a fixed range (like 0–1).
•	Encoding Categorical Variables
o	One-hot encoding: Turns categories into binary columns.
o	Label encoding: Assigns a unique number to each category.
•	Handling Imbalanced Data
o	SMOTE: Creates synthetic examples of minority class.
________________________________________
🧪 Model Evaluation:

  •	Metrics
  o	MSE: Average of squared errors.
  o	RMSE: Square root of MSE.
  o	Accuracy: Correct predictions / total.
  o	Precision: True positives / predict
  ed positives.
  o	Recall: True positives / actual positives.
  o	F1-score: Balance between precision & recall.
  •	Validation Techniques
  o	Train-test split: Separates data for training and testing.
  o	K-fold cross-validation: Repeated splitting for stable results.
  •	Overfitting vs. Underfitting
  o	Overfitting: Model memorizes training data, poor on new data.
  o	Underfitting: Model too simple to capture data patterns.
________________________________________
📐 Mathematics (Intermediate):
  •	Linear Algebra
  o	Matrix decomposition: Breaks matrix into simpler forms.
  o	Dot product: Combines two vectors to get a scalar.
  •	Calculus
  o	Gradient descent: Optimizes by moving toward lowest error.
  o	Partial derivatives: Rate of change with respect to one variable.
  •	Probability
  o	Conditional probability: Probability given some condition.
  o	Bayes’ theorem: Updates probability based on new evidence.
________________________________________

**Neural Networks & Deep Learning**

*A broad category including specialized architectures and techniques.*

*Activation Functions*  
- **Sigmoid** – Outputs between 0 and 1, good for binary classification.  
- **Tanh** – Outputs between -1 and 1, centered around zero.  
- **Softmax** – Converts values into a probability distribution.  
- **ReLU** – Returns positive values or zero.  
- **Leaky ReLU** – Allows small negative slope for negative inputs.  
- **ELU** – Like ReLU, but smooth for negative inputs.  
- **SELU** – Self-normalizing activation for deep networks.  
- **Swish** – A smooth function: \( f(x) = x \cdot \text{sigmoid}(x) \)  
- **GELU** – Approximates ReLU using Gaussian error function.

*Basic Architectures*  
- **Feedforward Neural Networks (FNN)** – Data flows forward through layers.  
- **Multilayer Perceptron (MLP)** – FNN with multiple hidden layers.  
- **Perceptron** – The simplest model with a single layer and binary output.

*Specialized Architectures*  
- **Convolutional Neural Networks (CNN)** – Ideal for image data using filters.  
- **Recurrent Neural Networks (RNN)** – Designed for sequences and temporal patterns.  
- **Long Short-Term Memory Networks (LSTM)** – RNN variant that captures long-term dependencies.  
- **Radial Basis Function Networks (RBF)** – Uses radial functions for classification.  
- **Generative Adversarial Networks (GANs)** – Two networks compete to generate realistic data.  
- **Autoencoders** – Learn efficient codings by encoding and decoding input data.  
- **Modular Neural Networks** – Composed of independent subnetworks.  
- **Sequence-to-Sequence Models** – Maps input sequences to output sequences, great for translation.

*Other Concepts*  
- **Deep Learning** – Multiple-layer networks for learning hierarchical features.  
- **Graph Neural Networks** – Work with graph-structured data.  
- **Quantum Neural Networks** – Combine quantum computing with neural networks.

---

There we go—everything in one detailed yet readable format. Want this turned into flashcards or visual mind maps? I’m totally game for that too.

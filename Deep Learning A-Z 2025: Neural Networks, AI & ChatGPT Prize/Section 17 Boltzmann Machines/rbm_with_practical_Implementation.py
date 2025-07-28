""""
RBM (Restricted Boltzmann Machines)

    - Applications
        1. Dimentionality Reductoin
        2. Feature Learning
        3. Collaborative Filtering

    - Falls under the category of generative models

    - Consists of two layers
        1. visible units
        2. hidden units

    - The units within each layer are fully connected, but there are no connection between units within the same layer
    - RBMs called 'restricted' because restrictions imposed on the connections between the units
    - The restriction ensures that visible units are only connected to the hidden units and vice versa, making the RBMs a bipartite graph

    - The working of the RBMs devided into two main steps training and inference
        - Training  :
            - involves adjusting the weights and biases to maximize the likelihood of the training data. This is done using the technique called Contrastive Divergence (CD)
            - The CD algorithm compares the activations of the visible and hidden units in the RBM to update weights and biases iteratively

            - The training process starts by initializing weights and biases randomly
            - training sample are presented to RBM
            - activations of the hidden units are computed using current weights and biases
            - Next, activations of the hidden units are used to reconstruct the visible units and the process is repeated for a few steps to obtain the reconstructed visible units
            - Finally, the updates to the weights and biases are computed based on the differences betweenthe original units and reconstructed visible units
       - Inference : process of using trained model to make predictions on decisions on new data, unseen data
            - Once trained can be used for generating new samples or performing classification
            - For 'Generating new samples', RBM starts with random configuration of visible units then iteratively updates the hidden units and reconstructed visible units. This allows the RBM to generate new samples that that are similar to training data
            - For 'Classification task', RBM can be used as feature extractors
            - The hidden units can be used as compressed representation of the input data, capturing the most relevant features
            - These features then can be fed into another classifier such as logistic regression model to perform actual classification task

    - Steps to train RBMs
        1. Data Preprocessing (Normalization, scaling)
        2. Initializing the RBM (initialized by randomly assigning weights and biases to its connections, weights and biases can be sampled from Gaussian distribution)
        3. Computing hidden units activations: Given a training sample, the activations of the hidden units are computed using current weights and biases. This is done by apply sigmoid function to the weighted sum of visible units connected to each hidden units
        4. Sampling the hidden units : Once the hidden units activations are computed, the hidden units are sampled based on their activations. This is done by comparing activations with to the random numbers drawn from the uniform distribution
        5. Computing reconstructed visivle units: The reconstructed visible units are computed using the activations of hidden units and corresponding weights and biases. This is similar to computation of hidden unit activations but in opposite direction
        6. Updating the weights and biases : The weights and biases of the RBMs are updated based on difference between original visible units and reconstrcuted visible units. This update usually done by learning rate that controls the magnitude of the weights and biases
        7. Repeating step 3 and 6: steps 3 and 6 are repeated for multiple iterations or until convergence criteria is met. This allows the RBM learn underlying patterns in the training data and update the weights and biases update accordingly.

    - Implementing the RBM with sklearn involves several steps :
        1. Data preprocessing: prepare dataset by cleaning, normalzing, stadardizing as required
        2. RBM congiguration : set the hyperparameters such as no of visible and hidden units, learing rate, no of training epochs, batch size
        3. Model Initialization: initialize using sklearn BernoulliRBM class
        4. Model training: train the model using preprocessed data, sklern provided fit method for this purpose
        5. Feature extraction: After training you can use the RBM as feature extractor, transform your data using RBM to obtain learned features
        6. Application: Apply these learned features to various ML tasks like classification. regression, clustering
"""
# Example Implementation of RBM

# Importing Libraries
import numpy as np          # for numerical computing
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Preprocessing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into Train and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree')

# KNN without using the RBM
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("KNN without using the RBM\n", classification_report(y_test, y_pred))

# Initialize Bernoulli Model

rbm = BernoulliRBM(n_components=625, learning_rate=0.00001, n_iter=10, verbose=False, random_state=42)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("knn", knn)])

# Training Logistic Pipeline
rbm_features_classifier.fit(X_train, y_train)

y_pred = rbm_features_classifier.predict(X_test)

print("KNN using RMB Features\n", classification_report(y_test, y_pred))

# Transform the data using RBM to get feature representation
X_transformed = rbm.transform(X)

# Visualize the original data and the transformed feature representation
# Now you can use your X transfored as your feature representation

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(X[0].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
axes[0].set_title("Original Image")
axes[1].imshow(X_transformed[0].reshape(25, 25), cmap=plt.cm.gray_r, interpolation='nearest')
axes[1].set_title("Transformed Image")

plt.tight_layout()
plt.show()

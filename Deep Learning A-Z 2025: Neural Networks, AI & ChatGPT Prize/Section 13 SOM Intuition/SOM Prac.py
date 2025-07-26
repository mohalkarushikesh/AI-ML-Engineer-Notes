# SOM - Self Organizing Maps aka Kohonen Maps (Unsupervised Learning) 

### Consists of 2 primary layers
    # Input Layer  : features of the data
    # Output Layer : 2D grid of neurons each neuron represents a cluster in data 
    
### Working of self organizing maps 
    # 1. Initialize the weights of the neurons randomly 
    # 2. For each input, find the neuron with the closest weight vector 
    # 3. Update the weights of the neuron and its neighbors 
    # 4. Repeat for a number of iterations 
    
### Advantages of SOM 
    # 1. Can be used for clustering 
    # 2. Can be used for dimensionality reduction 
    
### Disadvantages of SOM 
    # 1. Can be slow for large datasets 
    # 2. Can be sensitive to initialization 
    # 3. Can be sensitive to the number of neurons in the output layer 
    
### Applications of SOM 
    # 1. Clustering 
    # 2. Dimensionality Reduction 
    # 3. Data Visualization 
    # 4. Anomaly Detection 
    # 5. Image Segmentation 
    # 6. Time Series Analysis 
    # 7. Text Analysis 
    # 8. Recommendation Systems 
    # 9. Anomaly Detection  
    
## Implementation of SOM 

## 1. Importing the libraries 
# we will use the math library to calculate the Euclidean distance between the weights and the weight vector 
import math 

## 2. Defining the SOM class 
# we define the two important functions 
# 1. winner function : to find the winner neuron by calculating the Euclidean distance between the input and the weight vectors of each cluster
# 2. update function : to update the weights vectors of winning neurons acc to weight update rule 

class SOM: 
    def winner(self, weights, sample):
        D0 = 0
        D1 = 0
        
        for i in range(len(sample)):
            D0 += math.pow((sample[i] - weights[0][i]), 2)
            D1 += math.pow((sample[i] - weights[1][i]), 2)
        
        return 0 if D0 < D1 else 1 
    
    def update(self, weights, sample, J, alpha):
        for i in range(len(weights[0])):
            weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])

        return weights 
    
# 3. Define the main function 
# we define the traning data and initialize the weights 
# T : training data with four examples, each having 4 features 
# m, n : number of neurons in the output layer 
# weights : initial weights for two clusters, each with 4 features 
# epochs : number of iterations for training 
# alpha : learning rate for updating the weights 

def main():
    T = [[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]
    m, n = len(T), len(T[0])
    weights = [[0.2, 0.6, 0.5, 0.9], [0.8, 0.4, 0.7, 0.3]]
    ob = SOM()
    epochs = 3
    alpha = 0.5
    
# 4. Train the SOM network
# compute the wining cluster and update the weights 
    for i in range(epochs):
        for j in range(m):
            sample = T[j]
            J = ob.winner(weights, sample)
            weights = ob.update(weights, sample, J, alpha)
            
# 5. Classify Test Sample
# we will use a test sample s and classify it into one of the clusters by computing which cluster has the 
# closet weight vector to the input smaple 
# finally we will print the cluster assignments and trained weights for each cluster 
    s = [0, 0, 0, 1]
    J = ob.winner(weights, s)
    
    print("Test sample s belongs to cluster : ", J)
    print("Trained weights for each cluster : ", weights)
    
# 6. Run the main function 
if __name__ == "__main__":
    main()

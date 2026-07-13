import numpy as np 

class Perceptron:
    def __init__(self, l_r = 0.01, n_iters=1000):
        self.lr = l_r 
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None 
        
    def _step_funct(self, x):
        return np.where(x >= 0, 1, 0)
                   
    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0        
        
        for _ in range(self.n_iters):
            for _idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias 
                y_predicted = self._step_funct(linear_output)
                update = self.lr * (y[_idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update             
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias 
        return self._step_funct(linear_output)

if __name__ == "__main__":
    X = np.array([[0, 0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 1])
    
    clf = Perceptron(l_r=0.1, n_iters=10)
    clf.fit(X, y)

    print("Learned Weights: ", clf.weights)    
    print("Learned Bias: ", clf.bias)
    print("Predictions: ", clf.predict(X))
    
    
        
        
     

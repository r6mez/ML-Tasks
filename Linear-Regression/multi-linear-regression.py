import numpy as np
from sklearn.model_selection import train_test_split
from random import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# this is Stochastic Gradient Descent (SGD) implementation of linear regression

def linear_regression(x, y, lr = 0.0001, epochs = 1000):
    data_count = x.shape[0]
    feature_count = x.shape[1]
    weights = np.array([random() for _ in range(feature_count)])
    bias = 0
    error = []

    for epoch in range(epochs):
        y_new = y.copy()
        
        # calculate y for each data point
        for i in range(data_count):
            y_new[i] = weights @ x[i] + bias
            weights += lr * x[i] * (y[i] - y_new[i])
            bias += lr * (y[i] - y_new[i])
        
        mse = ((y - y_new) ** 2).mean()
        error.append(mse)

    return weights, bias, error

def predict(x, weights, bias):
    feature_count = x.shape[1]
    y_new = bias
    for j in range(feature_count):
        y_new += weights[j] * x[j]
        
    return y_new
    
if __name__ == "__main__":
    x, y = load_diabetes(return_X_y=True)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    weights, bias, error = linear_regression(x_train, y_train)
    
    print ("Weights:", weights)
    print ("Bias:", bias)
    print ("MSE:", error[-1])
    
    plt.plot(error)
    plt.show()
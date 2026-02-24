import pandas as pd
from sklearn.model_selection import train_test_split
from random import random
import matplotlib.pyplot as plt

def linear_regression(x, y, lr = 0.0001, epochs = 10):
    weight = random()
    bias = 0
    error = []

    for epoch in range(epochs):
        data_count = x.shape[0]
        y_new = y.copy()
        
        for i in range(data_count):
            y_new[i] =  weight * x[i] + bias
            weight += lr * x[i] * (y[i] - y_new[i])
            bias += lr * (y[i] - y_new[i])
        
        mse = ((y - y_new) ** 2).mean()
        error.append(mse)

    return weight, bias, error

def predict(x, weight, bias):
    return weight * x + bias
    
if __name__ == "__main__":
    data = pd.read_csv("Data/income.csv")
    x = data["income"].values.reshape(-1, 1)
    y = data["happiness"].values.reshape(-1, 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    weight, bias, error = linear_regression(x_train, y_train)
    
    print ("Weight:", weight)
    print ("Bias:", bias)
    print ("MSE:", error[-1])
    
    # train figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x_train, y_train)
    ax1.plot(x_train, predict(x_train, weight, bias), color='red', linewidth=2)
    ax2.plot(error)
    plt.show()

    # test figure
    plt.scatter(x_test, y_test)
    plt.plot(x_test, predict(x_test, weight, bias), color='red', linewidth=2)
    plt.show()
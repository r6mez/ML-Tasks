from sklearn.preprocessing import StandardScaler
import numpy as np

arr = [20, 30, 50, 70, 40, 15, 52, 63, 100, 105, 23, 64]

# mine
mean = sum(arr) / len(arr)

# variance =  sum(x - mean) / n
# standard_deviation = sqrt(variance)

var = sum((x - mean) ** 2 for x in arr) / len(arr)
std = var ** 0.5

# standarded result = (x - mean) / standard_deviation
result1 = [(x - mean) / std for x in arr]

print(result1)

# scikit
data = np.array(arr)
data = data.reshape(-1, 1)

p = StandardScaler()
result2 = p.fit_transform(data)
print(result2)
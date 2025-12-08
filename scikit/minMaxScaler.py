from sklearn.preprocessing import MinMaxScaler
import numpy as np

arr = [20, 30, 50, 70, 40, 15, 52, 63, 100, 105, 23, 64]

# mine
mn = min(arr)
mx = max(arr)

result1 = []
for x in arr:
    y = (x - mn)/(mx - mn)
    result1.append(y)

print(result1)  

# scikit
data = np.array(arr)
data = data.reshape(-1, 1) # rows, cols (-1 means automatic calc)
print(data)

processor = MinMaxScaler()
result2 = processor.fit_transform(data)
print(result2)
import numpy as np

data = np.arange(20)
print("Data array:")
print(data)

matrix = np.random.rand(5,5)
print("\nMatrix:")
print(matrix)

print("\nThird element of data array:", data[2])

data[0] = 100
print("\nData array after updating the first element:")
print(data)

data = np.delete(data, [0,1,2])
print("\nData array after deleting the first three elements:")
print(data)

# Adding elements to the array (Create operation)
data = np.append(data, [30,40,50])
print("\nData array after adding elements:")
print(data)

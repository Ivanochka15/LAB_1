import numpy as np
import matplotlib.pyplot as plt

a = 2.0
b = 1.0
sigma = 0.5
x = np.linspace(0, 10, num=10)
y = a * x + b + np.random.normal(loc=0.0, scale=sigma, size=len(x))
print('x:', x)
print('y:', y)
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
y_hat = a * x + b
mae = np.mean(np.abs(y_hat - y))
mse = np.mean((y_hat - y) ** 2)


import csv
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X', 'Y', 'Y_hat', 'mAE', 'mSE'])
    for i in range(len(x)):
        writer.writerow([x[i], y[i], y_hat[i], mae, mse])

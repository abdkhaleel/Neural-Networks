import numpy as np

input = [[5, 2, 7, 5],
         [2, 7, 2, 6],
         [2, 653, 4, 7]]

weights = [[0.5, -0.2, -0.4, -0.6], 
           [-0.4, 0.1, 0.44, 0.24],
           [-0.3, -0.4, -0.01, -0.28]]

biases = [3, 5, 4.7]

output = np.dot(input, np.array(weights).T) + biases

print(output)

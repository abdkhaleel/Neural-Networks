import numpy as np

input = [[5, 2, 7, 5],
         [2, 7, 2, 6],
         [2, 653, 4, 7]]

weights = [[0.5, -0.2, -0.4, -0.6], 
           [-0.4, 0.1, 0.44, 0.24],
           [-0.3, -0.4, -0.01, -0.28]]

biases = [3, 5, 4.7]

weights2 = [[0.4, 0.6, 0.11],
            [-0.54, -0.22, 0.52],
            [0.44, 0.58, -0.31]]

biases2 = [4, 6, 9]

layer1_output = np.dot(input, np.array(weights).T) + biases

layer2_output = np.dot(layer1_output, np.array(layer1_output).T) + biases2

print(layer2_output)



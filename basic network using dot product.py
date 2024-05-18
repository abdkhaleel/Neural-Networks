
import numpy as np

input = [5, 2, 7]

weights = [[0.5, -0.2, -0.4], 
           [-0.4, 0.1, 0.44],
           [-0.3, -0.4, -0.01]]

biases = [3, 5, 4.7]

output = np.dot(weights, input) + biases

print(output)
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]

class Layer_Dens:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
layer1 = Layer_Dens(4, 5)

layer2 = Layer_Dens(5, 2)

layer1.forward(X)

print(layer1.output)

layer2.forward(layer1.output)

print(layer2.output)
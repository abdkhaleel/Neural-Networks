import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

class Layer_Dens:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        
layer1 = Layer_Dens(2, 5)
layer1.forward(X)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)

layer2 = Layer_Dens(5, 3)
activation2 = Activation_Softmax()

layer2.forward(activation1.outputs)
activation2.forward(layer2.output)

print(activation2.outputs[:5])
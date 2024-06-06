import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confedence = y_pred[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confedence = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confedence)
        return negative_log_likelihoods
    
    
X, y = spiral_data(100, 3)
     
layer1 = Layer_Dens(2, 5)
layer1.forward(X)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)

layer2 = Layer_Dens(5, 3)
activation2 = Activation_Softmax()

layer2.forward(activation1.outputs)
activation2.forward(layer2.output)

print(activation2.outputs[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.outputs, y)

print("Loss:",loss)

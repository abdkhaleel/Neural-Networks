input = [5, 2, 7]

weights = [[0.5, -0.2, -0.4], 
           [-0.4, 0.1, 0.44],
           [-0.3, -0.4, -0.01]]

biases = [3, 5, 4.7]

output = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for weight, inp in zip(neuron_weights, input):
        neuron_output += weight*inp
    neuron_output += neuron_bias
    output.append(neuron_output)
    

print(output)

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

x, y = spiral_data(samples=100, classes=3)
dense = Layer_Dense(2, 3)
activation = Activation_ReLU()
dense.forward(x)
activation.forward(dense.output)
print(activation.output[:5])

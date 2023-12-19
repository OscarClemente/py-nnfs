inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]
bias = [2.0, 3.0, 0.5]

layerOutputs = []
for neuronWeights, neuronBias in zip(weights, bias):
    neuronOutput = 0
    for nInput, weight in zip(inputs, neuronWeights):
        neuronOutput += nInput * weight
    neuronOutput += neuronBias
    layerOutputs.append(neuronOutput)

print(layerOutputs)

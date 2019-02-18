import numpy as np
import math

def main(relFilePath, hasColumnLabel, hasID, hiddenLayers, hiddenLayerNodes):
  trainSet, testSet, validationSet = trainTestValidationSplit(relFilePath, hasColumnLabel, hasID)
  # print('testSet: \n', testSet)
  # print('trainSet: \n', trainSet)
  # print('validationSet: \n', validationSet)
  network = initNetwork(trainSet.shape[1]-1, hiddenLayers, hiddenLayerNodes, getClassCount(trainSet))
  # Cutting off class labels
  dataToFeed = trainSet[-1:,:-1]
  outputs = np.apply_along_axis(feedforward, 1, dataToFeed, network)
  print('outputs ', outputs)

  '''
  for layer in weights:
    print(np.shape(layer))
    print(layer)
  '''

def trainTestValidationSplit(relFilePath, hasColumnLabel, hasID):
  data = np.genfromtxt(relFilePath, delimiter=',')
  # Removing the column labels
  if (hasColumnLabel):
    data = data[1:, :]
  # Removing the row that contains ID attribute
  if (hasID):
    # TODO want to check which column has id before removing first
    data = data[:, 1:]
  # Randomizing the data
  np.random.shuffle(data)
  split_a = int(0.75 * data.shape[0])
  split_b = int(0.85 * data.shape[0])
  trainSet = data[:split_a]
  validationSet = data[split_a:split_b]
  testSet = data[:split_b]
  # 75% training set, 10% validation set, 15% test set
  return trainSet, testSet, validationSet

def getClassCount(data):
  # This data is assumed to contain class labels
  classColumn = data[:,-1:]
  classesVisited = []
  classCount = 0
  for item in np.nditer(classColumn):
    if item not in classesVisited:
      classesVisited.append(item)
      classCount += 1
  return classCount

def initNetwork(attributeCount, hiddenLayers, hiddenLayerNodes, outputNodes):
  print('attributeCount ', attributeCount)
  weights = list()
  biases = list()
  # Connecting input layer to hidden layer
  weights.append(np.random.rand(hiddenLayerNodes, attributeCount))
  biases.append(np.random.rand(hiddenLayerNodes, ))
  # Connecting hidden layers to other hidden layers
  for layerIndex in range(0, hiddenLayers):
    index = layerIndex + 1
    if (index != hiddenLayers):
      weights.append(np.random.rand(hiddenLayerNodes, hiddenLayerNodes))
      biases.append(np.random.rand(hiddenLayerNodes, ))
  # Connecting output layer
  weights.append(np.random.rand(outputNodes, hiddenLayerNodes))
  biases.append(np.random.rand(outputNodes, ))
  return {'weights': weights, 'biases': biases}

def sigmoid(input):
  return 1.0 / (1.0 + math.exp(-input))

def sigmoidPrime(input):
  return input * (1 - input)

def feedforward(data, network):
  inputs = data
  for index, layer in enumerate(network['weights']):
    outputs = []
    for neuronWeights in layer:
      inputProduct = np.dot(neuronWeights, inputs)
      outputs.append(inputProduct)
    inputs = sigmoid(np.add(outputs, network['biases'][index]))
  return inputs

'''
Parameters are:
  (String relativeFilePath,
  Boolean columnLablesPresent,
  Boolean IDColumnPresent,
  Int numberOfHiddenLayers,
  Int numberOfNodesPerHiddenLayer)
'''

sigmoid = np.vectorize(sigmoid)
main('GlassData.csv', True, True, 1, 9)

'''
Next steps:
1) Write activation function (sum of weights*inputs as input to sigmoid)
2) Write feedforward function
3) 
'''
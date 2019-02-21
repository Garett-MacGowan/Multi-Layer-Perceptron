'''
Author: Garett MacGowan
Student Number: 10197107
CISC 452 Neural and Genetic Computing
'''

import numpy as np
import math

def main(relFilePath, hasColumnLabel, hasID, hiddenLayers, hiddenLayerNodes):
  data = readData(relFilePath, hasColumnLabel, hasID)
  data = cleanClassLabels(data)
  trainSet, testSet, validationSet = trainTestValidationSplit(data)
  # print('testSet: \n', testSet)
  # print('trainSet: \n', trainSet)
  # print('validationSet: \n', validationSet)
  network = initNetwork(trainSet.shape[1] - 1, hiddenLayers, hiddenLayerNodes, getClassCount(trainSet))
  network = train(network, trainSet, 1, 1, 2)
  evaluator(network, testSet)
  #print('outputs ', outputs)

def initNetwork(attributeCount, hiddenLayers, hiddenLayerNodes, outputNodes):
  weights = list()
  biases = list()
  # Connecting input layer to hidden layer
  #print('attribute count ', attributeCount)
  weights.append(np.random.uniform(low=0.1, high=0.9, size=(hiddenLayerNodes, attributeCount)))
  biases.append(np.random.uniform(low=0.1, high=0.9, size=(hiddenLayerNodes, )))
  # Connecting hidden layers to other hidden layers
  for layerIndex in range(0, hiddenLayers):
    index = layerIndex + 1
    if (index != hiddenLayers):
      weights.append(np.random.uniform(low=0.1, high=0.9, size=(hiddenLayerNodes, hiddenLayerNodes)))
      biases.append(np.random.uniform(low=0.1, high=0.9, size=(hiddenLayerNodes, )))
  # Connecting output layer
  weights.append(np.random.uniform(low=0.1, high=0.9, size=(outputNodes, hiddenLayerNodes)))
  biases.append(np.random.uniform(low=0.1, high=0.9, size=(outputNodes, )))
  return {'weights': weights, 'biases': biases}

# Defines out
def sigmoid(input):
  return 1.0 / (1.0 + math.exp(-input))

# Defines ∂out/∂net
def sigmoidPrime(input):
  return input * (1 - input)

def feedforward(data, network):
  #print('feeding forward')
  inputs = data
  activations = []
  activations.append(inputs)
  for index, layer in enumerate(network['weights']):
    #print('layer ', layer)
    outputs = []
    for neuronWeights in layer:
      inputProduct = np.dot(neuronWeights, inputs)
      outputs.append(inputProduct)
    inputs = sigmoid(np.add(outputs, network['biases'][index]))
    activations.append(inputs)
  return inputs, activations

# backpropagates an entire training set
def backpropagation(network, activations, classLabel):
  deltaWeights = list()
  backpropagatingError = None
  for index in reversed(range(len(network['weights']))):
    #print('index', index)
    # Backpropagation for all other layers
    if (index != len(network['weights'])-1):
      doutBYdnet = np.array(list(map(sigmoidPrime, activations[index])))
      dnetBYweight = np.array(activations[index-1])
      dnetBYweight = np.reshape(dnetBYweight, (dnetBYweight.shape[0], 1))
      dnetBYweight = np.full((dnetBYweight.shape[0], backpropagatingError.shape[0]), dnetBYweight)
      intermediate = np.multiply(backpropagatingError, network['weights'][index])
      backpropagatingError = np.multiply(intermediate, doutBYdnet)
      
      dJBYdW = np.dot(dnetBYweight, backpropagatingError)
      deltaWeights.append(dJBYdW)
    # Backpropagation for output layer
    else:
      dEtotalBYdout = np.array(errorMatrix(activations[index], classLabel))
      doutBYdnet = np.array(list(map(sigmoidPrime, activations[index])))
      dnetBYweight = np.array(activations[index-1])
      # print('dEtotalBYdout ', dEtotalBYdout)
      # print('doutBYdnet ', doutBYdnet)
      # print('dnetBYweight ', dnetBYweight)
      # print('dEtotalBYdout shape ', dEtotalBYdout.shape)
      # print('doutBYdnet shape', doutBYdnet.shape)
      # print('dnetBYweight shape', dnetBYweight.shape)
      backpropagatingError = np.multiply(dEtotalBYdout, doutBYdnet)
      backpropagatingError = np.reshape(backpropagatingError, (backpropagatingError.shape[0], 1))
      deltaWeights.append(np.multiply(backpropagatingError, dnetBYweight))
  return deltaWeights

def updateWeights(network, deltaWeights, learningRate, momentum):
  newWeights = list()
  for index in range(len(network['weights'])):
    deltaWeightsFinal = np.multiply(learningRate, deltaWeights[index])
    print('weights ', network['weights'][index])
    print('deltaWeightsFinal ', deltaWeightsFinal)
    newWeights.append(np.subtract(network['weights'][index], deltaWeightsFinal))
    print('newWeights ', newWeights)
    quit()
  network['weights'] = newWeights
  return network

def train(network, data, learningRate, momentum, epochs):
  # TODO change terminating conditions
  for epoch in range(epochs):
    # Remove class labels before feeding forward
    dataToFeed = data[:, :-1]
    classLabels = data[:, -1:]
    #print('filledNetwork ', np.full((dataToFeed.shape[0], 1), network))
    #outputs, activations = np.apply_along_axis(feedforward, 1, dataToFeed, network)
    for index, row in enumerate(dataToFeed):
      outputs, activations = feedforward(row, network)
      #print('outputs ', outputs)
      #print('activations ', activations)
      deltaWeights = backpropagation(network, activations, classLabels[index])
      network = updateWeights(network, deltaWeights, learningRate, momentum)
    #print(str(epoch/epochs*100) + '% complete')
    evaluator(network, data)
  print('training complete')
  return network

# Assumes consecutive integer clas labels beginning at index 0
def decodePrediction(probabilityDistribution):
  '''
  call this method with
  outputs = np.apply_along_axis(decodePrediction, 1, outputs)
  outputs = np.reshape(outputs, (outputs.shape[0], 1))
  '''
  highestProbability = 0
  prediction = 0
  index = 0
  for probability in np.nditer(probabilityDistribution):
    if (probability > highestProbability):
      prediction = index
      highestProbability = probability
    index += 1
  return prediction

# Defines ∂Etotal/∂out 
def errorMatrix(results, classLabel):
  errorMatrix = []
  for index, result in enumerate(results):
    # Assumes consecutive integer class labels beginning at index 0
    if (index == classLabel):
      errorMatrix.append(result-1)
    else:
      errorMatrix.append(result-0)
  return errorMatrix

'''
Helper Functions
'''
def readData(relFilePath, hasColumnLabel, hasID):
  data = np.genfromtxt(relFilePath, delimiter=',')
  # Removing the column labels
  if (hasColumnLabel):
    data = data[1:, :]
  # Removing the row that contains ID attribute
  if (hasID):
    # TODO want to check which column has id before removing first
    data = data[:, 1:]
  return data

'''
def outputResults(initialWeights, finalWeights, totalIterations, precisionAndRecallArray, confusionMatrixArray):
  text_file = open('output.txt', 'w')
  text_file.write('Initial weights: \n')
  for iw in list(initialWeights):
    text_file.write(str(iw) + '\n')
  text_file.write('\n')
  text_file.write('Final weights: \n')
  for fw in list(finalWeights):
    text_file.write(str(fw) + '\n')
  text_file.write('\n')
  text_file.write('Total iterations: \n' + str(totalIterations) + '\n')
  for i in range(0,3):
    text_file.write('Class ' + str(i+1) + ', Precision: ' + str(precisionAndRecallArray[i][0]) + ' Recall: ' + str(precisionAndRecallArray[i][1]) + '\n')
  text_file.write('\n')
  text_file.write('Confusion matrix: \n')
  for j in range(len(confusionMatrixArray)):
    text_file.write('Class ' + str(j) + '\n')
    for k in range(len(confusionMatrixArray[j])):
      text_file.write(str(confusionMatrixArray[j][k]) + '\n')
    text_file.write('\n')
  text_file.close()
'''

def evaluator(network, data):
  dataToFeed = data[:, :-1]
  classLabels = data[:, -1:]
  print('evaluator')
  correctlyClassifiedCount = 0
  for index, row in enumerate(dataToFeed):
    outputs, activations = feedforward(row, network)
    outputs = decodePrediction(outputs)
    if (outputs == int(classLabels[index])):
      correctlyClassifiedCount += 1
  print(correctlyClassifiedCount / classLabels.shape[0])


def trainTestValidationSplit(data):
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
  # NOTE This data is assumed to contain class labels
  classColumn = data[:,-1:]
  classesVisited = []
  classCount = 0
  for item in np.nditer(classColumn):
    if item not in classesVisited:
      classesVisited.append(item)
      classCount += 1
  return classCount

# This function converts class labels to consecutive integers
def cleanClassLabels(data):
  classLabels = []
  # NOTE Assumes last column is the class column
  classColumn = data[:, -1]
  for row in np.nditer(classColumn):
    if str(row) not in classLabels:
      classLabels.append(str(row))
  # Creating a mapping between class columns and consecutive integers
  mapping = dict()
  for index, label in enumerate(classLabels):
    mapping[label] = index
  # Applying the mapping to the class column
  newClassColumn = np.apply_along_axis(lambda x: mapping[str(x[-1])], 1, data)
  newClassColumn = newClassColumn.reshape(newClassColumn.shape[0],1)
  # Putting together the cleaned data
  data = np.append(data[:, :-1], newClassColumn, axis=1)
  return data

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
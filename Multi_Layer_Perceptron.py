'''
Author: Garett MacGowan
Student Number: 10197107
CISC 452 Neural and Genetic Computing
Description: This file implements a multi layer perceptron for 6 class classification
'''

import numpy as np
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def main(relFilePath, hasColumnLabel, hasID, hiddenLayers, hiddenLayerNodes, learningRate, momentum, epochs):
  data = readData(relFilePath, hasColumnLabel, hasID)
  data = cleanClassLabels(data)
  data = normalize(data)
  classCount = getClassCount(data)
  trainSet, testSet, validationSet = trainTestValidationSplit(data, classCount)
  network = initNetwork(trainSet.shape[1] - 1, hiddenLayers, hiddenLayerNodes, classCount)
  initialWeights = network['weights']
  network = train(network, trainSet, validationSet, learningRate, momentum, epochs)
  print(str(evaluator(network, testSet) * 100) + '% accuracy on the testing set')
  precisionAndRecallArray, confusionMatrixArray = test(network, testSet)
  outputDesignAndPerformance(initialWeights, network['weights'], learningRate, momentum, hiddenLayers, hiddenLayerNodes, classCount, precisionAndRecallArray, confusionMatrixArray)

def initNetwork(attributeCount, hiddenLayers, hiddenLayerNodes, outputNodes):
  weights = list()
  biases = list()
  # Connecting input layer to hidden layer
  #print('attribute count ', attributeCount)
  weights.append(np.random.uniform(low=-1.0, high=1.0, size=(hiddenLayerNodes, attributeCount)))
  biases.append(np.ones(hiddenLayerNodes, ))
  # Connecting hidden layers to other hidden layers
  for layerIndex in range(0, hiddenLayers):
    index = layerIndex + 1
    if (index != hiddenLayers):
      weights.append(np.random.uniform(low=-1.0, high=1.0, size=(hiddenLayerNodes, hiddenLayerNodes)))
      biases.append(np.ones(hiddenLayerNodes, ))
  # Connecting output layer
  weights.append(np.random.uniform(low=-1.0, high=1.0, size=(outputNodes, hiddenLayerNodes)))
  biases.append(np.ones(outputNodes, ))
  return {'weights': weights, 'biases': biases}

# Defines out
def sigmoid(input):
  # Two parts b/c it prevents overflow
  if (input < 0):
    return (1 - (1 / (1 + math.exp(input))))
  return 1.0 / (1.0 + math.exp(-input))

# Defines ∂out/∂net
def sigmoidPrime(input):
  return input * (1 - input)

def feedforward(data, network):
  inputs = data
  activations = []
  activations.append(inputs)
  for index, layer in enumerate(network['weights']):
    outputs = []
    for neuronWeights in layer:
      inputProduct = np.dot(neuronWeights, inputs)
      outputs.append(inputProduct)
    # Biases are not traied in this version (they are all 1)
    inputs = sigmoid(np.add(outputs, network['biases'][index]))
    activations.append(inputs)
  return inputs, activations

# Backpropagation for a single example
def backpropagation(network, activations, classLabel):
  deltaWeights = list()
  backpropagatingError = None
  for index in reversed(range(len(network['weights']))):
    # Backpropagation for all other layers
    if (index != len(network['weights'])-1):
      backpropagatingError = np.reshape(backpropagatingError, (backpropagatingError.shape[0], 1))
      intermediate = np.multiply(backpropagatingError, network['weights'][index+1])
      
      doutBYdnet = np.array(list(map(sigmoidPrime, activations[index+1])))
      doutBYdnet = np.full((intermediate.shape[0], intermediate.shape[1]), doutBYdnet)
      backpropagatingError = np.multiply(intermediate, doutBYdnet)

      dnetBYweight = np.array(activations[index])
      dnetBYweight = np.full((backpropagatingError.shape[0], dnetBYweight.shape[0]), dnetBYweight)
      dnetBYweight = np.transpose(dnetBYweight)

      ErrBYWeight = np.dot(dnetBYweight, backpropagatingError)
      deltaWeights.insert(0, ErrBYWeight)
    # Backpropagation for output layer
    else:
      dEtotalBYdout = np.array(errorMatrix(activations[index+1], classLabel))
      doutBYdnet = np.array(list(map(sigmoidPrime, activations[index+1])))
      backpropagatingError = np.multiply(dEtotalBYdout, doutBYdnet)
      backpropagatingError = np.reshape(backpropagatingError, (backpropagatingError.shape[0], 1))

      dnetBYweight = np.array(activations[index])
      ErrBYWeight = np.multiply(backpropagatingError, dnetBYweight)
      deltaWeights.insert(0, ErrBYWeight)
  return deltaWeights

def updateWeights(network, deltaWeights, previousIterDeltaWeights, learningRate, momentum):
  newWeights = list()
  for index in range(len(network['weights'])):
    deltaWeightsFinal = np.multiply(learningRate, deltaWeights[index])
    # Applying momentum
    if (previousIterDeltaWeights != None):
      momentumDelta = np.multiply(momentum, previousIterDeltaWeights[index])
      deltaWeightsFinal = np.add(deltaWeightsFinal, momentumDelta)
    # Applying negative gradient to weights
    newWeights.append(np.subtract(network['weights'][index], deltaWeightsFinal))
  network['weights'] = newWeights
  return network

def train(network, trainingData, validationData, learningRate, momentum, epochs):
  print('Training')
  previousIterDeltaWeights = None
  # Validation accuracy for early stopping
  validationAccuracy = 0
  learningModifier = 1
  for epoch in range(epochs):
    # Removing class labels before feeding forward
    dataToFeed = trainingData[:, :-1]
    classLabels = trainingData[:, -1:]
    deltaWeights = []
    for index, row in enumerate(dataToFeed):
      outputs, activations = feedforward(row, network)
      deltaWeights.append(backpropagation(network, activations, classLabels[index]))
    # Taking the mean of the delta weights to find the gradient
    deltaWeights = np.mean(deltaWeights, axis=0)
    if ((epoch + 1) % 1000 == 0):
      learningModifier += 1
      print(str(round(epoch/epochs*100, 3)) + '% complete; ' + 'training set accuracy ' + str(evaluator(network, trainingData) * 100) + '%' )
    network = updateWeights(network, deltaWeights, previousIterDeltaWeights, learningRate*(1/(learningModifier)), momentum*(1/(learningModifier)))
    previousIterDeltaWeights = deltaWeights
    # Early stopping condition: validation accuracy diverging (decreasing)
    currentValidationAccuracy = evaluator(network, validationData)
    if (currentValidationAccuracy < validationAccuracy):
      break
  print('training complete')
  return network

'''
Converts a probability distribution to a class label.
Assumes consecutive integer class labels beginning at index 0
'''
def decodePrediction(probabilityDistribution):
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

# Reads the data from the relative file path and stores it as a numpy array
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

# Returns the prediction accuracy for a given set
def evaluator(network, data):
  dataToFeed = data[:, :-1]
  classLabels = data[:, -1:]
  correctlyClassifiedCount = 0.0
  for index, row in enumerate(dataToFeed):
    outputs, activations = feedforward(row, network)
    outputs = decodePrediction(outputs)
    if (outputs == int(classLabels[index])):
      correctlyClassifiedCount += 1
  return (correctlyClassifiedCount / classLabels.shape[0])

# 75% training set, 10% validation set, 15% test set with equal class distribution
def trainTestValidationSplit(data, classCount):
  # Randomizing the data
  np.random.shuffle(data)
  # Separating data into separate arrays based on class
  dataByClass = list()
  for index in range(classCount):
    dataByClass.append(data[data[:, -1] == index, :])
  # Creating empty arrays to append to
  trainSet = np.array([]).reshape(0,data.shape[1])
  validationSet = np.array([]).reshape(0,data.shape[1])
  testSet = np.array([]).reshape(0,data.shape[1])
  # Appending correct proportion of data for each class into final sets
  for arr in dataByClass:
    split_a = int(0.75 * arr.shape[0])
    split_b = int(0.85 * arr.shape[0])
    trainSet = np.append(trainSet, arr[:split_a], axis=0)
    validationSet = np.append(validationSet, arr[split_a:split_b], axis=0)
    testSet = np.append(testSet, arr[split_b:], axis=0)
  return trainSet, testSet, validationSet

# Returns the amount of classes present for a given dataset
def getClassCount(data):
  # NOTE This data is assumed to contain class labels in the last column
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

# Normalizing all attributes to a range between 0 and 1 via max normalization
def normalize(data):
  # Don't want to normalize the class column
  dataToNormalize = data[:, :-1]
  classLabels = data[:, -1:]
  normalizedData = dataToNormalize / dataToNormalize.max(axis=0)
  normalizedData = np.concatenate((normalizedData, classLabels), axis=1)
  return normalizedData

def test(network, data):
  dataToFeed = data[:, :-1]
  classLabels = data[:, -1:]
  # Getting predictions for all the data in the test set
  predicted = []
  for row in dataToFeed:
    outputs, activations = feedforward(row, network)
    outputs = decodePrediction(outputs)
    predicted.append(outputs)

  confusionMatrixArray = confusion_matrix(classLabels.flatten(), predicted)
  precisionAndRecallArray = precision_recall_fscore_support(classLabels.flatten(), predicted, average=None)
  precisionAndRecallArray = np.array(precisionAndRecallArray[:2]).tolist()
  outputResults(classLabels, predicted)
  return precisionAndRecallArray, confusionMatrixArray

def outputResults(classLabels, predicted):
  predicted = np.array(predicted).reshape(len(predicted), 1)
  output = np.append(classLabels, predicted, axis=1)
  text_file = open('output.txt', 'w')
  text_file.write('Author: Garett MacGowan \n')
  text_file.write('Student Number: 10197107 \n')
  text_file.write('\n')
  text_file.write('This file contains class lables and predictions for the testing set.')
  text_file.write('\n')
  text_file.write('Class Labels, Predicted \n')
  text_file.write(str(output))

def outputDesignAndPerformance(initialWeights, finalWeights, learningRate, momentum, hiddenLayers, nodesPerHiddenLayer, classCount, precisionAndRecallArray, confusionMatrixArray):
  text_file = open('DesignAndPerformance.txt', 'w')
  text_file.write('Author: Garett MacGowan \n')
  text_file.write('Student Number: 10197107 \n')
  text_file.write('Learning rate: ' + str(learningRate) + '\n')
  text_file.write('Learning rate is chosen as such through trial and error. \n')
  text_file.write('The learning rate decreases over time proportional to the epoch. \n')
  text_file.write('\n')
  text_file.write('Momentum: ' + str(momentum) + '\n')
  text_file.write('Momentum is chosen as such through trial and error. \n')
  text_file.write('The momentum decreases over time proportional to the epoch. \n')
  text_file.write('\n')
  text_file.write('I have used a sigmoid activation function so that I can create a probability distribution for the class predictions. \n')
  text_file.write('Sigmoid is also used so that the network is differentiable everywhere. \n')
  text_file.write('\n')
  text_file.write('Number of hidden layers: ' + str(hiddenLayers) + '\n')
  text_file.write('There should not be too many hidden layers. This helps to avoid overfitting. \n')
  text_file.write('\n')
  text_file.write('Number of nodes in each hidden layer: ' + str(nodesPerHiddenLayer) + '\n')
  text_file.write('There should be the same number of nodes in the hidden layer as in input layer to allow for complex relations between all attributes. \n')
  text_file.write('\n')
  text_file.write('Number of nodes in output layer: ' + str(classCount) + '\n')
  text_file.write('There should be the same number of nodes in the output layer as there are classes to prevent crosstalk. \n')
  text_file.write('\n')
  text_file.write('Regularization: \n')
  text_file.write('I have used early stopping based on when my validation accuracy decreases as my main regularization approach. \n')
  text_file.write('In addition, I am also only using one hidden layer, which helps to prevent data memorization. \n')
  text_file.write('\n')
  text_file.write('Preprocessing: \n')
  text_file.write('I have removed the ID attribute from the data because it could give an unfair advantage to the algorithm. \n')
  text_file.write('I have also normalized the attributes (between 0 and 1) so that no one attribute is given more importance over another. \n')
  text_file.write('\n')
  text_file.write('Training, validation, and testing split: \n')
  text_file.write('The split follows 75% training, 10% validation, and 15% testing. \n')
  text_file.write('Each set is statistically equivalent. That is, they have the same proportion of each class relative to their size. \n')
  text_file.write('\n')
  text_file.write('Confusion Matrix: \n')
  text_file.write(str(confusionMatrixArray) + '\n')
  text_file.write('\n')
  text_file.write('Precision for each class (ascending): \n')
  text_file.write(str(precisionAndRecallArray[0]) + '\n')
  text_file.write('\n')
  text_file.write('Recall for each class (ascending): \n')
  text_file.write(str(precisionAndRecallArray[1]) + '\n')
  text_file.write('\n')
  text_file.write('Initial weights: \n')
  for iw in list(initialWeights):
    text_file.write(str(iw) + '\n')
  text_file.write('\n')
  text_file.write('Final weights: \n')
  for fw in list(finalWeights):
    text_file.write(str(fw) + '\n')
  text_file.write('\n')
  text_file.close()

'''
Parameters are:
  (String relativeFilePath,
  Boolean columnLablesPresent,
  Boolean IDColumnPresent,
  Int numberOfHiddenLayers,
  Int numberOfNodesPerHiddenLayer,
  Int learningRate,
  Int momentum,
  Int epochs)
'''
sigmoid = np.vectorize(sigmoid)
# Not to be changed. Network depth and layer size not currently dynamic
main('GlassData.csv', True, True, 1, 9, 0.4, 0.1, 100000)
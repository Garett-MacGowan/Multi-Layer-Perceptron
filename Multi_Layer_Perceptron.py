import numpy as np

def main(relFilePath, hasColumnLabel, hasID, hiddenLayers, hiddenLayerNodes):
  trainSet, testSet, validationSet = trainTestValidationSplit(relFilePath, hasColumnLabel, hasID)
  # print('testSet: \n', testSet)
  # print('trainSet: \n', trainSet)
  # print('validationSet: \n', validationSet)
  weights = initNetwork(trainSet.shape[1]-1, hiddenLayers, hiddenLayerNodes, getClassCount(trainSet))
  for layer in weights:
    print(np.shape(layer))
    print(layer)

def trainTestValidationSplit(relFilePath, hasColumnLabel, hasID):
  data = np.genfromtxt(relFilePath, delimiter=',')
  # Removing the column labels
  if (hasColumnLabel):
    data = data[1:, :]
  # Removing the row that contains ID attribute
  if (hasID):
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
  network = list()
  # Connecting input layer to hidden layer
  # Input layer weights; attributeCount + 1 column (b/c of bias)
  hiddenLayer = np.random.rand(attributeCount + 1, hiddenLayerNodes)
  network.append(hiddenLayer)
  # Connecting hidden layers to other hidden layers or to output layer
  for layerIndex in range(1, hiddenLayers):
    index = layerIndex + 1
    if (index == hiddenLayers):
      hiddenLayer = np.random.rand(hiddenLayerNodes + 1, outputNodes)
    else:
      hiddenLayer = np.random.rand(hiddenLayerNodes + 1, hiddenLayerNodes)
    network.append(hiddenLayer)
  return network

# Parameters are (String relativeFilePath, Boolean columnLablesPresent, Boolean IDColumnPresent)
main('GlassData.csv', True, True, 2, 9)
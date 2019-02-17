import numpy as np

def main():
  data = np.genfromtxt('trainSeeds.csv', delimiter=',')
  data = np.insert(data, 0, 1, axis=1)
  # Setting random weights (eight because first one is theta threshold)
  # Three sets because we need three output neurons


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


def activate(totalActivation, threshold):
  if (totalActivation >= threshold):
    return 1
  else:
    return 0

def predict(weight, inputData):
  threshold = weight[0]
  totalActivation = np.sum(np.dot(weight, inputData))
  return activate(totalActivation, threshold)

def train(weights, data):
  # Slice (take all of the first dimension (rows)
  # and all but the last item in the second dimension (columns))
  inputData = data[0:, 0:-1]
  classLabels = data[0:, -1:]
  totalIterations = 0
  counter = 0
  currentAccuracy = 0
  currentBestAccuracy = 0
  finalWeights = None
  # While accuracy has not improved after 100000 iterations
  while(counter < 100000):
    '''
    Using map which takes in arrays of the same shape (hense np.full) and applies the weightTrainer function to each row
    This trains each neuron to try and be predictive of its class (1, 2, or 3)
    A new weight is generated for each data point and all the weights are averaged together to get the new weight.
    This is instead of updating the weight after each data point and may help to reduce output toggling
    '''
    weights0 = np.mean(np.array(list(map(weightTrainer, np.full((data.shape[0], 8), weights[0]), inputData, classLabels, np.full((data.shape[0], 1), 1)))), 0)
    weights1 = np.mean(np.array(list(map(weightTrainer, np.full((data.shape[0], 8), weights[1]), inputData, classLabels, np.full((data.shape[0], 1), 2)))), 0)
    weights2 = np.mean(np.array(list(map(weightTrainer, np.full((data.shape[0], 8), weights[2]), inputData, classLabels, np.full((data.shape[0], 1), 3)))), 0)
    weights = [weights0, weights1, weights2]
    currentAccuracy, precisionAndRecallArray, confusionMatrixArray = test(weights, data, True)
    # If a set of weights with a better accuracy on the training data is found...
    if (currentAccuracy > currentBestAccuracy):
      currentBestAccuracy = currentAccuracy
      finalWeights = weights
      counter = 0
    counter += 1
    totalIterations += 1
  return finalWeights, totalIterations

def predictionDecider(class1, class2, class3):
  # Obviously showing a bias for predicting class 1 followed by class 2 and 3
  if (class1 == 1):
    return 1
  if (class2 == 1):
    return 2
  if (class3 == 1):
    return 3
  return 1

def getTruePositives(prediction, classLabel, targetClass):
  if (targetClass == classLabel and prediction == targetClass):
    return 1
  else:
    return 0

def getTrueNegatives(prediction, classLabel, targetClass):
  if (targetClass != classLabel and prediction != targetClass):
    return 1
  else:
    return 0

def getFalsePositives(prediction, classLabel, targetClass):
  if (targetClass != classLabel and prediction == targetClass):
    return 1
  else:
    return 0

def getFalseNegatives(prediction, classLabel, targetClass):
  if (targetClass == classLabel and prediction != targetClass):
    return 1
  else:
    return 0

def test(weights, data, isTraining):
  inputData = data[0:, 0:-1]
  classLabels = data[0:, -1:].flatten()
  class1 = np.array(list(map(predict, np.full((data.shape[0], 8), weights[0]), inputData)))
  class2 = np.array(list(map(predict, np.full((data.shape[0], 8), weights[1]), inputData)))
  class3 = np.array(list(map(predict, np.full((data.shape[0], 8), weights[2]), inputData)))

  prediction = predictionDecider(class1, class2, class3)

  precisionAndRecallArray = []
  confusionMatrixArray = []

  if (not isTraining):

    truePositivesC1 = np.sum(getTruePositives(prediction, classLabels, np.full(data.shape[0], 1)))
    truePositivesC2 = np.sum(getTruePositives(prediction, classLabels, np.full(data.shape[0], 2)))
    truePositivesC3 = np.sum(getTruePositives(prediction, classLabels, np.full(data.shape[0], 3)))

    trueNegativesC1 = np.sum(getTrueNegatives(prediction, classLabels, np.full(data.shape[0], 1)))
    trueNegativesC2 = np.sum(getTrueNegatives(prediction, classLabels, np.full(data.shape[0], 2)))
    trueNegativesC3 = np.sum(getTrueNegatives(prediction, classLabels, np.full(data.shape[0], 3)))

    falsePositivesC1 = np.sum(getFalsePositives(prediction, classLabels, np.full(data.shape[0], 1)))
    falsePositivesC2 = np.sum(getFalsePositives(prediction, classLabels, np.full(data.shape[0], 2)))
    falsePositivesC3 = np.sum(getFalsePositives(prediction, classLabels, np.full(data.shape[0], 3)))

    falseNegativesC1 = np.sum(getFalseNegatives(prediction, classLabels, np.full(data.shape[0], 1)))
    falseNegativesC2 = np.sum(getFalseNegatives(prediction, classLabels, np.full(data.shape[0], 2)))
    falseNegativesC3 = np.sum(getFalseNegatives(prediction, classLabels, np.full(data.shape[0], 3)))


    class1Precision = truePositivesC1 / (truePositivesC1 + falsePositivesC1)
    class2Precision = truePositivesC2 / (truePositivesC2 + falsePositivesC2)
    class3Precision = truePositivesC3 / (truePositivesC3 + falsePositivesC3)

    class1Recall = truePositivesC1 / (truePositivesC1 + falseNegativesC1)
    class2Recall = truePositivesC2 / (truePositivesC2 + falseNegativesC2)
    class3Recall = truePositivesC3 / (truePositivesC3 + falseNegativesC3)

    precisionAndRecallArray = [[class1Precision, class1Recall], [class2Precision, class2Recall], [class3Precision, class3Recall]]

    confusionMatrixArray = [[[truePositivesC1, falsePositivesC1],
                            [falseNegativesC1, trueNegativesC1]],
                            
                            [[truePositivesC2, falsePositivesC2],
                            [falseNegativesC2, trueNegativesC2]],
                            
                            [[truePositivesC3, falsePositivesC3],
                            [falseNegativesC3, trueNegativesC3]]]

  totalPositivesMatrix = getTotalCorrect(prediction, classLabels)
  accuracy = np.sum(totalPositivesMatrix) / totalPositivesMatrix.shape[0]
  
  return accuracy, precisionAndRecallArray, confusionMatrixArray

# Vectorizing these two functions (essentially creates a for loop)
getTruePositives = np.vectorize(getTruePositives)
getTrueNegatives = np.vectorize(getTrueNegatives)
getFalsePositives = np.vectorize(getFalsePositives)
getFalseNegatives = np.vectorize(getFalseNegatives)
getTotalCorrect = np.vectorize(getTotalCorrect)
predictionDecider = np.vectorize(predictionDecider)

main()
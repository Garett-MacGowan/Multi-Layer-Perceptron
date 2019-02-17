import numpy as np

def main(relFilePath, hasColumnLabel, hasID):
  trainSet, testSet, validationSet = trainTestValidationSplit(relFilePath, hasColumnLabel, hasID)
  # print('testSet: \n', testSet)
  # print('trainSet: \n', trainSet)
  # print('validationSet: \n', validationSet)

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
  return trainSet, testSet, validationSet

# Parameters are (String relativeFilePath, Boolean columnLablesPresent, Boolean IDColumnPresent)
main('GlassData.csv', True, True)
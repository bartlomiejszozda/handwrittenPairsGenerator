import os
import glob

def getSortedListOfEnoughRecognizedPairs(recognizeThreshold, numOfGenerated, recognizeQuantityPath):
  recognizedPairs = []
  i=0
  with open(os.path.join(recognizeQuantityPath), 'r') as fileWithStats:
                            #when change file, then be aware of change numOfGenerated number
    pass
    # fileLines = fileWithStats.readlines()
    # for line in fileLines:
    fileWithStats.readline()#ignore first line
    line = fileWithStats.readline()
    while line:
      pairString, numOfRecognizedString = line.split(' ')[0], line.split(' ')[1]
      recognizeProbability = float(numOfRecognizedString)/(numOfGenerated)
      if recognizeProbability >= recognizeThreshold:
        recognizedPairs.append([pairString, recognizeProbability])
        i+=1
      line = fileWithStats.readline()
    fileWithStats.close()
  print(i)
  recognizedPairs.sort(key=lambda x: x[1], reverse=True)
  return recognizedPairs


def splitPairsFairly(sortedPairsWithRecognitionVal, numOfProcess):
  pairsPerProcess = [[] for i in range(numOfProcess)]
  processNum=0
  for pair_recognitionVal in sortedPairsWithRecognitionVal:
    pairsPerProcess[processNum].append(pair_recognitionVal[0])
    processNum+=1
    processNum=processNum%numOfProcess
  return pairsPerProcess


def checkIfPathNotExist(g_r_c_dirName, i):
  generatedPaths = os.path.join('./data/generatedPairs', g_r_c_dirName + str(i))
  recognizedPaths = os.path.join('./data/recognizedPairs', g_r_c_dirName + str(i))
  if os.path.exists(generatedPaths) or os.path.exists(recognizedPaths):
    os.sys.exit('path exist, try to give another dirname')


def createCommands(numOfProcess, numOfGenerated, recognizeThreshold, recognizeQuantityPath, howMany, g_r_c_dirName, g_askIfCopyToGans):
  processes=[]
  sortedPairsWithRecognitionVal = getSortedListOfEnoughRecognizedPairs(recognizeThreshold, numOfGenerated, recognizeQuantityPath)
  pairsSplittedPerProcess = splitPairsFairly(sortedPairsWithRecognitionVal, numOfProcess)
  i=0
  for pairsPerProcess in pairsSplittedPerProcess:
    checkIfPathNotExist(g_r_c_dirName, i)
    pairsPerProcess = ' '.join(pairsPerProcess)
    process = 'python3 generateOnlyEnoughRecognizedPairs.py --g_howMany {0} --g_r_c_dirName="{1}{2}" --g_text="{3}" --g_askIfCopyToGans={4}'.format(howMany, g_r_c_dirName, i, pairsPerProcess, g_askIfCopyToGans)
    print(process)
    processes.append(process)
    i+=1
  return processes

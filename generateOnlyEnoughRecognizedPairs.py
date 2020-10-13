import argparse
import datetime
import os
import shutil
from itertools import product
from subprocess import call
from generatorNetworks.RNN.myGenerate import generate
from helpers.filesCounter.countFiles import CountFiles
from helpers.convertToMnistFormat.convert_to_mnist_format import main as convertToMnistFormat_main
from helpers.renameFiles.renameFiles import RenameFiles
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')



allLetters = "abcdefghijklmnopqrstuvwxyz"
countedPairs = []#TODO refactor (create class?)


def generateAndCountRecognized(pairsToGenerate, howMany, numOfGenerated):
  generate([], 'manyPairsInText', howMany, args.g_r_c_dirName, pairsToGenerate, numOfGenerated, False, args.g_generateMode, args.g_size)
  call(
    'cd ./recognitionNetworks/SimpleHTR/src ; python3 main.py --sourceDir={0} --destDir={1} --destNotRecognizedDir={2} --numOfGenerated={3}'.format(
      os.path.join("../../../", args.g_destPath, args.g_r_c_dirName), os.path.join("../../../", args.r_destPath, args.g_r_c_dirName),
      os.path.join("../../../", args.r_destNotRecognizedPath, args.g_r_c_dirName), numOfGenerated), shell=True)

  countFiles()


def dismissPoorlyRecognizedPairs(recognizeThreshold, numOfGenerated):
  global countedPairs
  recognizedPairs = ""
  for el in countedPairs:
    pairString, numOfRecognized = el[0], el[1]
    recognizeProbability = float(numOfRecognized)/(numOfGenerated)
    if recognizeProbability >= recognizeThreshold:
      recognizedPairs += pairString + " "
    else:
      deleteDir(os.path.join(args.r_destPath, args.g_r_c_dirName, pairString))
  return recognizedPairs


def createInputFileForGenerator(recognizeThreshold, numOfGenerated):
  global countedPairs
  whatToGenerate = []
  for el in countedPairs:
    pairString, numOfRecognized = el[0], el[1]
    recognizeProbability = float(numOfRecognized)/(numOfGenerated)
    if recognizeProbability >= float(recognizeThreshold):
      howMuchIsMissing = numOfGenerated - int(numOfRecognized)
      numToGenerate = int(howMuchIsMissing/recognizeProbability)
      whatToGenerate.append((pairString, numToGenerate))
      print(pairString + ' ' + str(numToGenerate))
  return whatToGenerate

def createInputFileForGenerator_FromFile(recognizeThreshold, numOfGenerated):
  global countedPairs
  with open(os.path.join(args.g_destPath, args.g_r_c_dirName, 'generatorInputFile.csv'), 'w+') as generatorInput:
    for el in countedPairs:
      pairString, numOfRecognized = el[0], el[1]
      recognizeProbability = float(numOfRecognized)/(numOfGenerated)
      if recognizeProbability >= float(recognizeThreshold):
        howMuchIsMissing = numOfGenerated - int(numOfRecognized)
        numToGenerate = int(howMuchIsMissing/recognizeProbability)
        generatorInput.write(pairString + ',' + str(numToGenerate) + '\n')
        print(pairString + ' ' + str(numToGenerate))
    generatorInput.close()


def deleteDir(path):
  if os.path.isdir(path):
    shutil.rmtree(path)
  else:
    print('given path is not a directory')


def pairsListToString(pairs):
  allPairsString = ""
  for firstLetter, secondLetter in pairs:
    allPairsString += firstLetter + secondLetter + " "
  return allPairsString


def convertToMnist():
  srcPath = os.path.join(args.r_destPath, args.g_r_c_dirName)
  dstPath = os.path.join(args.c_destPath, args.g_r_c_dirName, 'dataset', 'raw')
  print(srcPath)
  convertToMnistFormat_main(srcPath, "train", dstPath)
  convertToMnistFormat_main(srcPath, "test", dstPath)
  print('mnistTest convertedFrom', srcPath)


def generateAndDismissPoorlyRecognizedPairs(recognizeThreshold, howMany, numOfGenerated, pairsToGenerate):
  generateAndCountRecognized(pairsToGenerate, howMany, numOfGenerated)
  recognizedPairsString = dismissPoorlyRecognizedPairs(recognizeThreshold, numOfGenerated+howMany)
  print(recognizedPairsString)
  return recognizedPairsString


def getOrCreatePath(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def copyForGans():
  gansDataPath = "mainNetworks/gans/generatedMNIST/MNIST"
  if os.path.exists(os.path.join(gansDataPath, "dataset")):
    print('Do you want to overwrite actual mainNetworks/gans/generatedMNIST/MNIST/dataset (y/n)')
  else:
    print('Do you want to copy created MNIST into actual mainNetworks/gans/generatedMNIST/MNIST/dataset? (y/n)')
  copy = input()
  if copy == "y":
    getOrCreatePath(gansDataPath)
    call('cp -r {0} {1}'.format(os.path.join(args.c_destPath, args.g_r_c_dirName, 'dataset'), gansDataPath), shell=True)
    print("copied")


def fixNumbersInImagesNames():
  path = os.path.join(args.r_destPath, args.g_r_c_dirName)
  renameFiles = RenameFiles(path, ".png", "blurred")
  renameFiles.rename()


def generateToBalance(numOfGenerated, recognizeThreshold):
  whatToGenerate = createInputFileForGenerator(recognizeThreshold, numOfGenerated)
  generate(whatToGenerate, 'None', 0, args.g_r_c_dirName, 'None', numOfGenerated, True, args.g_generateMode, args.g_size)#TODO to much args, deletion or some serving class will be needed
  call(
    'cd ./recognitionNetworks/SimpleHTR/src ; python3 main.py --sourceDir={0} --destDir={1} --destNotRecognizedDir={2} --numOfGenerated={3}'.format(
      os.path.join("../../../", args.g_destPath, args.g_r_c_dirName), os.path.join("../../../", args.r_destPath, args.g_r_c_dirName),
      os.path.join("../../../", args.r_destNotRecognizedPath, args.g_r_c_dirName), numOfGenerated), shell=True)


def countFiles():
  global countedPairs
  path = os.path.join(args.r_destPath, args.g_r_c_dirName)
  countFiles = CountFiles()
  countedPairs = countFiles.count(path)


def main():
  print(__file__ + str(datetime.datetime.now()))
  from tensorflow.python.client import device_lib
  print(device_lib.list_local_devices())

  firstRecognizeThreshold = 0.40
  lastRecognizeThreshold = 0.37
  recognizeThresholdDecreaseBy = 0.01
  generatePerIteration = 100
  generatePerIterationMultiplyBy = 2

  pairsToGenerate = args.g_text

  numOfGenerated = 0
  recognizeThreshold = firstRecognizeThreshold
  while numOfGenerated < args.g_howMany:
    if numOfGenerated + generatePerIteration >= args.g_howMany:
      generatePerIteration = args.g_howMany - numOfGenerated
    recognizedPairsString = generateAndDismissPoorlyRecognizedPairs(recognizeThreshold=recognizeThreshold,
                                                                    howMany=generatePerIteration,
                                                                    numOfGenerated=numOfGenerated,
                                                                    pairsToGenerate=pairsToGenerate)
    numOfGenerated += generatePerIteration
    generatePerIteration *= generatePerIterationMultiplyBy
    pairsToGenerate = recognizedPairsString
    print(pairsToGenerate)
    if recognizeThreshold > lastRecognizeThreshold:
      recognizeThreshold -= recognizeThresholdDecreaseBy
    print('numOfGenerated: ' + str(numOfGenerated) + 'args.g_howMany' + str(args.g_howMany) + 'recognizeThreshold: ' + str(recognizeThreshold))

  generateToBalance(numOfGenerated, recognizeThreshold)

  fixNumbersInImagesNames()
  countFiles()
  convertToMnist()
  print('copy for gans/generateOnlyEnough...' + str(args.g_askIfCopyToGans))
  if('True' == args.g_askIfCopyToGans):
    copyForGans()
  return 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # args parsed with this prefixes:
  # g(generator),
  # r(recognition),
  # c(convert to Mnist)

  # base parameters
  parser.add_argument('--g_text', dest='g_text', type=str, default=None)  # used
  parser.add_argument('--g_howMany', dest='g_howMany', type=int, default=1)  # used
  parser.add_argument('--g_r_c_dirName', dest='g_r_c_dirName', type=str, default='default')  # used
  parser.add_argument('--g_askIfCopyToGans', dest='g_askIfCopyToGans', type=str, default='True')  # used

  # optional parameters
  parser.add_argument('--g_size', dest='g_size', type=str, default='28x28')  # normal, 28x28
  parser.add_argument('--g_generateMode', dest='g_generateMode', type=str,
                      default='onlyBlurred')  # onlyBlurred, onlyNormal, all
  parser.add_argument('--g_lineWidth', dest='g_lineWidth', type=int, default=18)
  parser.add_argument('--g_blurRadius', dest='g_blurRadius', type=int, default=6)
  # paths
  parser.add_argument('--g_destPath', dest='g_destPath', type=str, default='data/generatedPairs')
  parser.add_argument('--r_destPath', dest='r_destPath', type=str, default='data/recognizedPairs')
  parser.add_argument('--r_destNotRecognizedPath', dest='r_destNotRecognizedPath', type=str,
                      default='data/notRecognizedPairs')
  parser.add_argument('--c_destPath', dest='c_destPath', type=str, default='data/createdMNIST')
  args = parser.parse_args()
  main()


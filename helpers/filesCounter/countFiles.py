import datetime
import os
import argparse

class CountFiles:
  __minNumOfElements = -1
  __dirWithMoreElementsThanMin = 0
  __sumOfAllDirsAndFiles = 0

  # def cleanUpResult(self):
  #   self.__result.clear()

  def saveCountingResultToFile(self, pathToFile, result):
    with open(pathToFile, 'w+') as fileWithStats:
      for el in result:
        logString = "{0},{1}\n".format(el[0], el[1]);
        fileWithStats.write(logString)
      fileWithStats.close()

  def recoursiveCountByPath(self, path, result):
    if os.path.isdir(path):
      numOfElements = len(os.listdir(path))
      if numOfElements > self.__minNumOfElements:
        self.__dirWithMoreElementsThanMin += 1
        elName = os.path.basename(os.path.normpath(path))
        result.append((elName, numOfElements))
        self.__sumOfAllDirsAndFiles += numOfElements
      for element in os.listdir(path):
        pathToEl = os.path.join(path, element)
        self.recoursiveCountByPath(pathToEl, result)


  def count(self, path):
    if path:
      result = []
      self.recoursiveCountByPath(path, result)
      print('num of elements with dirs more than ', self.__minNumOfElements, ': ', self.__dirWithMoreElementsThanMin)
      print('num of all dirs and files in path: ', self.__sumOfAllDirsAndFiles)

      pathToFile = os.path.join(path, 'recognizedPairsQuantity.txt')
      self.saveCountingResultToFile(pathToFile, result)
      return result



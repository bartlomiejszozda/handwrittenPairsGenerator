from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import shutil

import cv2
import editdistance

from os.path import dirname, abspath, join
import sys
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

"main function"



class Recognize:
  class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'

  def __init__(self, train, validate, beamsearch, wordbeamsearch, dump, sourceDir, destDir, destNotRecognizedDir, numOfGenerated):
    self.__train = train
    self.__validate = validate
    self.__beamsearch = beamsearch
    self.__wordbeamsearch = wordbeamsearch
    self.__dump = dump
    self.__sourceDir = sourceDir
    self.__destDir = destDir
    self.__destNotRecognizedDir = destNotRecognizedDir
    self.__numOfGenerated = numOfGenerated
    pass

  def train(self, model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
      epoch += 1
      print('Epoch:', epoch)

      # train
      print('Train NN')
      loader.trainSet()
      while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        batch = loader.getNext()
        loss = model.trainBatch(batch)
        print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

      # validate
      charErrorRate = self.validate(model, loader)

      # if best validation accuracy so far, save model parameters
      if charErrorRate < bestCharErrorRate:
        print('Character error rate improved, save model')
        bestCharErrorRate = charErrorRate
        noImprovementSince = 0
        model.save()
        open(self.FilePaths.fnAccuracy, 'w').write(
          'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
      else:
        print('Character error rate not improved')
        noImprovementSince += 1

      # stop training if no more improvement in the last x epochs
      if noImprovementSince >= earlyStopping:
        print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
        break


  def validate(self, model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
      iterInfo = loader.getIteratorInfo()
      print('Batch:', iterInfo[0], '/', iterInfo[1])
      batch = loader.getNext()
      (recognized, _) = model.inferBatch(batch)

      print('Ground truth -> Recognized')
      for i in range(len(recognized)):
        numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
        numWordTotal += 1
        dist = editdistance.eval(recognized[i], batch.gtTexts[i])
        numCharErr += dist
        numCharTotal += len(batch.gtTexts[i])
        print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate


  def infer(self, model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    # print('Recognized:', '"' + recognized[0] + '"')
    # print('recognized ', recognized, 'with probability:', probability[0])
    return recognized[0]


  def prepareDecoder(self):
    decoderType = DecoderType.BestPath
    if self.__beamsearch:
      decoderType = DecoderType.BeamSearch
    elif self.__wordbeamsearch:
      decoderType = DecoderType.WordBeamSearch
    return decoderType


  def testRecognition(self, decoderType):
    # print(open(FilePaths.fnAccuracy).read())
    model = Model(open(self.FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=self.__dump)
    self.infer(model, self.FilePaths.fnInfer)


  def countRecognized_FromDirByName(self, decoderType):
    model = Model(open(self.FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=self.__dump)
    countRecognized = 0
    countNotRecognized = 0
    for subDirectory in os.listdir(self.__sourceDir):
      dirPath = os.path.join(self.__sourceDir, subDirectory)
      for filename in os.listdir(dirPath):
        # if filename < 'wc26':
        #   continue
        if filename.endswith(".png"):
          pathToFile = os.path.join(dirPath, filename)
          # print(pathToFile)
          # print ('we suppose to recognize: ' + subDirectory)
          try:
            recognizedText = self.infer(model, pathToFile)
          except ValueError as ve:
            print('skip this image because of: ', ve)
            continue
          if subDirectory == recognizedText:
            print('recognized ', recognizedText)
            countRecognized += 1
          else:
            countNotRecognized += 1

        if ((countRecognized % 100) == 1):
          print(countRecognized)
          print(countNotRecognized)


  def trainOrValidate(self, decoderType):
    # load training data, create TF model
    loader = DataLoader(self.FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
    # save characters of model for inference mode
    open(self.FilePaths.fnCharList, 'w').write(str().join(loader.charList))
    # save words contained in dataset into file
    open(self.FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
    # execute training or validation
    if self.__train:
      model = Model(loader.charList, decoderType)
      self.train(model, loader)
    elif self.__validate:
      model = Model(loader.charList, decoderType, mustRestore=True)
      self.validate(model, loader)


  def getOrCreatePath(self, directory, name):
    if not os.path.exists(directory):
      os.makedirs(directory)
    pathToImage = os.path.join(directory, name)
    return pathToImage


  def pairGeneratedInThisIteration(self, filename):
    pairName = filename.split('_')[0]
    pairNum = pairName[2:]
    return int(pairNum) >= int(self.__numOfGenerated)



  def recognizeAndSave_byGivenPath(self, decoderType):
    print(__file__ + str(datetime.datetime.now()))
    model = Model(open(self.FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=self.__dump)
    countRecognized = 0
    countNotRecognized = 0
    for subDirectory in os.listdir(self.__sourceDir):
      dirPath = os.path.join(self.__sourceDir, subDirectory)
      if os.path.isdir(dirPath):
        for filename in os.listdir(dirPath):
          if not self.pairGeneratedInThisIteration(filename):
            # print('filename: ', filename)
            continue
          # if filename < 'ky':
          #   continue
          if filename.endswith(".png"):
            pathToFile = os.path.join(dirPath, filename)
            # print(pathToFile)
            # print ('we suppose to recognize: ' + subDirectory)
            try:
              recognizedText = self.infer(model, pathToFile)
            except ValueError as ve:
              print('skip this image because of: ', ve)
              continue
            if subDirectory == recognizedText:
              print('recognized ', recognizedText)
              countRecognized += 1
              shutil.copy(pathToFile, self.getOrCreatePath(os.path.join(self.__destDir, subDirectory), filename))

            else:
              countNotRecognized += 1
              if self.__destNotRecognizedDir:
                shutil.copy(pathToFile, self.getOrCreatePath(os.path.join(self.__destNotRecognizedDir, subDirectory), filename))
      else:
        print('not dir: ' + dirPath + 'ommited')

        # if ((countRecognized % 100) == 1):
        #   print(countRecognized)
        #   print(countNotRecognized)


  def recognize(self):
    decoderType = self.prepareDecoder()

    # train or validate on IAM dataset
    if self.__train or self.__validate:
      self.trainOrValidate(decoderType)
    elif self.__sourceDir and self.__destDir:
      self.recognizeAndSave_byGivenPath(decoderType)
    elif self.__sourceDir and not self.__destDir:
      self.countRecognized_FromDirByName(decoderType)
    # infer text on test image
    else:
      self.testRecognition(decoderType)
  #
  #
  # def main(self):
  #   decoderType = self.prepareDecoder()
  #
  #   # train or validate on IAM dataset
  #   if self.__train or self.__validate:
  #     self.trainOrValidate(decoderType)
  #   elif self.__sourceDir and self.__destDir:
  #     self.recognizeAndSave_byGivenPath(decoderType)
  #   elif self.__sourceDir and not self.__destDir:
  #     self.countRecognized_FromDirByName(decoderType)
  #
  #   # infer text on test image
  #   else:
  #     self.testRecognition(decoderType)


if __name__ == '__main__':
  # optional command line args
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', help='train the NN', action='store_true')
  parser.add_argument('--validate', help='validate the NN', action='store_true')
  parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
  parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
                      action='store_true')
  parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
  parser.add_argument('--sourceDir', help='directory with images to recognition', type=str)
  parser.add_argument('--destDir', help='directory with recognized images', type=str)
  parser.add_argument('--destNotRecognizedDir', help='directory with not recognized images', type=str)
  parser.add_argument('--numOfGenerated', help='num of already generated Images', type=str)
  args = parser.parse_args()

  recognize = Recognize(args.train, args.validate, args.beamsearch, args.wordbeamsearch, args.dump, args.sourceDir, args.destDir, args.destNotRecognizedDir, args.numOfGenerated)
  recognize.recognize()

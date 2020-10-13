import os

def dismissPoorlyRecognizedPairs(recognizeThreshold, numOfGenerated):
  recognizedPairs = ""
  i=0
  with open(os.path.join('../showcase/usefulRecognitionQuantity/recognizeQuantity_200_NOTbalanced.txt'), 'r') as fileWithStats:
    # fileLines = fileWithStats.readlines()
    # for line in fileLines:
    fileWithStats.readline()#ignore first line
    line = fileWithStats.readline()
    while line:
      pairString, numOfRecognizedString = line.split(' ')[0], line.split(' ')[1]
      recognizeProbability = float(numOfRecognizedString)/(numOfGenerated)
      if recognizeProbability >= recognizeThreshold:
        recognizedPairs += pairString + " "
        i+=1
      line = fileWithStats.readline()
    fileWithStats.close()
  print(i)
  return recognizedPairs

def main():
  print("dupa")
  print(dismissPoorlyRecognizedPairs(0.43, 200))


if __name__ == '__main__':
  main()

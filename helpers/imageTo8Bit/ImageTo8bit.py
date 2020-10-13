import os
import sys

from PIL import Image


def main(argv):
  iter = 0
  mainPath = '/mnt/c/Users/szozda/thesis/newGans/generated/multiPairs'
  for dirname in os.listdir(mainPath):
    dirPath = os.path.join(mainPath, dirname)
    print dirPath
    for filename in os.listdir(dirPath):
      if filename.endswith(".png"):
        pathToFile = os.path.join(dirPath, filename)
        img = Image.open(pathToFile).convert('L')
        img.save(pathToFile)
        if ((iter % 1000) == 0):
          print(iter)
        iter += 1


if __name__ == '__main__':
  main(sys.argv)

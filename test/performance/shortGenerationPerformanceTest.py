from subprocess import call
import time
import os
class testPair:
  num = 43
  str = "yo"

def cleanUpDirs():
  call("rm -r data/createdMNIST/testPerformance", shell=True)
  call("rm -r data/generatedPairs/testPerformance", shell=True)
  call("rm -r data/recognizedPairs/testPerformance", shell=True)
  call("rm -r data/notRecognizedPairs/testPerformance", shell=True)

def checkNumOfGenerated(g_how_many, g_r_c_dirName, pairNumber):
  pathToRecognizedDir = os.path.join("data/recognizedPairs", g_r_c_dirName + str(pairNumber))
  accuracy = 0.2

  numOfElementsInDir = len(os.listdir(os.path.join(pathToRecognizedDir, testPair.str)))
  assert (1 - accuracy)*g_how_many < numOfElementsInDir < (1 + accuracy)*g_how_many, "numOfElementsInDir={} should be equal with accuracy={}% to g_how_many={}".format(numOfElementsInDir, accuracy*100, g_how_many)

  with open(os.path.join(pathToRecognizedDir, "recognizedPairsQuantity.txt"), "r") as fileWithStats:
    fileWithStats.readline()#ignore first line
    line = fileWithStats.readline()
    pairString, numOfRecognized = line.split(',')[0], line.split(',')[1]
    assert (1 - accuracy) * g_how_many < int(numOfRecognized) < (1 + accuracy)*g_how_many ,\
      "numOfRecognized={} should be equal with accuracy={}% to g_how_many={}".format(numOfRecognized, accuracy * 100, g_how_many)
    assert int(numOfRecognized) == numOfElementsInDir, "numOfRecognized={} should be equal to numOfElementsInDir={}".format(numOfRecognized, numOfElementsInDir)
    print("numOfElementsInDIr:{}, numOfRecognized:{}".format(numOfElementsInDir, numOfRecognized))


def test(minTime, maxTime):
  cleanUpDirs()
  timeStart = time.perf_counter()
  g_howMany, pairNumber, g_r_c_dirName = 100, testPair.num, "testPerformance/test"
  returnCode = call("pwd && python3 multiProcessGenerator.py --g_howMany={} --mp_numAllProcesses=98 --g_r_c_dirName={} --mp_range={}:{} --g_askIfCopyToGans=False".format(g_howMany, g_r_c_dirName, pairNumber, (pairNumber+1)), shell=True)
  timeEnd = time.perf_counter()
  performanceTestTime = timeEnd - timeStart
  print("returnCode={}, time={}".format(returnCode, performanceTestTime))
  assert returnCode == 0, "return code != 0"
  assert minTime <= performanceTestTime <= maxTime, "performance test takes too long or incredibly short. performanceTestTime={}"
  checkNumOfGenerated(g_howMany, g_r_c_dirName, pairNumber)
  return performanceTestTime

if __name__ == "__main__":
  test(50, 180)

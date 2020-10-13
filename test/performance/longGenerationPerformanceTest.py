from subprocess import call
import time

def cleanUpDirs():
  call("rm -r ../data/createdMNIST/testPerformance", shell=True)
  call("rm -r ../data/generatedPairs/testPerformance", shell=True)
  call("rm -r ../data/recognizedPairs/testPerformance", shell=True)

def test():
  timeStart = time.perf_counter()
  returnCode = call("cd .. && pwd && python3 multiProcessGenerator.py --g_howMany=10 --mp_numAllProcesses=98 --g_r_c_dirName='testPerformance/test' --mp_range=40:50 --g_askIfCopyToGans=False", shell=True)
  timeEnd = time.perf_counter()
  print("time = {}".format(timeEnd-timeStart))
  cleanUpDirs()
  print ("returnCode is " + str(returnCode))
  if returnCode != 0:
      exit("return code != 0")
  return (timeEnd-timeStart)


from helpers.distributePairsBetweenGenerators import createCommands
import os
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description="args parsed with this prefixes: "\
"mp(multiProcessGenerator), g(generator), r(recognition), c(convert Mnist)\n"\
"every process will create specific pairs."\
"For example: We can generate 98 pairs, so running 20 processes will give you 5 pairs per process in most cases.")
parser.add_argument('--g_howMany', dest='g_howMany', type=int, default=None, help='number of every pair elements')
parser.add_argument('--mp_numAllProcesses', dest='mp_numAllProcesses', type=int, default=None, help='divide all (98) pairs into processes')
parser.add_argument('--g_r_c_dirName', dest='g_r_c_dirName', type=str, default='generateParalell/Process', help='directory name')
parser.add_argument('--mp_range', dest='mp_range', type=str, default=None, help='range for processes to run (0:5 will run processes 0,1,...,4)')
parser.add_argument('--g_askIfCopyToGans', dest='g_askIfCopyToGans', type=str, default='True', help='ask if copy to gans directory? Useful in tests.')
args = parser.parse_args()

def runProcess(process):
  os.system(process)


def getRangeOfProcessesToRun():
  rangeFrom = int(args.mp_range.split(':')[0])
  rangeTo = int(args.mp_range.split(':')[1])
  numProcessesToRun = rangeTo - rangeFrom
  return rangeFrom, rangeTo, numProcessesToRun


def main():
  numOfAllProcesses = args.mp_numAllProcesses
  recognizeQuantityPath = './showcase/usefulRecognitionQuantity/recognizeQuantity_200_NOTbalanced.txt'
  numOfGeneratedInRecognizeFile = 200
  processes = createCommands(numOfAllProcesses, numOfGeneratedInRecognizeFile, 0.43, recognizeQuantityPath, args.g_howMany, args.g_r_c_dirName, args.g_askIfCopyToGans)
  rangeFrom, rangeTo, numProcessesToRun = getRangeOfProcessesToRun()
  processesToRun = processes[rangeFrom:rangeTo]
  print("will run:\n" + str(processesToRun))

  pool = Pool(numProcessesToRun)
  pool.map(runProcess, processesToRun)


if __name__ == '__main__':
  main()


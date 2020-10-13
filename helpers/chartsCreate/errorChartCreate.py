import os.path
from matplotlib import pyplot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chartsNames', dest='chartsNames', type=str, default='')
parser.add_argument('--logDir', dest='logDir', type=str, default='')
parser.add_argument('--logFile', dest='logFile', type=str, default='')
args = parser.parse_args()

def getOrCreatePath(directory, name):
  if not os.path.exists(directory):
    os.makedirs(directory)
  pathToFile = os.path.join(directory, name)
  print('open log file: ', os.path.join(directory, name))
  return pathToFile


#'/mnt/c/Users/szozda/newThesis/mainNetworks/GansFromScratch/data2/logs', 'logFrom3000Generation_100numSamples_epoch123To455.log'
def readValuesForChart(simpleLogFilePath):
  epochs, batches, discrLosses, genLosses, Dxs, DGzs = [], [], [], [], [], []
  with open(simpleLogFilePath, 'r') as fileWithStats:
    # fileWithStats.readline()#epoch
    # fileWithStats.readline()#batch
    # fileWithStats.readline()#discrLoss
    # fileWithStats.readline()#genLoss
    # fileWithStats.readline()#Dx
    # fileWithStats.readline()#DGz
    # fileWithStats.readline()#empty line

    epochs.append(int(fileWithStats.readline().split(':')[1][:-1]))
    maxEpoch = int(fileWithStats.readline().split(':')[1][:-1])
    batches.append(int(fileWithStats.readline().split(':')[1][:-1]))
    maxBatch = int(fileWithStats.readline().split('h')[1][:-1])
    discrLosses.append(float(fileWithStats.readline().split(':')[1][:-1]))
    genLosses.append(float(fileWithStats.readline().split(':')[1][:-1]) )
    Dxs.append(float(fileWithStats.readline().split(':')[1][:-1]))
    DGzs.append(float(fileWithStats.readline().split(':')[1][:-1]))
    lineExist = fileWithStats.readline()

    while lineExist:
      nextStatsExist = nextLine = fileWithStats.readline()
      if not nextLine:
        break
      if(int(nextLine.split(':')[1][:-1]) < 20):
        fileWithStats.readline()
        fileWithStats.readline()
        fileWithStats.readline()
        fileWithStats.readline()
        fileWithStats.readline()
        fileWithStats.readline()
        fileWithStats.readline()
        fileWithStats.readline()
        continue
      epochs.append(int(nextLine.split(':')[1][:-1]))
      fileWithStats.readline()#maxEpoch
      batches.append(int(fileWithStats.readline().split(':')[1][:-1]))
      fileWithStats.readline()#maxBatch
      discrLosses.append(float(fileWithStats.readline().split(':')[1][:-1]))
      genLosses.append(float(fileWithStats.readline().split(':')[1][:-1]))
      Dxs.append(float(fileWithStats.readline().split(':')[1][:-1]))
      DGzs.append(float(fileWithStats.readline().split(':')[1][:-1]))
      lineExist = fileWithStats.readline()

  return epochs, maxEpoch, batches, maxBatch, discrLosses, genLosses, Dxs, DGzs


def printStatistics(DGzs, Dxs, batches, discrLosses, epochs, genLosses, maxBatch, maxEpoch):
  print('epochs: ', epochs)
  print('maxEpoch: ', maxEpoch)
  print('batches: ', batches)
  print('maxBatch: ', maxBatch)
  print('discrLosses: ', discrLosses)
  print('genLosses: ', genLosses)
  print('Dxs: ', Dxs)
  print('DGzs: ', DGzs)
  # print(epochs, maxEpoch, batches, maxBatch, discrLosses, genLosses, Dxs, DGzs)


def main():
  if(args.chartsNames == '' or args.logDir == '' or args.logFile == ''):
    os.sys.exit('you should give all args: --chartsNames, --logDir and --logFile')
  simpleLogFilePath = getOrCreatePath(args.logDir, args.logFile)
  epochs, maxEpoch, batches, maxBatch, discrLosses, genLosses, Dxs, DGzs = readValuesForChart(simpleLogFilePath)
  # printStatistics(DGzs, Dxs, batches, discrLosses, epochs, genLosses, maxBatch, maxEpoch)
  plot = pyplot.plot(epochs, discrLosses, '.')
  fig = pyplot.gcf()
  fig.set_size_inches(20, 5)
  fig.savefig('discrLosses{}.png'.format(args.chartsNames), dpi=100)

  pyplot.clf()
  plot2 = pyplot.plot(epochs, genLosses, '.')
  fig2 = pyplot.gcf()
  fig2.set_size_inches(20, 5)
  fig2.savefig('genLosses{}.png'.format(args.chartsNames), dpi=100)

  pyplot.clf()
  plot2 = pyplot.plot(epochs, Dxs, '.')
  fig2 = pyplot.gcf()
  fig2.set_size_inches(20, 5)
  fig2.savefig('discrRealAccuracy{}.png'.format(args.chartsNames), dpi=100)

  pyplot.clf()
  plot2 = pyplot.plot(epochs, DGzs, '.')
  fig2 = pyplot.gcf()
  fig2.set_size_inches(20, 5)
  fig2.savefig('discrGenAccuracy{}.png'.format(args.chartsNames), dpi=100)

if __name__ == '__main__':
  main()


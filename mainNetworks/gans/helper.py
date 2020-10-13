import os

import numpy as np
import torch
from IPython import display
from matplotlib import pyplot
from tensorboardX import SummaryWriter
from torchvision import utils

lettersSorted = ['ab', 'ac', 'ad', 'ak', 'al', 'an', 'ar', 'as', 'at', 'aw', 'ba', 'bd', 'be', 'bn', 'bo', 'bs', 'bt',
                 'by', 'da', 'de', 'di', 'do', 'dr', 'em', 'fr', 'go', 'ha', 'he', 'hi', 'hm', 'ho', 'hp', 'hr', 'hw',
                 'ic', 'id', 'if', 'im', 'in', 'ip', 'ir',
                 'is', 'it', 'ma', 'me', 'mo', 'mp', 'mu', 'my', 'oe', 'of', 'oh', 'om', 'on', 'op', 'or', 'os', 'ot',
                 'ou', 'ow', 'oy',
                 'pa', 'pd', 'pe', 'ph', 'po', 'pr', 'ps', 'pu', 'py', 'sa', 'sd', 'se', 'sh', 'sm', 'sn', 'so', 'sp',
                 'sr', 'st', 'su', 'sw', 'ta', 'te', 'to', 'tw', 'ty', 'un', 'up', 'wa', 'wd', 'we', 'wh', 'wo', 'ws',
                 'wt', 'wy', 'yo'
                 ]

class Helper:

    def __init__(self, dataDir="data", continueWorking=False):
        if not continueWorking and os.path.exists('./{}'.format(dataDir)):
            print('this path is occupied, try another one', os.sys.stderr)
            os.sys.exit(1)
        self.dataDir = dataDir
        self.tensorBoard = SummaryWriter(comment=dataDir)


    #labels
    def printLabels(self, labels):
        for label, i in zip(labels, range(len(labels))):
            print(lettersSorted[int(label)]+" ", end='')
            if i%10 == 9:
                print()
    #status
    def showLossAndErrorsAndSaveInFile(self, epoch, numEpochs, nBatch, numBatches, dError, gError, dPredReal, dPredFake):
        self.__saveLossErrorsInFile(dError, dPredFake, dPredReal, epoch, gError, nBatch, numBatches, numEpochs)

        print('MaxEpoch: {},      numBatches:  {}'.format(numEpochs, numBatches))
        print('Epoch:    {},          Batch Num:   {}'.format(epoch,nBatch))
        print('D(x):       {:.5f}, D(G(z)): {:.5f}'.format(dPredReal.mean(), dPredFake.mean()))
        print('Discr Loss: {:.5f}, Gen Loss:{:.5f}'.format(dError, gError))


    def __saveLossErrorsInFile(self, dError, dPredFake, dPredReal, epoch, gError, nBatch, numBatches, numEpochs):
        simpleLogFilePath = self.__getOrCreatePath('./{}/logs/'.format(self.dataDir), 'simplerLog.txt')
        with open(simpleLogFilePath, 'a') as fileWithStats:
            fileWithStats.write('Epoch:{}\nMax Epoch:{}\nBatch:{}\nMax Batch{}\n'.format(
                epoch, numEpochs, nBatch, numBatches)
            )
            fileWithStats.write('Discriminator Loss:{:.10f}\nGenerator Loss:{:.10f}\n'.format(dError, gError))
            fileWithStats.write('D(x):{:.10f}\nD(G(z)):{:.10f}\n\n'.format(dPredReal.mean(), dPredFake.mean()))


    #model
    def saveModel(self, generator, discriminator, gOptimizer, dOptimizer, epoch, gError, dError, testNoise, constLabels):
        savedModelsDir = './{}/savedModels'.format(self.dataDir)
        self.__createPath(savedModelsDir)
        torch.save({'epoch': epoch, 'g_model_state_dict': generator.state_dict(),
                    'g_optimizer_state_dict': gOptimizer.state_dict(), 'g_error': gError, 'test_noise': testNoise, 'constLabels': constLabels,
                    }, '{}/G_epoch_{}'.format(savedModelsDir, epoch, testNoise, constLabels))
        torch.save({'epoch': epoch, 'd_model_state_dict': discriminator.state_dict(),
                    'd_optimizer_state_dict': dOptimizer.state_dict(), 'dError': dError
                    }, '{}/D_epoch_{}'.format(savedModelsDir, epoch))


    def loadModel(self, generator, discriminator, gOptimizer, dOptimizer, epoch):
        gCheckpoint = torch.load(
            './{}/savedModels/G_epoch_{}'.format(self.dataDir, epoch))
        dCheckpoint = torch.load(
            './{}/savedModels/D_epoch_{}'.format(self.dataDir, epoch))

        loadedEpoch = gCheckpoint['epoch']
        generator.load_state_dict(gCheckpoint['g_model_state_dict'])
        discriminator.load_state_dict(dCheckpoint['d_model_state_dict'])
        gOptimizer.load_state_dict(gCheckpoint['g_optimizer_state_dict'])
        dOptimizer.load_state_dict(dCheckpoint['d_optimizer_state_dict'])
        generator.eval()
        discriminator.eval()
        generator.train()
        discriminator.train()
        test_noise = gCheckpoint['test_noise']
        constLabels = gCheckpoint['constLabels']
        return loadedEpoch, test_noise, constLabels


    #images
    def logOnBoardAndSaveImages(self, images, epoch, nBatch, numBatches, logResultImages=False, imageNum=0):
        images = self.__checkImageType(images)
        grid = utils.make_grid(images, nrow=10, normalize=True, scale_each=True)

        #log On TensorBoard
        self.tensorBoard.add_image(self.dataDir + '/images', grid, nBatch+(numBatches*epoch))

        self.__saveImgs(grid, epoch, nBatch, logResultImages, imageNum)

    def __saveImgs(self, grid, epoch, nBatch, logResultImages=False, imageNum=0):
        imgDir = './{}/images'.format(self.dataDir)
        self.__createPath(imgDir)

        # Plot and save horizontal
        fig = pyplot.figure(figsize=(16, 16))
        pyplot.axis('off')
        pyplot.imshow(np.moveaxis(grid.numpy(), 0, -1))
        display.display(pyplot.gcf())
        if logResultImages and imageNum != 0:
            self.__saveResultImages(fig, epoch, nBatch, str(imageNum))
        else:
            fig.savefig('{}/epoch_{}_batch_{}.png'.format(imgDir, epoch, nBatch))
        pyplot.close()

    def __saveResultImages(self, fig, epoch, nBatch, imageNum=''):
        resultImgDir = './{}/resultImages'.format(self.dataDir)
        self.__createPath(resultImgDir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(resultImgDir, imageNum, epoch, nBatch))
        print('{}/{}_epoch_{}_batch_{}.png'.format(resultImgDir, imageNum, epoch, nBatch))

    def close(self):
        self.tensorBoard.close()

    #private helpers
    def __checkImageType(self, images):
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        return images

    def __createPath(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __getOrCreatePath(self, directory, name):
        if not os.path.exists(directory):
            os.makedirs(directory)
        pathToFile = os.path.join(directory, name)
        return pathToFile



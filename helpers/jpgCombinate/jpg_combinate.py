import os
import sys
from PIL import Image
def combinateImages(pathToRead, pathToSave, maxNumOfImages):
    for dir1 in range(0, 9, 2):
        dir2 = dir1 + 1
        if not os.path.exists(pathToSave + "/%d_%d" % (dir1, dir2)):
            os.makedirs(pathToSave + "/%d_%d" % (dir1, dir2))
        for num in range(1,(maxNumOfImages+1)):
            images = map(Image.open, [pathToRead + r"/%d/ (%d).png" % (dir1, num),
                                      pathToRead + r"/%d/ (%d).png" % (dir2, num)])
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('L', (total_width, max_height))

            x_offset = 0
            for im in images:
              new_im.paste(im, (x_offset,0))
              x_offset += im.size[0]

            new_im.save(pathToSave + "/%d_%d/ (%d).png" % (dir1, dir2, num))

pathToBaseDir = r"C:/Users/szozda/thesis/newGans/gans/png_mnist_png_converter"
pathToRead = pathToBaseDir + r"/mnist_png-master/mnist_png/testing"
pathToSave = pathToBaseDir + r"/mnist_png-master/mnist_png/testing_combined"
maxNumOfImages = 892
combinateImages(pathToRead, pathToSave, maxNumOfImages)

pathToRead = pathToBaseDir + r"/mnist_png-master/mnist_png/training"
pathToSave = pathToBaseDir + r"/mnist_png-master/mnist_png/training_combined"
maxNumOfImages = 5421
combinateImages(pathToRead, pathToSave, maxNumOfImages)


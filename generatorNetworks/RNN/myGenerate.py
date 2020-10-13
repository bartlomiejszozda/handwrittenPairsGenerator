import argparse
# import matplotlib
# matplotlib.use('pdf')
import matplotlib
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import product
from itertools import combinations

from PIL import ImageFilter, Image
import datetime

def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation, style, bias, force):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.compat.v1.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0] # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text] # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        # print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not force and finish[0, 0] > 0.8:
                # print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords


def getHorizontalMin(stroke):
    return min(stroke[:, 0])


def getHorizontalMax(stroke):
    return max(stroke[:, 0])


def isStrokesOverlapping(stroke1, stroke2):
    stroke1Min = getHorizontalMin(stroke1)
    stroke1Max = getHorizontalMax(stroke1)
    stroke2Min = getHorizontalMin(stroke2)
    stroke2Max = getHorizontalMax(stroke2)
    if (stroke2Min <= stroke1Min <= stroke2Max) or (stroke2Min <= stroke1Max <= stroke2Max):
        return True
    elif (stroke1Min <= stroke2Min <= stroke1Max) or (stroke1Min <= stroke2Max <= stroke1Max):
        return True
    else:
        return False


class StrokesOverlappingException(Exception):
    pass


def getClosestPointsBetweenStrokes_Horizontal(stroke1, stroke2):
    stroke1Min = getHorizontalMin(stroke1)
    stroke1Max = getHorizontalMax(stroke1)
    stroke2Min = getHorizontalMin(stroke2)
    stroke2Max = getHorizontalMax(stroke2)
    if isStrokesOverlapping(stroke1, stroke2):
        raise StrokesOverlappingException

    if (stroke1Max - stroke2Min) < (stroke1Min - stroke2Max):
        # print("neighbouring coordinates1: ", stroke1Max, stroke2Min)
        return stroke1Max, stroke2Min
    else:
        # print("neighbouring coordinates2: ", stroke2Max, stroke1Min)
        return stroke2Max, stroke1Min


def isStrokeBetween_Horizontal(point1, point2, strokes):
    if point1 > point2:
        point1, point2 = point2, point1
    for stroke in strokes:
        if point1 < getHorizontalMax(
                stroke) < point2:  # we find stroke which right border(or both borders) is between range
            # print(getVerticalMax(stroke), "is between: ", point1, point2)
            return True
        elif point1 < getHorizontalMin(stroke) < point2:  # we find stroke which left border is between range
            # print(getVerticalMin(stroke), "is between1: ", point1, point2)
            return True
        elif getHorizontalMin(stroke) < point1 < getHorizontalMax(stroke) and getHorizontalMin(
                stroke) < point2 < getHorizontalMax(
                stroke):  # we find stroke which are wider than our range. Which mean that part of that stroke is in range
            # print("range where we search for stroke: (", point1, ":", point2, ") is between another stroke: (", getVerticalMin(stroke), ":", getVerticalMax(stroke), ")")
            return True
    return False


def findWidestEmptyGapBetweenAllStrokes(strokes):
    point1, point2 = 0, 0
    for stroke1 in strokes:
        for stroke2 in strokes:
            if stroke1.all != stroke2.all:
                try:
                    point1_tmp, point2_tmp = getClosestPointsBetweenStrokes_Horizontal(stroke1, stroke2)
                except StrokesOverlappingException:
                    # print ("strokes overlapping go next iteration normally")
                    continue
                if not isStrokeBetween_Horizontal(point1_tmp, point2_tmp, strokes):
                    if abs(point1_tmp - point2_tmp) > abs(point1 - point2):
                        point1, point2 = point1_tmp, point2_tmp

    return point1, point2


def findFirstLeftStroke(strokes):
    firstLeftStroke = strokes[0]
    for stroke in strokes:
        if getHorizontalMin(stroke) < getHorizontalMin(firstLeftStroke):
            firstLeftStroke = stroke
    return firstLeftStroke


def distanceFromLeftBorderToLeftStroke(plt, strokes):
    firstLeftStroke = findFirstLeftStroke(strokes)
    left, right = plt.xlim()
    # print("leftBorder: ", left, "firstLeftStrokeMin", getVerticalMin(firstLeftStroke))
    return abs(getHorizontalMin(firstLeftStroke) - left)


def distanceFromLeftBorderToCropThere(plt, cropThere):
    left, right = plt.xlim()
    return abs(cropThere - left)



def getOrCreatePath(directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    pathToImage = os.path.join(directory, name)
    return pathToImage


def getPlot(coords, linewidth):
    fig, ax = plt.subplots(1, 1)
    strokes = split_strokes(cumsum(np.array(coords)))
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], 'k', linewidth=linewidth)
    ax.set_aspect('equal')
    plt.axis('off')
    return fig, strokes


def setUpStyles(style):
    style = None
    if style is not None:
        with open(os.path.join('generatorNetworks/RNN/data', 'styles.pkl'), 'rb') as file:
            styles = pickle.load(file)

        if style > len(styles[0]):
            raise ValueError('Requested style is not in style list')

        style = [styles[0][style], styles[1][style]]
    return style


def find_CropAndGapLines(strokes):
    gapLeft, gapRight = findWidestEmptyGapBetweenAllStrokes(strokes)
    distanceFromLeftBorderToFirstStroke = distanceFromLeftBorderToLeftStroke(plt, strokes)
    cropThere = gapLeft + distanceFromLeftBorderToFirstStroke
    return cropThere, gapLeft, gapRight


def savePlotWithLines(cropThere, gapLeft, gapRight, pathToImage):
    drawGapsAndCropLines(cropThere, gapLeft, gapRight)
    plt.savefig(fname = pathToImage, bbox_inches='tight', pad_inches=0)
    print("widest gap, boundary points imageName: ", pathToImage, gapLeft, gapRight)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def save8bitImageAsPng(image, pathToImage):
    image = image.convert('L')
    image.save(pathToImage + '.png', "png")


def getCroppedImage(cropThere, fig):
    left, right = plt.xlim()
    cropPlot(cropThere, left)
    croppedImage = fig2img(fig)
    plt.xlim(left, right)
    return croppedImage


def cropPlot(cropThere, left):
    plt.xlim(left, cropThere)


def drawGapsAndCropLines(cropThere, gapLeft, gapRight):
    plt.axvline(gapLeft)
    plt.axvline(gapRight)
    plt.axvline(cropThere)


def saveCroppedPlot(croppedImage, pathToImage, size):
    if(size == 'normal'):
        save8bitImageAsPng(croppedImage, pathToImage)
    elif(size == '28x28'):
        resizedCroppedImage = croppedImage.resize((28, 28), Image.LANCZOS)
        save8bitImageAsPng(resizedCroppedImage, pathToImage)


def blurAndSaveImage(image, pathToBlurredImage, blurRadius, size):
    image = image.filter(ImageFilter.GaussianBlur(radius=blurRadius))
    if(size == 'normal'):
        save8bitImageAsPng(image, pathToBlurredImage + '_blurred')
    elif(size == '28x28'):
        resizedImage = image.resize((28,28), Image.LANCZOS)
        save8bitImageAsPng(resizedImage, pathToBlurredImage + '_blurred')


def saveBlurredImage_FromPath(pathToImage, pathToBlurredImage, blurRadius):
    image = Image.open(pathToImage + '.png')
    image = image.filter(ImageFilter.GaussianBlur(radius=blurRadius))
    image.save(pathToBlurredImage + '_blurred.png', "png")


def probablyBadlyGenerated(gapLeft, gapRight):
    minPairSection = 0.3
    minGapSection = 0.19
    left, right = plt.xlim()
    plotWidth = right - left;
    if abs(gapLeft - left) < abs (minPairSection * plotWidth):
        return True
    if abs(gapRight - gapLeft) < abs (minGapSection * plotWidth):
        return True
    return False


def saveBadImage(cropThere, gapLeft, gapRight, name, textToGenerate):
    badGeneratedDirname = os.path.join(args.destPath, args.dirName + '_bad', textToGenerate[0:2])
    pathToImageWithLines = getOrCreatePath(badGeneratedDirname, name)
    savePlotWithLines(cropThere, gapLeft, gapRight, pathToImageWithLines)


def saveBackupImage(cropThere, gapLeft, gapRight, name, textToGenerate):
    oryginalBackupDirname = os.path.join(args.destPath, args.dirName + '_backup', textToGenerate[0:2])
    pathToBackupImage = getOrCreatePath(oryginalBackupDirname, name + 'oryginal')
    savePlotWithLines(cropThere, gapLeft, gapRight, pathToBackupImage)


def generate_And_Save_Images(imageNum, sess, translation, textToGenerate, linewidth, blurRadius, generateMode, size, bias, force, style, destPath, dirName):
    style = setUpStyles(style)
    phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, textToGenerate, translation, style, bias, force)
    fig, strokes = getPlot(coords, linewidth)
    cropThere, gapLeft, gapRight = find_CropAndGapLines(strokes)

    name = textToGenerate[0:2] + str(imageNum)
    if probablyBadlyGenerated(gapLeft, gapRight):
        # saveBadImage(cropThere, gapLeft, gapRight, name, textToGenerate)
        pass
    else:
        croppedImage = getCroppedImage(cropThere, fig)
        if(generateMode == 'all' or generateMode == 'onlyNormal'):
            croppedDirname = os.path.join(destPath, dirName, textToGenerate[0:2])
            pathToImage = getOrCreatePath(croppedDirname, name)
            saveCroppedPlot(croppedImage, pathToImage, size)
        if(generateMode == 'all' or generateMode == 'onlyBlurred'):
            blurredDirname = os.path.join(destPath, dirName, textToGenerate[0:2])
            pathToBlurredImage = getOrCreatePath(blurredDirname, name)
            blurAndSaveImage(croppedImage, pathToBlurredImage, blurRadius, size)

        # saveBackupImage(cropThere, gapLeft, gapRight, name, textToGenerate)
        # plt.show()
    plt.close(fig)

def generate(whatToGenerate, pairCombinationMode='allPairs', howMany=1, dirName='default', text='None', numOfGenerated=0, generateFromArg=False, generateMode='onlyBlurred', size='28x28', model_path=os.path.join('generatorNetworks/RNN/pretrained', 'model-29'), lineWidth=18, blurRadius=6, destPath='data/generatedPairs', bias=1., force=False, style=None, calledAsMainProgram=False):
    print(__file__ + str(datetime.datetime.now()))
    pathToData='generatorNetworks/RNN/data'
    if(calledAsMainProgram):
        pathToData='data'
    with open(os.path.join(pathToData, 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        if text.startswith('None') and pairCombinationMode.startswith('onePair'):
            print('\n\nFATAL please give text to generate when use onePair mode\n\n')
            return
        if text.startswith('None') and pairCombinationMode == 'manyPairsInText':
            print('\n\nFATAL please give text to generate when use manyPairsInText mode. Text format is \"ab bc cd\"\n\n')
            return

        allLetters = "abcdefghijklmnopqrstuvwxyz"
        linewidth = lineWidth
        blurRadius = blurRadius
        if bool(generateFromArg):
            for el in whatToGenerate:
                pairString, numToGenerateString = el[0], el[1]
                print('generate ' + pairString + str(numToGenerateString) + 'times')
                for i in range(int(numToGenerateString)):
                    textToGenerate = pairString + " " + allLetters[i % len(allLetters)]
                    generate_And_Save_Images(numOfGenerated + i, sess, translation, textToGenerate, linewidth, blurRadius, generateMode, size, bias, force, style, destPath, dirName)

        else:
            for i in range(howMany):
                print("iteration: ", i)
                if (pairCombinationMode == 'allPairs'):
                    for firstLetter, secondLetter in product(allLetters, repeat=2):
                        textToGenerate = firstLetter + secondLetter + " " + allLetters[i % len(allLetters)]
                        generate_And_Save_Images(numOfGenerated + i, sess, translation, textToGenerate, linewidth, blurRadius, generateMode, size, bias, force, style, destPath, dirName)
                elif (pairCombinationMode == 'manyPairsInText'):
                    charsPerPair = 3
                    for j in range(0, len(text), charsPerPair):
                        if(j + 2 < len(text)):
                            if not (text[j].isalpha() and text[j + 1].isalpha() and text[j + 2] == ' '):
                                print('\n\nFATAL please give well formated text to generate when use manyPairsInText mode (use --text=). Text format is \"ab bc cd\"\n\n')
                                return
                        textToGenerate = text[j] + text[j + 1] + " " + allLetters[i % len(allLetters)]
                        generate_And_Save_Images(i+numOfGenerated, sess, translation, textToGenerate, linewidth, blurRadius, generateMode, size, bias, force, style, destPath, dirName)
                elif (pairCombinationMode == 'onePair'):
                    textToGenerate2 = text + " " + allLetters[i % len(allLetters)]
                    generate_And_Save_Images(i, sess, translation, textToGenerate2, linewidth, blurRadius, generateMode, size, bias, force, style, destPath, dirName)
                elif (pairCombinationMode == 'multiPairs'):
                    lettersToCombine = "abcdef"
                    for firstLetter, secondLetter in combinations(lettersToCombine, 2):
                        textToGenerate = firstLetter + secondLetter + " " + allLetters[i % len(allLetters)]
                        generate_And_Save_Images(i, sess, translation, textToGenerate, linewidth, blurRadius, generateMode, size, bias, force, style, destPath, dirName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
    parser.add_argument('--text', dest='text', type=str, default=None)
    parser.add_argument('--style', dest='style', type=int, default=None)
    parser.add_argument('--bias', dest='bias', type=float, default=1.)
    parser.add_argument('--force', dest='force', action='store_true', default=False)
    # parser.add_argument('--animation', dest='animation', action='store_true', default=False)
    # parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
    # parser.add_argument('--save', dest='save', type=str, default=None)
    parser.add_argument('--howMany', dest='howMany', type=int, default=1)
    # parser.add_argument('--allPairs', dest='allPairs', type=bool, default=False)
    parser.add_argument('--pairMode', dest='pairCombinationMode', type=str,
                        default='allPairs')  # onePair, multiPairs, allPairs, manyPairsInText
    parser.add_argument('--lineWidth', dest='lineWidth', type=int, default=18)
    parser.add_argument('--blurRadius', dest='blurRadius', type=int, default=6)
    parser.add_argument('--generateMode', dest='generateMode', type=str,
                        default='onlyBlurred')  # onlyBlurred, onlyNormal, all
    parser.add_argument('--size', dest='size', type=str, default='28x28')  # normal, 28x28
    parser.add_argument('--destPath', dest='destPath', type=str, default='../../data/generatedPairs')
    parser.add_argument('--numOfGenerated', dest='numOfGenerated', type=int, default=0)
    parser.add_argument('--dirName', dest='dirName', type=str, default='default')
    parser.add_argument('--generateFromArg', dest='generateFromArg', type=bool, default=False)
    args = parser.parse_args()
    generate([], args.pairCombinationMode, args.howMany, args.dirName, args.text, args.numOfGenerated,
             args.generateFromArg, args.generateMode, args.size,
             args.model_path, args.lineWidth, args.blurRadius,
             args.destPath, args.bias, args.force, args.style, True)


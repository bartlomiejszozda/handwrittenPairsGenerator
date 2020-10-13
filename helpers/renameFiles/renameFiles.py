import argparse
import datetime
import os


class RenameFiles:
  def __init__(self, path, extension, namePostfix='', text=None, namePrefix=''):
    self.__text=text
    self.__path=path
    self.__namePrefix=namePrefix
    self.__namePostfix=namePostfix
    self.__extension=extension

  def rename(self):
    print(__file__ + str(datetime.datetime.now()))
    charsPerPair = 3

    if self.__text is None:
      self.renameAllDirs()
    else:
      self.renameDirsFromText(charsPerPair)


  def renameAllDirs(self):
    for dirName in os.listdir(self.__path):
      if os.path.isdir(os.path.join(self.__path, dirName)):
        self.renameAllFiles(self.__path, dirName)
      print(str(dirName) + ' renamed')


  def renameDirsFromText(self, charsPerPair):
    for j in range(0, len(self.__text), charsPerPair):
      if (j + 2 < len(self.__text)):
        if not (self.__text[j].isalpha() and self.__text[j + 1].isalpha() and self.__text[j + 2] == ' '):
          print(
            '\n\nFATAL please give well formated text to generate when use manyPairsInText mode (use --text=). Text format is \"ab bc cd\"\n\n')
          return
      dirName = self.__text[j] + self.__text[j + 1]
      if os.path.isdir(os.path.join(self.__path, dirName)):
        self.renameAllFiles(self.__path, dirName)
      else:
        print('path not exist')


  # Function to rename multiple files
  def renameAllFiles(self, path, dirName):
    temporaryPrefix = 'toRename'
    for filename in os.listdir(os.path.join(path, dirName)):
      if not filename.startswith(temporaryPrefix):
        self.temporaryRenameToAvoidNameMismatch(dirName, filename, path, temporaryPrefix)
    i = 0
    for filename in os.listdir(os.path.join(path, dirName)):
      if (self.__namePrefix == ''):
        # dirName = os.path.basename(os.path.normpath(path))
        namePrefix = dirName
        dst = namePrefix + str(i) + '_' + self.__namePostfix + self.__extension
      else:
        dst = self.__namePrefix + str(i) + '_' + self.__namePostfix + self.__extension
      src = os.path.join(path, dirName, filename)
      dst = os.path.join(path, dirName, dst)

      os.rename(src, dst)
      i += 1


  def temporaryRenameToAvoidNameMismatch(self, dirName, filename, path, temporaryPrefix):
    src = os.path.join(path, dirName, filename)
    dst = os.path.join(path, dirName, temporaryPrefix + filename)
    os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--text', dest='text', type=str, default=None)
  parser.add_argument('--path', type=str, help='path where to rename files')
  parser.add_argument('--namePrefix', type=str, default='',
                      help='name prefix. Name contain 3 parts: namePrefixNum_namePostfixExtension')
  parser.add_argument('--namePostfix', type=str, default='',
                      help='name postfix. Name contain 3 parts: namePrefixNum_namePostfixExtension')
  parser.add_argument('--extension', type=str, help='file extension')
  args = parser.parse_args()

  renameFiles = RenameFiles(args.path, args.extension, args.namePostfix, args.text, args.namePrefix)
  renameFiles.rename()

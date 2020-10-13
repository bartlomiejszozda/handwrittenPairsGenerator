from os.path import dirname, abspath, join
import subprocess

def lastCommitToFile(logFilePath):
    gitLogSubprocess = subprocess.Popen("git log -n 1 HEAD", shell=True, stdout=subprocess.PIPE)
    subprocessOut = gitLogSubprocess.stdout.read()

    print(subprocessOut.decode('unicode_escape'))
    with open(logFilePath, 'a') as logFile:
        logFile.write(subprocessOut.decode('unicode_escape') + \
        "-----------------------------------" + "\n")

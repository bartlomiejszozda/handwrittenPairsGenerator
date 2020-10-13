from os.path import dirname, abspath, join
import sys

THIS_DIR = dirname(__file__)
REPO_DIR = abspath(join(THIS_DIR, '..'))
print("repo dir:"+REPO_DIR)
sys.path.insert(0, REPO_DIR)
import test.performance.longGenerationPerformanceTest as performanceTest
import helpers.writeLastCommitToFile.writeLastCommitToFile as lastCommitWriter

performanceTestTime = performanceTest.test()
with open(REPO_DIR + "/showcase/performanceTests/longGenerationPerformance_EachCall.txt", 'a') as performanceFile:
    performanceFile.write("time: " + str(performanceTestTime) + "\n")

logFilePath = REPO_DIR + "/showcase/performanceTests/longGenerationPerformance_EachCall.txt"
lastCommitWriter.lastCommitToFile(logFilePath)


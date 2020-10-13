
from matplotlib import pyplot as PLT
import numpy as NP
import datetime

import argparse

print(__file__ + str(datetime.datetime.now()))

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to file from where we will generate histogram')
args = parser.parse_args()

with open(args.path) as f:
  descriptions, values = NP.loadtxt(f, delimiter=",",dtype=NP.dtype("U2,int"), comments="#", skiprows=1, usecols=(0,1), unpack=True)
  # values = NP.loadtxt(f, delimiter=" ", dtype=int, comments="#", skiprows=1, usecols=1)

# descriptions= ['G1', 'G2', 'G3', 'G4', 'G5']
# values = [20, 34, 30, 35, 27]


print(descriptions)
print(values)
width = 0.01
# v_hist = NP.ravel(values)  # 'flatten' v
fig, ax = PLT.subplots()
# fig = PLT.figure()
ax.bar(descriptions, values, width)
PLT.xticks(rotation=45)
PLT.ylabel('Ilo narodzin')
PLT.xlabel('Pe')
# ax1 = fig.add_subplot(111)
# n, bins, patches = ax1.hist(v_hist, bins=50, normed=1, facecolor='green')
PLT.show()
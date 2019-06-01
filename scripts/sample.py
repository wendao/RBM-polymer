import numpy as np
from numpy.random import choice
import sys

confs = []
weights = []
kT = float(sys.argv[2])
n = int(sys.argv[3])

lines = open(sys.argv[1], 'r').readlines()
for l in lines:
    es = l.split()
    confs.append(es[0])
    weights.append(-float(es[1])/kT)

weights = np.array(weights)
weights = np.exp(weights)
sum_w = np.sum(weights)
weights /= sum_w

if n>0:
    for i in xrange(n):
        print(choice(confs, p=weights, replace=True))
else:
    print len(confs)
    for i, c in enumerate(confs):
        print c, weights[i]

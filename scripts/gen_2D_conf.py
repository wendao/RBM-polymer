import sys
import random

L = int(sys.argv[1])
N = int(sys.argv[2])

for n in xrange(N):
    for l in xrange(L-1):
       x = random.randint(0, 1)
       y = random.randint(0, 1)
       print str(x)+str(y),
    print

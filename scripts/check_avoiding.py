import sys
import numpy as np

#xyz: +x:0, +y:1, -x:2, -y:3
#int: left:0, forward:1, right:2, backward:3

def real_conf_from_int(conf):
    L = len(conf)
    x = np.zeros([L+2, 2])
    x[0,0] = 0
    x[0,1] = 0
    x[1,0] = 1
    x[1,1] = 0
    for i in xrange(L):
        if conf[i] == "1":
            x[i+2,0] = 2*x[i+1,0]-x[i,0]
            x[i+2,1] = 2*x[i+1,1]-x[i,1]
        elif conf[i] == "3":
            x[i+2,0] = x[i,0]
            x[i+2,1] = x[i,1]
        else:
            dx1 = x[i+1,0] - x[i,0]
            dy1 = x[i+1,1] - x[i,1]
            if conf[i] == "2":
                if dx1 == 0:
                    dy = 0
                    if dy1>0:
                        dx = -1
                    else:
                        dx = 1
                elif dy1 == 0:
                    dx = 0
                    if dx1>0:
                        dy = 1
                    else:
                        dy = -1
                else:
                    print("error!")
            elif conf[i] == "0":
                if dx1 == 0:
                    dy = 0
                    if dy1>0:
                        dx = 1
                    else:
                        dx = -1
                elif dy1 == 0:
                    dx = 0
                    if dx1>0:
                        dy = -1
                    else:
                        dy = 1
                else:
                    print("error!")
            x[i+2,0] = x[i+1,0] + dx
            x[i+2,1] = x[i+1,1] + dy
    return x

def real_conf_from_xyz(conf):
    L = len(conf)
    x = np.zeros([L+1, 2])
    x[0,0] = 0
    x[0,1] = 0
    for i in xrange(L):
        if conf[i] == "0":
            x[i+1,0] = x[i,0] + 1
            x[i+1,1] = x[i,1]
        elif conf[i] == "1":
            x[i+1,0] = x[i,0]
            x[i+1,1] = x[i,1] + 1
        elif conf[i] == "2":
            x[i+1,0] = x[i,0] - 1
            x[i+1,1] = x[i,1]
        elif conf[i] == "3":
            x[i+1,0] = x[i,0]
            x[i+1,1] = x[i,1] - 1
    return x

def count_clash(conf):
    n, d = conf.shape
    c = 0
    for i in xrange(n):
        for j in xrange(i):
            if (conf[i,:] == conf[j,:]).all(): c+=1
    return c

#cmd [file] [skip] [xyz/int]
lines = open(sys.argv[1], 'r').readlines()
for l in lines[int(sys.argv[2]):]:
    if sys.argv[3] == "int":
        conf = real_conf_from_int(l.strip())
    else:
        conf = real_conf_from_xyz(l.strip())
    n = len(l.strip())+2-len(np.unique(conf, axis=0))
    print l.strip(), count_clash(conf)

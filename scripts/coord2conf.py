import sys

#   11
# 00  10
#   01

x0 = 0
y0 = 0
n0 = 0
lines = open(sys.argv[1], 'r').readlines()
for l in lines:
    elems = l.split()
    if len(elems) < 3: continue
    n = int(elems[2])
    if n==1:
        x0 = float(elems[0])
        y0 = float(elems[1])
        #end of conf
        if n0 == 0:
            n0 = 1
        else:
            print
    else:
        x = float(elems[0])
        y = float(elems[1])
        dx = int(x-x0)
        dy = int(y-y0)
        if dx == -1 and dy == 0:
            sys.stdout.write("0")
        elif dx == 0 and dy == -1:
            sys.stdout.write("1")
        elif dx == 1 and dy == 0:
            sys.stdout.write("2")
        elif dx == 0 and dy == 1:
            sys.stdout.write("3")
        else:
            print x, y
            print "error!"
        x0 = x
        y0 = y

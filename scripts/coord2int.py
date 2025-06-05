import sys

#forward 0
#left 1
#backward 2
#right 3

## CHANGE!! left->0, forward->1, right->2, back->3

x0 = 0
y0 = 0
n0 = 0
x1 = 0
y1 = 0
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
    elif n==2:
        x1 = float(elems[0])
        y1 = float(elems[1])
        sys.stdout.write("0") #the first step is forward
    else:
        x = float(elems[0])
        y = float(elems[1])
        dx2 = int(x-x1)
        dy2 = int(y-y1)
        dx1 = int(x1-x0)
        dy1 = int(y1-y0)
        c = dx1*dx2 + dy1*dy2
        s = dx1*dy2 - dx2*dy1
        if c == 1 and s == 0:
            sys.stdout.write("1")
        elif c == 0 and s == 1:
            sys.stdout.write("2")
        elif c == -1 and s == 0:
            sys.stdout.write("3")
        elif c == 0 and s == -1:
            sys.stdout.write("0")
        else:
            print x, y
            print "error!"
        x0 = x1
        y0 = y1
        x1 = x
        y1 = y

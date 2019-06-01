import sys
import numpy as np
import math

def real_conf_from_int(conf):
    L = len(conf)
    x = np.zeros([L+2, 2])
    x[0,0] = 0
    x[0,1] = 0
    x[1,0] = 1
    x[1,1] = 0
    for i in range(L):
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
    coor = np.reshape(x, (L+2)*2)
    return coor

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
    coor = np.reshape(x, (L+1)*2)
    return coor

#########Calculate the mmean square radius of gyration of chains according to its definition
def cal_Rg2(coordinates):
	coor = coordinates
	num_xy = coor.shape[0]
	num_monomers = num_xy//2
	R_cmx = 0.0
	R_cmy = 0.0

	for j in range(0, num_xy, 2):
		R_cmx += coor[j]
		R_cmy += coor[j+1]
	R_cmx = R_cmx/num_monomers
	R_cmy = R_cmy/num_monomers

	Rg_x = 0.0
	Rg_y = 0.0
	Rg2 = 0.0

	for j in range(0, num_xy, 2):
		Rg_x += (coor[j] - R_cmx)**2 
		Rg_y += (coor[j+1] - R_cmy)**2

	Rg2 = (Rg_x+Rg_y)/num_monomers
	print(Rg2)
	return

##### main #########

fn_gen = sys.argv[1]
shift = int(sys.argv[2])

lines = open(fn_gen, 'r').readlines()

if sys.argv[3] == "int":
    for l in lines[shift:]:
        s = l.strip()
        cal_Rg2(real_conf_from_int(s))
else:
    for l in lines[shift:]:
        s = l.strip()
        cal_Rg2(real_conf_from_xyz(s))


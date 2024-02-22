import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

data_1D = np.load(sys.argv[1])["kernel"]
dlen = data_1D.shape[0]

accum2D = np.zeros((dlen,dlen))

for idx_1D in range(dlen):
    radius = idx_1D-dlen/2
    for angle_2D_idx in range(10*360):
        angle_2D = 2*math.pi*angle_2D_idx/(10*360)
        x = int(dlen/2+radius*math.cos(angle_2D))
        x = max(0,min(x,dlen-1))
        y = int(dlen/2+radius*math.sin(angle_2D))
        y = max(0,min(y,dlen-1))
        #print(str(x)+":"+str(y))
        accum2D[x,y] += data_1D[idx_1D]

np.save("kernel_2D",accum2D)
#plt.imshow(accum2D, interpolation='none', cmap='gray')
#plt.show()

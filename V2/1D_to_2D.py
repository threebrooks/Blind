import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

data = np.load(sys.argv[1]) 
kernel_1D = data["kernel"]
blur_size = data["blur_size"]
dlen = kernel_1D.shape[0]

accum2D = np.zeros((dlen,dlen))
accum2D_count = np.zeros((dlen,dlen))

for radius in np.arange(-dlen/2, dlen/2, 1/100.0):
    #print(radius)
    for angle_2D in np.arange(-math.pi,math.pi,2*math.pi/1E5):
        x = int((dlen-1)/2+radius*math.cos(angle_2D)+0.5)
        x = max(0,min(x,dlen-1))
        y = int((dlen-1)/2+radius*math.sin(angle_2D)+0.5)
        y = max(0,min(y,dlen-1))
        accum2D[y,x] += kernel_1D[int((dlen-1)/2+radius+0.5)]
        accum2D_count[y,x] += 1

accum2D /= accum2D_count+1E-10
#accum2D /= np.linalg.norm(accum2D)

plt.plot(accum2D[int((dlen-1)/2),:])
plt.plot(kernel_1D)
plt.show()

np.savez("kernel_2D_"+str(blur_size)+"px.npz",kernel=accum2D, blur_size=blur_size)
plt.imshow(accum2D, interpolation='none', cmap='gray')
plt.show()

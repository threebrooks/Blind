import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

bs = 3
ws = 100

plt.ion()

def obj_func(data):
    dlen = len(data)
    sq_data = np.square(data)
    total = np.sum(sq_data)
    #inner = np.sum(sq_data[int(dlen/2-bs/16):int(dlen/2+bs/16)])
    inner = sq_data[int(dlen/2)] # np.sum(sq_data[int(dlen/2-bs/16):int(dlen/2+bs/16)])
    return (inner/total)

blurred = np.zeros(ws)
blurred[int(ws/2-bs/2):int(ws/2+bs/2)] = 1.0/bs

highest_kernel = np.ones(ws)
highest_obj_score = -1E10
highest_idx = 0
null_idx = 0
good_pos = None
pos_add = 0.1
while(True):
    null_idx += 1
    new_kernel = copy.deepcopy(highest_kernel)

    if (good_pos is None):
        good_pos = int(random.random()*ws) 
        pos_add = (random.random()-0.5)/5.0
    new_kernel[good_pos] += pos_add
    new_kernel[len(new_kernel)-1-good_pos] += pos_add

    new_kernel -= np.mean(new_kernel)

    conv = np.convolve(blurred, new_kernel)#/ws
    new_obj_score = obj_func(conv)

    if (new_obj_score > highest_obj_score):
        highest_kernel = new_kernel
        highest_obj_score = new_obj_score
    
        highest_idx += 1
        if (highest_idx % 500 == 0):
            print(str(null_idx)+" "+str(highest_obj_score))
            plt.clf()
            #plt.plot(highest_kernel)
            plt.plot(conv)
            #plt.plot(blurred)
            plt.draw()
            plt.pause(0.001)
            null_idx = 0
            np.savez("kernel.npz", kernel=highest_kernel, blurred=blurred, conv=conv)
    else:
        good_pos = None
            

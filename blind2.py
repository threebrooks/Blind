import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

bs = 10
ws = 5*bs

#plt.ion()

def obj_func(conv, orig_img):
    return np.sum(np.abs(conv-orig_img))

#plt.plot(orig_img)
#plt.plot(blur_kernel)
#plt.plot(blurred_img)
#plt.show()

blur_kernel = np.zeros(ws)
blur_kernel[int(ws/2-bs/2):int(ws/2+bs/2)] = 1.0/bs

lowest_kernel = np.ones(ws)
lowest_obj_score = 1E10
lowest_idx = 0
null_idx = 0
good_pos = None
pos_add = 0.1
while(True):
    null_idx += 1

    orig_img = np.random.random(ws)
    blurred_img = np.convolve(blur_kernel,orig_img, mode='same')
    
    new_kernel = copy.deepcopy(lowest_kernel)

    #if (good_pos is None):
    #    good_pos = int(random.random()*ws) 
    #    pos_add = (random.random()-0.5)/5.0
    #new_kernel[good_pos] += pos_add
    #new_kernel[len(new_kernel)-1-good_pos] += pos_add
    new_kernel += (np.random.random(len(new_kernel))-0.5)

    #new_kernel -= np.mean(new_kernel)

    conv = np.convolve(blurred_img, new_kernel, mode='same')#/ws
    new_obj_score = obj_func(conv, orig_img)

    if (new_obj_score < lowest_obj_score):
        lowest_kernel = new_kernel
        lowest_obj_score = new_obj_score
    
        lowest_idx += 1
        if (lowest_idx % 2 == 0):
            print(str(null_idx)+" "+str(lowest_obj_score))
            plt.clf()
            plt.plot(lowest_kernel)
            plt.draw()
            plt.pause(0.001)
            null_idx = 0
            np.savez("kernel.npz", kernel=lowest_kernel, blurred=blurred_img, orig=orig_img, conv=conv)
    else:
        good_pos = None
            

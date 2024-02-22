import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys
import cv2
import scipy

def get_disk_kernel(n, r):
    a, b = n/2, n/2
    
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    
    array = np.zeros((n, n))
    array[mask] = 1.0
    return array

def get_disk_kernel2(dlen, r):
    accum2D = np.zeros((dlen,dlen))
    accum2D_count = np.zeros((dlen,dlen))
    for radius in np.arange(-r, r, 1/100.0):
        #print(radius)
        for angle_2D in np.arange(-math.pi,math.pi,2*math.pi/1E3):
            x = int((dlen-1)/2+radius*math.cos(angle_2D)+0.5)
            x = max(0,min(x,dlen-1))
            y = int((dlen-1)/2+radius*math.sin(angle_2D)+0.5)
            y = max(0,min(y,dlen-1))
            accum2D[y,x] += 1
            accum2D_count[y,x] += 1
    accum2D /= accum2D_count+1E-10
    return accum2D
    

deblur_kernel = np.load(sys.argv[1])["kernel"]
bs = np.load(sys.argv[1])["blur_size"]
print(bs)

blur_disk = get_disk_kernel2(deblur_kernel.shape[0], bs/2)
#blur_disk = np.zeros(deblur_kernel.shape)
#blur_disk[int(blur_disk.shape[0]/2)-1:int(blur_disk.shape[0]/2)+1,int(blur_disk.shape[1]/2)-1:int(blur_disk.shape[1]/2)+1] = 1.0
#plt.plot(blur_disk[int(blur_disk.shape[0]/2),:], label='blur_disk')
plt.imshow(blur_disk, interpolation='none', cmap='gray')
plt.show()

image = cv2.imread(sys.argv[2])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.plot(image[int(image.shape[0]/2),int(image.shape[1]/2-150):int(image.shape[1]/2+150)], label='orig')

blurred_image = scipy.signal.convolve(image,blur_disk,method='fft',mode='same')
blurred_image = np.maximum(0, blurred_image)

plt.plot(blurred_image[int(blurred_image.shape[0]/2),int(blurred_image.shape[1]/2-150):int(blurred_image.shape[1]/2+150)], label='blurred')
plt.legend()
plt.show()
plt.clf()

print("blurred_image diff:   "+str(np.sum(np.abs(blurred_image-image))))

plt.imshow(blurred_image, interpolation='none', cmap='gray')
plt.show()

deblurred_image = scipy.signal.convolve(blurred_image,deblur_kernel,method='fft',mode='same')
deblurred_image = np.maximum(0, deblurred_image)

print("deblurred_image diff: "+str(np.sum(np.abs(deblurred_image-image))))

plt.imshow(deblurred_image, interpolation='none', cmap='gray')
plt.show()


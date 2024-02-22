import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys
import cv2
import scipy

kernel = np.load(sys.argv[1])
image = cv2.imread(sys.argv[2])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

conv_image = scipy.signal.convolve(kernel,image,method='fft')
conv_image = np.maximum(0, conv_image)

plt.imshow(conv_image, interpolation='none', cmap='gray')
plt.show()

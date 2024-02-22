import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

bs = 3
ws = 100

kernel = np.load(sys.argv[1])["kernel"]

blurred = np.zeros(ws)
blurred[int(ws/2-bs/2):int(ws/2+bs/2)] = 1.0/bs

test_image = np.random.random(ws)
blurred_test_image = np.convolve(test_image, blurred)

deblurred_test_image = np.convolve(blurred_test_image, kernel)
plt.plot(test_image)
plt.plot(deblurred_test_image)
#plt.plot(conv)
plt.show()

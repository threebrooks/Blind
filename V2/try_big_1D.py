import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

bs = 3
ws = 100

kernel_data = np.load(sys.argv[1])
kernel = kernel_data["kernel"]
bs = kernel_data["blur_size"]

blurred = np.zeros(ws)
blurred[int(ws/2-bs/2):int(ws/2+bs/2)] = 1.0/bs

test_image = np.random.random(ws)
blurred_test_image = np.convolve(test_image, blurred, mode='same')
print("blurred_test_image diff: "+str(np.sum(np.abs(blurred_test_image-test_image))))

deblurred_test_image = np.convolve(blurred_test_image, kernel, mode='same')
print("Deblurred_test_image diff: "+str(np.sum(np.abs(deblurred_test_image-test_image))))
plt.plot(test_image,label='test_image')
plt.plot(blurred_test_image,label='blurred_test_image')
plt.plot(deblurred_test_image,label='deblurred_test_image')
plt.legend()
#plt.plot(conv)
plt.show()


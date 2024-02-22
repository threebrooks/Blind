import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import sys

data = np.load(sys.argv[1])
for d in data:
    if (d in sys.argv):
        plt.plot(data[d])
plt.show()


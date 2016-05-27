import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import peakdetect

xs = np.arange(0, np.pi, 0.05)

data = np.sin(xs)

peakind = signal.find_peaks_cwt(data, np.arange(1,10))

plt.plot(xs, data)
plt.scatter(xs[peakind],data[peakind])

print(peakind, xs[peakind], data[peakind])
plt.show()


ys = np.arange(0,8*np.pi,0.05)
dataY = np.sin(ys)

peaks, lows = peakdetect.peakdet(dataY, 0.01, ys)

plt.plot(ys,dataY)
plt.scatter(peaks[:,0], peaks[:,1])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack as spfft


#Number of sample points
N=600
#sample spacing, even
T = 1.0/800.0
x = np.linspace(0.0,N*T,N)

# values based on spacing
y = np.sin(50*2.0*np.pi*x) + 0.5*np.sin(80.0*2.0*np.pi*x) + np.random.random()

#Fourier Transforms, would suggest using numpy, as does same,
# but less to type in
yf = spfft.fft(y)
uf = np.fft.fft(y)

#Set values for plotting
xf = np.linspace(0.0,1.0/(2.0*T), N/2)

plt.plot(x,y)
plt.show()


plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
plt.grid()
plt.show()
plt.plot(xf, 2.0/N * np.abs(uf[0:N/2]))

plt.grid()
plt.show()

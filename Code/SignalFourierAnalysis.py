# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:57:40 2016

@author: silver
"""

import numpy as np
import matplotlib.pyplot as plt
import peakdetect as pd
import os

file = '2014_02_24-run01Data.txt'
minSum = 0.9


def signalCrop(time,sumSignal,LRSignal,TBSignal, minSum):
    '''
    Function to crop signal based on relative strength of sum signal
    compared to maximum. So minSum=0.8 would return all data at which
    the sum signal is minSum fraction  or higher than the maximum of the sum signal
    '''

    #define new arrays
    newtime = []
    newSum = []
    newLR = []
    newTB = []

    # crops signals
    for i in range(len(sumSignal)):
        if sumSignal[i] >=minSum*max(sumSignal):
            newtime.append(time[i])
            newSum.append(sumSignal[i])
            newLR.append(LRSignal[i])
            newTB.append(TBSignal[i])
    return newtime, newSum, newLR, newTB

#Brings in data from datafile in the /Data/ folder
script_dir = os.path.dirname(__file__)
rel_path = 'Data/'+file
abs_file_path = os.path.join(script_dir, rel_path)
data = np.genfromtxt(abs_file_path, delimiter='\t')

#Separates out values from larger array
time = data[:,0]
sumSignal = data[:,1]
LR_Signal = data[:,2]
TB_Signal = data[:,3]

#crops signal based on decimal fraction strength of sumSignal, if lower than specified, cropped out
time, sumSignal, LR_Signal, TB_Signal = signalCrop(time,sumSignal,LR_Signal,TB_Signal, minSum)

#removes mean, zeroes
sumSignal -= np.mean(sumSignal)
LR_Signal -= np.mean(LR_Signal)
TB_Signal -= np.mean(TB_Signal)

# does the fast fourier transform on each of the arrays and sets
#them up to be plotted
sumSignalF = np.fft.fft(sumSignal)
sumSignalF = np.abs(sumSignalF[:len(time)/2])/max(sumSignalF)

LR_SignalF = np.fft.fft(LR_Signal)
LR_SignalF = np.abs(LR_SignalF[:len(time)/2])/max(LR_SignalF)

TB_SignalF = np.fft.fft(TB_Signal)
TB_SignalF = np.abs(TB_SignalF[:len(time)/2])/max(TB_SignalF)

#find the max value that can be plotted out to
deltaT = []
for i in range(len(time)-1):
    deltaT.append(time[i+1]-time[i])

avgDeltaT = np.mean(deltaT)

# makes an array for frequencies
freq = np.linspace(0.0, 1.0/(2.0*avgDeltaT),len(time)/2)

#does peakfinding on each of them, still in progress
sumPeaks, sumLows = pd.peakdet(sumSignalF,0.5,freq)
LRPeaks, LRLows = pd.peakdet(LR_SignalF, 0.5, freq)
TBPeaks, TBLows = pd.peakdet(TB_SignalF, 0.5, freq)
print(sumPeaks)


# plots the original signals
plt.plot(time,sumSignal)
plt.show()

plt.plot(time,LR_Signal)
plt.show()

plt.plot(time,TB_Signal)
plt.show()



# plots  fourier transforms
plt.plot(freq,sumSignalF)
plt.scatter(np.real(sumPeaks[:,0]), sumPeaks[:,1])
plt.grid()
#plt.ylim(0,max(sumPeaks))
#plt.xlim(0,0.2)
plt.show()

plt.plot(freq,LR_SignalF)
plt.scatter(np.real(LRPeaks[:,0]), LRPeaks[:,1])
plt.grid()
#plt.ylim(0,max(LRPeaks))
plt.show()

plt.plot(freq,TB_SignalF)
plt.scatter(np.real(TBPeaks[:,0]),TBPeaks[:,1])
plt.grid()
#plt.ylim(0,max(TBPeaks)/10)
#plt.xlim(0,8)
plt.show()

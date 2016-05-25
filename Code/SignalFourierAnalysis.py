# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:57:40 2016

@author: silver
"""

import numpy as np
import matplotlib.pyplot as plt
import peakutils

def SignalAnalysis(file):
    data = np.genfromtxt('Data/'+file, delimiter='\t')

    time = data[:,0]
    sumSignal = data[:,1]
    LR_Signal = data[:,2]
    TB_Signal = data[:,3]
    
    sumSignal -= np.mean(sumSignal)
    LR_Signal -= np.mean(LR_Signal)
    TB_Signal -= np.mean(TB_Signal)

    sumSignalF = np.fft.fft(sumSignal)
    sumSignalF = np.abs(sumSignalF[:len(time)/2])/max(sumSignalF)
    
    LR_SignalF = np.fft.fft(LR_Signal)
    LR_SignalF = np.abs(LR_SignalF[:len(time)/2])/max(LR_SignalF)

    TB_SignalF = np.fft.fft(TB_Signal)
    TB_SignalF = np.abs(TB_SignalF[:len(time)/2])/max(TB_SignalF)

    sumPeaks = peakutils.indexes(sumSignalF)
    LRPeaks = peakutils.indexes(LR_SignalF)
    TBPeaks = peakutils.indexes(TB_SignalF)
    
    deltaT = []
    for i in range(len(time)-1):
        deltaT.append(time[i+1]-time[i])

    avgDeltaT = np.mean(deltaT)

    plt.plot(time,sumSignal)
    plt.show()

    plt.plot(time,LR_Signal)
    plt.show()

    plt.plot(time,TB_Signal)
    plt.show()

    timeF = np.linspace(0.0, 1.0/(2.0*avgDeltaT),len(time)/2)

    plt.plot(timeF,sumSignalF)
    plt.grid()
    plt.ylim(0,max(sumPeaks))
    plt.xlim(0,0.2)
    plt.show()

    plt.plot(timeF,LR_SignalF)
    plt.grid()
    plt.ylim(0,max(LRPeaks))
    plt.show()

    plt.plot(timeF,TB_SignalF)
    plt.grid()
    plt.ylim(0,max(TBPeaks))
#    plt.xlim(0,8)
    plt.show()
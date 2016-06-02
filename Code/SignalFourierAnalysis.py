# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:57:40 2016

@author: silver
"""

import numpy as np
import matplotlib.pyplot as plt
import peakdetect as pd
import os
import re


def FFTSignalAnalysis(file, minSum=1.0, minSumAbs = True, focus = None):


    fileNameRegex = re.compile(r'(\w|-)+')
    fileName = fileNameRegex.search(file).group()

    #Brings in data from datafile in the /Data/ folder
    script_dir = os.path.dirname(__file__)
    rel_path = 'Data/'+file
    abs_file_path = os.path.join(script_dir, rel_path)
    data = np.genfromtxt(abs_file_path, delimiter='\t')



    #Separates out values from larger array
    oldTime = data[:,0]
    oldSumSignal = data[:,1]
    oldLR_Signal = data[:,2]
    oldTB_Signal = data[:,3]


    if focus != None:
        oldTime, oldSumSignal, oldLR_Signal, oldTB_Signal = signalCropTime(\
        oldTime, oldSumSignal, oldLR_Signal, oldTB_Signal, focus)

    #crops signal based on decimal fraction strength of sumSignal, if lower than specified, cropped out
    time, sumSignal, LR_Signal, TB_Signal = signalCropSum(oldTime,oldSumSignal,oldLR_Signal,oldTB_Signal, minSum, minSumAbs = minSumAbs)

    #removes mean, zeroes
    sumSignal -= np.mean(sumSignal)
    LR_Signal -= np.mean(LR_Signal)
    TB_Signal -= np.mean(TB_Signal)

    # does the fast fourier transform on each of the arrays and sets
    #them up to be plotted
    sumSignalF = FourierTransform(sumSignal)
    LR_SignalF = FourierTransform(LR_Signal)
    TB_SignalF = FourierTransform(TB_Signal)

    #find the max value that can be plotted out to
    deltaT = []
    for i in range(len(time)-1):
        deltaT.append(time[i+1]-time[i])

    avgDeltaT = np.mean(deltaT)

    # makes an array for frequencies
    freq = np.linspace(0.0, 1.0/(2.0*avgDeltaT),len(time)/2)

    #does peakfinding on each of them, still in progress
    sumPeaks = PeakDetect(sumSignalF,freq)
    LRPeaks = PeakDetect(LR_SignalF, freq)
    TBPeaks = PeakDetect(TB_SignalF, freq)

    oldData = [oldTime, oldSumSignal, oldLR_Signal, oldTB_Signal]
    croppedData = [time, sumSignal, LR_Signal, TB_Signal]
    transformedData = [freq, sumSignalF, LR_SignalF, TB_SignalF]
    peakData = [sumPeaks, LRPeaks, TBPeaks]

    graphSignals(oldData, croppedData, transformedData, peakData, fileName, oldTime, script_dir)





def signalCropSum(time,sumSignal,LRSignal,TBSignal, minSum=1.0, minSumAbs=True):
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
        if minSumAbs ==True:
            if sumSignal[i] >=minSum:
                newtime.append(time[i])
                newSum.append(sumSignal[i])
                newLR.append(LRSignal[i])
                newTB.append(TBSignal[i])
        elif minSumAbs ==False:
            if sumSignal[i] >=minSum*max(sumSignal):
                newtime.append(time[i])
                newSum.append(sumSignal[i])
                newLR.append(LRSignal[i])
                newTB.append(TBSignal[i])
    return newtime, newSum, newLR, newTB

def signalCropTime(time, sumSignal, LRSignal, TBSignal, timeChunk):

    newtime = []
    newSum = []
    newLR = []
    newTB = []

    for i in range(len(time)):
        if time[i] >= timeChunk[0] and time[i] <= timeChunk[1]:
            newtime.append(time[i])
            newSum.append(sumSignal[i])
            newLR.append(LRSignal[i])
            newTB.append(TBSignal[i])
    return newtime, newSum, newLR, newTB


def graphSignals(old, cropped, transformed, peaks, fileName, time, script_dir):


    graphLabels = ['Sum Signal', 'Left-Right Signal', 'Top-Bottom Signal']

    i = 1
    plt.figure(figsize=(20,15))
    #plots the original signals
    for j in range(1,4):

        plt.subplot(3,3,i)
        plt.plot(old[0], old[j])
        plt.title('Original '+graphLabels[j-1], fontsize=16)
        plt.ylabel('Volts (V)',fontsize=12)
        plt.xlabel('Time (s)',fontsize=12)
        plt.grid()
        i +=1

    for j in range(1,4):
        #Plots the cropped signals
        plt.subplot(3,3,i)
        plt.title('Cropped '+graphLabels[j-1], fontsize=16)
        plt.ylabel('Volts (V)',fontsize=12)
        plt.xlabel('Time (s)',fontsize=12)
        plt.plot(cropped[0],cropped[j])
        plt.grid()
        i+=1

    for j in range(1,4):
        # plots  fourier transforms
        plt.subplot(3,3,i)
        plt.plot(transformed[0],transformed[j])
        PeaksLabel = 'Peaks at'
        try:
            plt.scatter((peaks[j-1][:,0]), peaks[j-1][:,1], label='Peaks')
            for k in range(len(peaks[j-1][:,0])):
                PeaksLabel += '\n %0.4f Hz,'%(peaks[j-1][k,0])
        except:
            PeaksLabel += ' nowhere'
        plt.plot(0,0, color='w', label=PeaksLabel)
        plt.title('Fourier Transform of Cropped '+graphLabels[j-1], fontsize=16)
        plt.ylabel('Volts per Hertz (V/Hz)',fontsize=12)
        plt.xlabel('Frequency (Hz)',fontsize=12)
        plt.legend(bbox_to_anchor=(1,1), fontsize = 10)
        plt.grid()
        i+=1

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    new_rel_path = 'Data/%s-%0.2fs-%0.2fs.png'%(fileName, time[0],time[-1])
    new_abs_file_path = os.path.join(script_dir, new_rel_path)

    plt.savefig(new_abs_file_path, dpi=300)
    plt.show()


def FourierTransform(data):
    transformData = np.fft.fft(data)
    transformData = 2.0/len(data) * np.abs(transformData[:len(data)/2])
    return transformData

def PeakDetect(data, domain):
    Peaks, Lows = pd.peakdet(data,(max(data)/15),domain)
    return Peaks




FFTSignalAnalysis('TestFile.txt', focus = None)

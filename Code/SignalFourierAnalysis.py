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


def FFTSignalAnalysis(file, minCrop=1.0, focus = None, \
                        takeLabel=True, dataLabels = None, colMajor = 1, \
                        keepHigh = True):
    '''
    This function is to analyze a signal that is recorded on a text file with
    tab seperated columns. The only requirements are that the first column of
    this file be time records.

    If an interval of time is to be focused on, the
    variable 'focus' should be a tuple of (beginning time, end time) as either
    a float or int.

    If further cropping is required, based on other data,
    user must specify which column to evaluate with colMajor, with minCrop
    determineing the line. If data evaluated higher than minCrop is to be kept,
    keepHigh = True. If data lower than minCrop, then keepHigh should be False.

    Finally, user can specify the labels for the data columns by giving an
    array of strings to dataLabels. If not, then the function will check
    to see if the first row of the data file contains strings, and will use
    these as labels if it finds any. Otherwise, each column is sequencially
    labeled.
    '''

    #Strips name of file
    fileNameRegex = re.compile(r'(\w|-)+')
    fileName = fileNameRegex.search(file).group()

    #Brings in data from datafile in the /Data/ fdataer
    script_dir = os.path.dirname(__file__)
    rel_path = 'Data/'+file
    abs_file_path = os.path.join(script_dir, rel_path)
    data = np.genfromtxt(abs_file_path, delimiter='\t')



    #Separates out values from larger array
    if all(isinstance(item, str) for item in data[0,:]) and \
    takeLabel == True and dataLabels == None:
        dataLabels = data[0,:]
        data = np.delete(data,(0),axis=0)


    #crops signal if one area is to be focused on
    if focus != None:
        data = signalCrop(data, 0, focus)

    #crops signal based on strength of specified signal.
    #Lower/Higher than specified values are cropped out.
    croppedData = signalCrop(data, colMajor, minCrop, keepHigh)

    #removes mean, setting average to zero,
    for i in range(1,len(data[0,:])):
        croppedData[:,i] -= np.mean(croppedData[:,i])

    #find the max value that can be plotted out to
    deltaT = []
    for i in range(len(croppedData[:,0])-1):
        deltaT.append(croppedData[i+1,0]-croppedData[i,0])
    avgDeltaT = np.mean(deltaT)

    # makes an array for frequencies
    freq = np.linspace(0.0, 1.0/(2.0*avgDeltaT),len(croppedData[:,0])/2)

    # does the fast fourier transform on each of the data columns and sets
    #them up to be plotted
    dataF = FourierTransform(croppedData, freq)

    #does peakfinding on each of them
    peakData = PeakDetect(dataF)

    #graphs data
    graphSignals(data, croppedData, dataF, peakData, \
                fileName, script_dir, dataLabels = dataLabels)

def signalCrop(data, colMajor, cropVal, keepHigh=True):
    '''
    This function crops the data based on a set of parameters. It consists of
    two major parts.

    First, if cropVal is a int or a float, then it crops all data for which
    the colMajor column's values are lower or higher than cropVale value,
    depending on if keepHigh is true or false, respectively.

    Second, if cropVal is a tuple, then it crops out all data for which the
    colMajor column's values are not between the maximum and minimum of cropVal.

    The second part is used for time selection primarily, while the first is
    used for data specific cropping.
    '''

    #define array of row indexes to be cropped
    croppedRows = []

    if type(cropVal) == int or type(cropVal) == float:


        for i in range(len(data[:,colMajor])):
            #appends row index if colMajor column value is lower than cropVal
            if data[i,colMajor] < cropVal and keepHigh == True:
                croppedRows.append(i)

            #appends row index if colMajor column value is higher than cropVal
            if data[i,colMajor] > cropVal and keepHigh == False:
                croppedRows.append(i)

    if type(cropVal) == list:

        for i in range(len(data[:,colMajor])):
            #appends row index if colMajor column value is not in cropVal
            if data[i,colMajor]< min(cropVal) \
                or data[i,colMajor] > max(cropVal):

                croppedRows.append(i)

    #deletes rows
    data = np.delete(data, croppedRows, axis = 0)

    return data



def graphSignals(data, cropped, transformed, peaks, fileName, \
                    script_dir, dataLabels = None):
    '''
    This function takes three sets of matricies (data, cropped,
    and transformed) and plots them in rows. The first columns of each is an
    array of the independant variable(time or frequency). The following columns
    are the signals themselves. Data is the original input, cropped is the signal
    after any specified cropping (not including focusing on a specific time),
    and transformed is after FFT. Peaks is a seperate array that will plot on
    the same graph as the transformed, and specify the peaks in those graphs.

    The graphs are saved as a png with the title and file name based on the
    original file input name. As such, fileName and script_dir are used here.

    Last, data labels may be entered to add the the graph. They must be input
    as strings into an array, where the ith' element of the array is the
    name of the the i'th signal. For example, if our signals are a sum signal,
    a left-right signal, and a top-bottom signal, dataLabels will be
    ['Sum Signal', 'Left-Right Signal', 'Tob-Bottom Signal'].
    If nothing is given, then each signal is labeled with a number (signal 1,
    signal 2, signal 3, ....)
    '''
    #This chechs to see if there is an array of data labels, and create
    #one if not.
    if dataLabels == None:
        dataLabels = []
        for i in range(1, len(transformed[0,1:])+1):
            dataLabels.append(str(i))
    elif dataLabels != None:
        dataLabels = map(str, dataLabels)

    #sets up variable to specify subplots
    subplotIndex = 1
    #Sets up figure, whose size is dependant on the number of signals
    plt.figure(figsize=(8*len(data[0,1:]),5*len(data[0,1:])))
    plt.suptitle(fileName + ',  %0.1f sec to %0.1f sec'
                    %(min(data[:,0]),max(data[:,0])), fontsize = 20)
    #Plots the original signals
    for j in range(1,len(data[0,1:])+1):

        plt.subplot(3,len(data[0,1:]),subplotIndex)
        plt.plot(data[:,0], data[:,j])
        plt.title('Original '+dataLabels[j-1], fontsize=16)
        plt.ylabel('Volts (V)',fontsize=12)
        plt.xlabel('Time (s)',fontsize=12)
        plt.grid()
        subplotIndex +=1

    #Plots the cropped signals
    for j in range(1,len(cropped[0,1:])+1):
        plt.subplot(3,len(cropped[0,1:]),subplotIndex)
        plt.title('Cropped '+dataLabels[j-1], fontsize=16)
        plt.ylabel('Volts (V)',fontsize=12)
        plt.xlabel('Time (s)',fontsize=12)
        plt.plot(cropped[:,0],cropped[:,j])
        plt.grid()
        subplotIndex+=1

    # plots  fourier transforms
    for j in range(1,len(transformed[0,1:])+1):
        plt.subplot(3,len(transformed[0,1:]),subplotIndex)
        plt.plot(transformed[:,0],transformed[:,j])
        PeaksLabel = 'Peaks'

        #Tries to plot peaks. If there are none, then skips plotting
        #which throws an error, and skips it.
        try:
            plt.scatter((peaks[j-1][:,0]), peaks[j-1][:,1], label='Peaks')
            for k in range(len(peaks[j-1][:,0])):
                PeaksLabel += '\n %0.4f Hz,'%(peaks[j-1][k,0])
        except:
            PeaksLabel += ' nowhere'
        plt.plot(0,0, color='w', label=PeaksLabel)
        plt.title('Fourier Transform of Cropped '+dataLabels[j-1], fontsize=16)
        plt.ylabel('Volts per Hertz (V/Hz)',fontsize=12)
        plt.xlabel('Frequency (Hz)',fontsize=12)
        plt.legend(bbox_to_anchor=(1,1), fontsize = 10)
        plt.grid()
        subplotIndex+=1

    #Formatting
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    #Specifies path and name
    new_rel_path = 'Data/%s-%0.2fs-%0.2fs.png'%(fileName, data[0,0],data[-1,0])
    new_abs_file_path = os.path.join(script_dir, new_rel_path)

    #save figure
    plt.savefig(new_abs_file_path, dpi=200)


    #plt.show()


def FourierTransform(data, freq):
    '''
    This function does a fast fourier tranform (FFT) of a series of signals. An array
    for the corresponding frequency is given also, and the resulting transformed
    signals are appended to form a matrix, where the first column is the
    frequency and each following column is a signal.
    '''

    transformData = freq
    transformData = np.reshape(transformData, (-1,1)) #reshapes into a vector

    for i in range(1,len(data[0,:])):

        #uses numpy for actual FFT
        tempArray = np.fft.fft(data[:,i])

        #does appropriate cropping and scaling
        tempArray = 2.0/len(data[:,0]) * np.abs(tempArray[:len(data[:,0])/2])

        #reshapes into a vector
        tempArray = np.reshape(tempArray,(-1,1))

        #appends to matrix
        transformData = np.append(transformData, tempArray, axis=1)
    return transformData


def PeakDetect(data):
    '''
    This function uses the accompanying program peakdetect.py to find peaks in
    the transformed data, and saves them out as an array of arrays, where each
    element array corresponds to each transformed signal.
    '''
    peakData = []
    for i in range(1, len(data[0,:])):
        Peaks, Lows = pd.peakdet(data[:,i],(max(data[:,i])/15),data[:,0])
        peakData.append(Peaks)
    return peakData



dataLabelLarge = ['sumSignal', 'LR_Signal', \
            'TB_Signal', 'Floppy_Signal', \
            'Not_Floppy_Signal']
dataLabel = ['sumSignal', 'LR_Signal', 'TB_Signal']
FFTSignalAnalysis('TestFile.txt', focus = None)

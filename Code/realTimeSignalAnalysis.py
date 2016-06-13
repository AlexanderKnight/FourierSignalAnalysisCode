import numpy as np
import matplotlib.pyplot as plt

def realTimeSignalFFTAnalysis(data, colMajor=1, cropVal=1.0, keepHigh=True):

    croppedData = signalCrop(data, colMajor, cropVal, keepHigh)
    time = croppedData[:,0]
    croppedData = np.delete(croppedData,0,axis=1)

    for i in range(len(croppedData[0,:])):
        croppedData[:,i] -= np.mean(croppedData[:,i])

    deltaT = []
    for i in range(len(croppedData[:,0])-1):
        deltaT.append(croppedData[i+1,0]-croppedData[i,0])
    removeIndex = []
    for i in range(len(deltaT)):
        if deltaT[i] >= 3*np.std(deltaT):
            removeIndex.append(i)
    deltaT = np.delete(deltaT, removeIndex)
    avgDeltaT = np.mean(deltaT)

    freq = np.linspace(0.0, 1.0/(2.0*avgDeltaT), len(time)/2)

    transformedData = FourierTransform(croppedData, freq, time)

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


def FourierTransform(data, freq, time):
    '''
    This function does a fast fourier tranform (FFT) of a series of signals. An array
    for the corresponding frequency is given also, and the resulting transformed
    signals are appended to form a matrix, where the first column is the
    frequency and each following column is a signal.
    '''

    transformData = freq
    transformData = np.reshape(transformData, (-1,1)) #reshapes into a vector

    for i in range(len(data[0,:])):

        #uses numpy for actual FFT
        tempArray = np.fft.fft(data[:,i])

        #does appropriate cropping and scaling
        tempArray = 2.0/len(time) * np.abs(tempArray[:len(time)/2])

        #reshapes into a vector
        tempArray = np.reshape(tempArray,(-1,1))

        #appends to matrix
        transformData = np.append(transformData, tempArray, axis=1)
    return transformData



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

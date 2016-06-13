import numpy as np
import scipy as sp
import os

pathName = os.path.join(os.path.dirname(__file__), 'Data/TestFile2.txt')
testFile = open(pathName, 'w')
testFile.write('#Time\tsumSignal\tLR_Signal\tTB_Signal \n')

t = np.linspace(0.0, 300.0, 1000)
sumsignal = 10*np.sin(0.005*t)
for i in range(len(sumsignal)):
    if sumsignal[i]>1.1:
        sumsignal[i]=1.1
    elif sumsignal[i]<0.1:
        sumsignal[i]=0.1
lrsignal = np.sin(0.04 * 2.0 * np.pi*t)
tbsignal = np.sin(0.25 * 2.0 * np.pi*t)+ 0.8 * np.sin(0.5 * 2.0 * np.pi * t) + 0.4 * np.sin(0.75 * 2.0 * np.pi * t)
fiddleSignal = np.sin(0.08 *2.0 * np.pi*t)
waddleSignal = np.sin(0.000003 * 2.0 * np.pi * t)
for i in range(1000):
    testFile.write('%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n'%(t[i],sumsignal[i],lrsignal[i],tbsignal[i],fiddleSignal[i],waddleSignal[i]))

testFile.close()

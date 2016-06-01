import numpy as np
import scipy as sp

testFile = open('TestFile.txt', 'w')
testFile.write('#Time\tsumSignal\tLR_Signal\tTB_Signal \n')

t = np.linspace(0.0, 300.0, 1000)
sumsignal = 3*np.sin(0.05*t)
for i in range(len(sumsignal)):
    if sumsignal[i]>1.1:
        sumsignal[i]=1.1
    elif sumsignal[i]<0.1:
        sumsignal[i]=0.1
lrsignal = np.sin(0.4*2.0*np.pi*t) + 0.25*np.sin(0.6*2.0*np.pi*t)# + 4*np.sin(80.0*2.0*np.pi*t) + 3*np.sin(100.0*2.0*np.pi*t)
tbsignal = 0.01*np.sin(0.2*2.0*np.pi*t) #+ 3*np.sin(0.5*2*np.pi*t) + 1*np.sin(1.0*2*np.pi*t)
for i in range(1000):
    testFile.write('%0.6f\t%0.6f\t%0.6f\t%0.6f\n'%(t[i],sumsignal[i],lrsignal[i],tbsignal[i]))

testFile.close()

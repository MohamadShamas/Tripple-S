# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 11:41:43 2017

@author: Mohamad Shamas
"""

# generate and plot gaussian noise
import numpy
import matplotlib.pyplot as plt
import scipy.io as sio

#mean = 0
#std = 0.08 
#num_samples =4096*2
#numpy.random.seed(seed=12)
#samples = numpy.random.normal(mean, std, size=num_samples)
#
#pulse = numpy.zeros(4096*2);
#
#samples[2300*2:2330*2] = 1
#plt.plot(samples,'k')
#plt.ylim(-2,2)
#plt.show()

#sig = sio.loadmat('sig_out.mat')
#print type(sig['Sig'])
#plt.plot(sig['Sig'],'k')
#plt.ylim(-2,2)
#plt.show()

def boltzman(x, xmid, tau):
    """
    evaluate the boltzman function with midpoint xmid and time constant tau
    over x
    """
    return 1. / (1. + numpy.exp(-(x-xmid)/tau))
    
t = numpy.arange(0,0.2+1./2048,1./2048);
S = boltzman(t,15, 1./2.5)
plt.ylim(-0.1,2)
plt.plot(S,'k')

y = t*numpy.exp(-t*30) +0.001
plt.xlim(-10,400)
plt.ylim(-0.0001,0.02)
plt.plot(y,'k')
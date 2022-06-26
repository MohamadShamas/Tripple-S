# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:06:19 2015

@author: Mohamad Shamas
"""

import numpy as np
import array
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os

#import gc
#from pylab import *
#import random

class EEGData():
   
   def __init__(self,dipCenters,senCenters,normals,area,temporalData,mode):
        self.dipCenters=dipCenters
        self.senCenters=senCenters
        self.normals=normals
        self.area=area
        self.temporalData=temporalData
        self.mode=mode

   def DirDis(self,senCenters,dipCenters):
        #senCenters=[(1,1,1),(2,2,2),(3,3,3)]
        #dipCenters=[(10,0,0),(0,10,0),(0,0,10),(10,10,0),(10,0,10),(0,10,10),(10,10,10)]    
        a=np.matrix(senCenters)
        b=np.matrix(dipCenters)    
        direction=[0]*a.shape[0]
        distance=[0]*a.shape[0]
        for i in range(0,a.shape[0]):
            direction[i]=b-a[i]
            distance[i]=np.linalg.norm(direction[i],axis=1)
            direction[i]=direction[i]/distance[i][None].T
        return direction, distance
   
   def dotProduct(self,UiMatrix,NiMatrix):   
        NiMatrix=np.array(NiMatrix)        
        size=NiMatrix.shape
        if len(size)==1:
            size=1
            UiMatrix=np.vstack(UiMatrix)
        else: 
            size=NiMatrix.shape[0]
            UiMatrix=np.array(UiMatrix)
        dotj=[[0]*NiMatrix.shape[0]]*UiMatrix.shape[0]
        dotj[0][0]=1    
        for j in range(UiMatrix.shape[0]):
          c=[]
          if size==1:
               c.append(np.dot(UiMatrix[j],NiMatrix[0:]))
          else:
               for i in range(size):                                
                    c.append(np.dot(UiMatrix[j][i],NiMatrix[i]))
          dotj[j]=c 
        return dotj
    
   def Vj(self):
        sigma=1
        
        if self.mode==0:
           # Spacial-Temporal averaging
           baryCenter=np.mean(self.dipCenters,axis=0)
           resultantN=np.sum(self.normals,axis=0)
           norm=np.linalg.norm(resultantN)
           if norm!=0:
              resultantN=resultantN/norm
           averageArea=np.mean(self.area)
           averageSignal=np.mean(self.temporalData,axis=0)
           Ui,r=self.DirDis(self.senCenters,baryCenter)
           dProduct=self.dotProduct(Ui,resultantN)  
           v=[]
           r2=np.array(r)
           averageSignal=np.array(averageSignal)           
           for j in range(np.matrix(self.senCenters).shape[0]):
               v.append(dProduct[j][0]*averageSignal*averageArea*0.525/((4*np.pi*sigma)*r2[j][0])**2)    
           return v
        if self.mode==1:
           # Temporal averaging
           averageSignal=np.mean(self.temporalData,axis=0)
           Ui,r=self.DirDis(self.senCenters,self.dipCenters)
           dProduct=self.dotProduct(Ui,self.normals)
           #dProduct=np.array(dProduct)
           v=[]
           averageSignal=np.array(averageSignal)
           for j in range(np.matrix(self.senCenters).shape[0]):          
               vn=[0]*np.matrix(self.dipCenters).shape[0]
               for i in range(np.matrix(self.dipCenters).shape[0]):
                   vn[i]=dProduct[j][i]*averageSignal*self.area[i]*0.525/((4*np.pi*sigma)*r[j][i]**2)                   
               v.append([sum(x) for x in zip(*vn)])    
           return v
        if self .mode==2 or self .mode==3 :
           # Spacial averaging (orientation only)
           Ui,r2=self.DirDis(self.senCenters,self.dipCenters)           
           if self .mode==2:
               resultantN=np.sum(self.normals,axis=0)
               norm= np.linalg.norm(resultantN)
               if norm!=0:
                   resultantN=resultantN/norm
               normals=[]
               normals.append(tuple(resultantN))
               normals=normals*len(self.dipCenters)
           else:
               normals=self.normals
           Ui,r=self.DirDis(self.senCenters,self.dipCenters)
           dProduct=self.dotProduct(Ui,normals)  
           v=[]
           for j in range(np.matrix(self.senCenters).shape[0]):          
               vn=[0]*np.matrix(self.dipCenters).shape[0]
               for i in range(np.matrix(self.dipCenters).shape[0]):
                   vn[i]=dProduct[j][i]*self.temporalData[i]*self.area[i]*0.525/((4*np.pi*sigma)*r2[j][i]**2)                   
               v.append([sum(x) for x in zip(*vn)])    
           return v

#def ReadEEGfile(filename,fs,nChannels,nSamples,T0,mode):
#    if mode=='b':
#        fin=open(filename,'rb')         
#        x=array.array('f')
#        x.read(fin,nSamples*nChannels)
#        start=T0*fs*nChannels
#        sig=[0 for i in range(nSamples-T0*fs)]
#        for i in range(nChannels):
#            sig[i]=x[start+i::nChannels]
#        return sig    
import cPickle
import scipy.io as sio

def SaveSig(filename,obj):
        fileObj=open(filename,'wb')
        cPickle.dump(obj,fileObj)
        fileObj.close()       

def LoadSig(filename):
        fileObj=open(filename,'rb')
        sig=cPickle.load(fileObj)
        fileObj.close()
        return sig
        
def ReadEEGfile(filename):
    name, ext=os.path.splitext(filename)    
    # read descriptive file  into fs, nChannels, nSamples
    data=open(name+'.des','r').readlines()[4]
    fs=float(data.split()[1])
    data=open(name+'.des','r').readlines()[7]
    nSamples=int(data.split()[1])
    data=open(name+'.des','r').readlines()[10]
    nChannels=int(data.split()[1])
    T0=0
    if ext=='.bin':
        fin=open(filename,'rb')         
        x=array.array('f')
        x.read(fin,nSamples*nChannels)
        start=T0*fs*nChannels
        sig=[0 for i in range(nSamples-T0*fs)]
        for i in range(nChannels):
            sig[i]=x[start+i::nChannels]             
        return sig    
    if ext=='.dat':
        h=np.array([[0.0]*nChannels]*nSamples)
        k=[]
        fin=open(filename,'r')  
        for lines in fin:
            k.append(lines.split(None, nChannels))
        for j in range(nChannels):
            for i in range(nSamples):
                h[i,j]= float(k[i][j])
                
        return h

def main():    
    #mode=LoadSig('mode0')
#    mode0=[]
#    for i in xrange(len(mode)):
#        mode0.append(mode[i].tolist()[0])
#    mode3=LoadSig('Electrode 1mode311')
 #   sio.savemat('mode.mat', {'vect':mode3})
    #del mode

    fs=4096
    nChannels=100
    nSamples=8192
    T0=0
    mode='b'
    filename='C:\\Users\\Mohamad Shamas\\Desktop\\slicerSim\\slicer.dat'
    s=ReadEEGfile(filename)
    print len(s)
    g=[]
    for i in xrange(nChannels):
        g.append(np.array(s[i],dtype=np.float32))
    g=np.array(g)
    
    
        #h=np.array(mode3[15])-np.array(mode3[15])
    mode3=g;
    print len(mode3)
#
#plt.figure()
#for i in range (15):
#    plt.plot(np.array(mode2[i]))
    plt.figure()
    a=np.linspace(0,1,28672)
    plt.plot(a,s.T[1])
    plt.show()
    k=[]
    mycmap = cm.get_cmap('jet')
    for z in range(1,100):
            k.append([z*.025]*  8192) 
    k=np.array(k)    
    plt.figure()
    a=np.linspace(0,1,8192)
    for i in range(100):
                c = mycmap(float(i)/(14))
                #fig.add_subplot(15,1,i+1)
                plt.plot(a,np.array(s.T[i+1])+k[i,:],color=c,label='E: '+repr(i+1))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()            
    plt.show()
##--------------------------- Read Binary File ---------------------------------    
#fs=512
#nChannels=500
#nSamples=31232
#T0=2
#mode='b'
#filename='D:\SimulatedSpikes\sim1_61sec_500pop_512Hz_MUX.bin'
#s=ReadEEGfile(filename,fs,nChannels,nSamples,T0,mode)
#g=[]
#for i in xrange(nChannels):
#    g.append(np.array(s[i],dtype=np.float32))
#g=np.array(g)
##------------------------------------------------------------------------------    
#nChannels=6
#x=arange(0,2*pi*3,2*pi*3/300)
#close("all")
#N=[(0,0,1),(0,0,1),(0,0,1),(0,0,1),(0,0,1),(0,0,1)]
#area=[1,1,1,1,1,1]
#dipcen=[(1,0,0),(0.5,0.866,0),(-0.5,0.866,0),(-1,0,0),(-0.5,-0.866,0),(0.5,-0.866,0)]
#sencen=[(0,0,-7),(0,0,-3.5),(0,0,0),(0,0,3.5),(0,0,7),(0,0,10.5),(0,0,14)]
#g=[]
#for i in xrange(nChannels):
#   g.append(np.array(sin(x+i*pi/6*x),dtype=np.float64))
#g=np.array(g)
#
#data=EEGData(dipcen,sencen,N,area,g,0)
#v=data.Vj()
#q=[]
#if data.mode==0:
#    q.append((v[1]).tolist()[0])
#    plt.figure()
#    plt.plot(q[0])
#    plt.show()
#    q.append((v[6]).tolist()[0])
#    plt.figure()
#    plt.plot(q[1])
#    plt.show()
#else:
#    plt.figure()
#    plt.plot(v[1])
#    plt.show()
#    plt.figure()
#    plt.plot(v[4])
#    plt.show()
#
## collect garbage explicitly
#gc.collect()
#
if __name__ == "__main__": main() 
   
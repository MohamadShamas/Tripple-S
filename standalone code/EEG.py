# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:53:15 2015

@author: Mohamad Shamas
"""
import time
import pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import gc
import SaveLoad

import scipy.io as sio

class EEGData():
   
   def __init__(self,dipCenters,normals,area,temporalData,numOfCells,mode):
        self.dipCenters=dipCenters
        self.normals=normals
        self.area=area
        self.temporalData=temporalData
        self.numOfCells=numOfCells
        self.mode=mode
        self.job_server=pp.Server()
        self.job_server.set_ncpus(18)
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
   
   def setSenCenters(self,centers):
       self.senCenters=centers
       
   def dotProduct(self,UiMatrix,NiMatrix):   
        sizeMode=0
        NiMatrix=np.array(NiMatrix)        
        sizes=NiMatrix.shape
        if len(sizes)==1:
            sizeMode=1
            UiMatrix=np.vstack(UiMatrix)
        else: 
            size=NiMatrix.shape[0]
            UiMatrix=np.array(UiMatrix)
        dotj=[[0]*NiMatrix.shape[0]]*UiMatrix.shape[0]
        dotj[0][0]=1    
        for j in range(UiMatrix.shape[0]):
          c=[]
          if sizeMode==1:
               c.append(np.dot(UiMatrix[j][0],NiMatrix[0:]))
          else:
               for i in range(size):                                
                    c.append(np.dot(UiMatrix[j][i],NiMatrix[i]))
          dotj[j]=c 
        return dotj
  
       
   def Vj(self):
        sigma=30*10**-5
        
        if self.mode==0:
           # Spacial-Temporal averaging
           start=0
           baryCenter=[]
           ECnormals=[]
           normals=[]
           TempNormals=[]
           for i in range (len(self.numOfCells)-1):              
               start=start+self.numOfCells[i]
               TempbaryCenter=np.mean(self.dipCenters[start:start+self.numOfCells[i+1]],axis=0)
               baryCenter.append(TempbaryCenter) 
               ECnormals=np.sum(self.normals[start:start+self.numOfCells[i+1]],axis=0)
               norm= np.linalg.norm(ECnormals)
               if norm!=0:
                  ECnormals=ECnormals/norm
               TempNormals.append(tuple(ECnormals))   
           averageArea=np.mean(self.area)
           averageSignal=np.mean(self.temporalData,axis=0)           
           Ui,r=self.DirDis(self.senCenters,baryCenter) 
           dProduct=self.dotProduct(Ui,TempNormals)  
           v=[]
           r2=np.array(r)
           averageSignal=np.array(averageSignal) 
           for j in range(np.matrix(self.senCenters).shape[0]):
               vn=[0]*(len(self.numOfCells)-1)
               for i in range(len(self.numOfCells)-1):
                  vn[i]=(dProduct[j][i]*averageSignal*averageArea*525*10**-9/((4*np.pi*sigma)*r2[j][i]**2)) 
               v.append([sum(x) for x in zip(*vn)]) 
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
                   vn[i]=dProduct[j][i]*averageSignal*self.area[i]*525*10**-9/((4*np.pi*sigma)*r[j][i]**2)                   
               v.append([sum(x) for x in zip(*vn)])    
           return v
        if self .mode==2 or self .mode==3 :
           # Spacial averaging (orientation only)
           if self .mode==2:
               ECnormals=[]
               normals=[]
               start=0
               for i in range (len(self.numOfCells)-1):
                   start=start+self.numOfCells[i]
                   ECnormals=np.sum(self.normals[start:start+self.numOfCells[i+1]],axis=0)
                   norm= np.linalg.norm(ECnormals)
                   if norm!=0:
                       ECnormals=ECnormals/norm
                   TempNormals=[]   
                   TempNormals.append(tuple(ECnormals))    
                   ECnormals=(TempNormals)*(int(self.numOfCells[i+1]))  
                   normals.extend(ECnormals)
           else:
               normals=self.normals
           Ui,r2=self.DirDis(self.senCenters,self.dipCenters)

           dProduct=self.dotProduct(Ui,normals)  
           v=[]
    
           jobs=[]
#           total=sum(self.numOfCells)
#           parts=16
#           start=0
#           step = (total - start) / parts + 1
           start_time = time.time()    
#           for i in range(16):
#               jobs.append(self.job_server.submit(CalculateV,(sum(self.numOfCells),dProduct[i],self.temporalData,self.area,r2[i]),(),("numpy",)))
           for j in range(np.matrix(self.senCenters).shape[0]):          
               vn=[0]*np.matrix(self.dipCenters).shape[0]
               for i in range(np.matrix(self.dipCenters).shape[0]):
                   #vn[i]=dProduct[j][i]*self.temporalData[i]*self.area[i]*0.525*10**-9/((4*np.pi*sigma)*r2[j][i]**2) 
                   vn[i]=self.temporalData[i]*self.area[i]*0.525*10**-9/((4*np.pi*sigma)*r2[j][i]**2)                  
               v.append([sum(x) for x in zip(*vn)])  
#           for index in xrange(16):
#               vn=np.array([0]*8192*2)
#               for j in range(sum(self.numOfCells)):
#                   vn=vn+jobs[index]()[j]
#               v.append(vn)
           print "Time elapsed: ", time.time() - start_time, "s" 
           sio.savemat('dot.mat', {'dot':dProduct})
           sio.savemat('Area.mat', {'Area':self.area})
           sio.savemat('r.mat', {'rt':r2})
           return v
   def CalculateOne(self):
       sigma=30*10**-5
       normals = self.normals
       Vn = np.zeros(len(self.temporalData[0]))
       for i in range(len(self.dipCenters)):
           direction = np.array(self.senCenters[:])-np.array(self.dipCenters[i][:])
           distance = np.linalg.norm(direction)
           #direction = direction/distance
           dProduct = np.dot(normals[i],direction)
           Vn = Vn + dProduct*self.temporalData[i]*self.area[i]/((4*np.pi*sigma)*distance**2)

       return Vn    
def CalculateV(numOfDip,dProduct,tempSig,area,r):
     sigma=30*10**-5
     vns=[0]*(numOfDip)
     for i in range(numOfDip):
            vns[i]=(dProduct[i]*tempSig[i]*area[i]*525*10**-9/((4*numpy.pi*sigma)*r[i]**2))
     return vns

    
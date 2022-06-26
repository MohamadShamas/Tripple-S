# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:56:14 2015

@author: Mohamad Shamas
"""

import cPickle

def SaveSig(filename,obj):
        fileObj=open(filename,'wb')
        cPickle.dump(obj,fileObj)
        fileObj.close()       

def LoadSig(filename):
        fileObj=open(filename,'rb')
        sig=cPickle.load(fileObj)
        fileObj.close()
        return sig
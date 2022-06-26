# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:04:38 2017

@author: Mohamad Shamas
"""
import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt

def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()
    
# Directory where the DICOM files are being stored 
pathDicom = "C:\Users\Mohamad Shamas\Desktop\RET-new\mprage-2011"

#segmentation will be limited to a single 2D image 
idxSlice = 100

# int labels to assign to the segmented white and gray matter.
labelWhiteMatter = 1
labelGrayMatter = 2    

reader = SimpleITK.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
reader.SetFileNames(filenamesDICOM)
imgOriginal = reader.Execute()

imgOriginal = imgOriginal[:,idxSlice,:]

#sitk_show(imgOriginal)

imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                    timeStep=0.0625,
                                    numberOfIterations=10)

#sitk_show(imgSmooth)

lstSeeds = [(125,67),(85,91),(128,56),(209,80),(202,60),(208,124)]
imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                              seedList=lstSeeds, 
                                              lower=210, 
                                              upper=310,
                                              replaceValue=labelWhiteMatter)
#sitk_show(imgWhiteMatter)  

# Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

# Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
#sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatter))
   

imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                          radius=[2]*5,
                                                          majorityThreshold=1,
                                                          backgroundValue=0,
                                                          foregroundValue=labelWhiteMatter)

#sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))                                         

lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43),(196,123)]

imgGrayMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                             seedList=lstSeeds, 
                                             lower=150, 
                                             upper=215,
                                             replaceValue=labelGrayMatter)

imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,
                                                         radius=[2]*3,
                                                         majorityThreshold=1,
                                                         backgroundValue=0,
                                                         foregroundValue=labelGrayMatter)

#sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgGrayMatterNoHoles))
imgLabels = imgWhiteMatterNoHoles | imgGrayMatterNoHoles

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, SimpleITK.LabelContour(imgLabels)))
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 10:49:25 2015

@author: Mohamad Shamas
"""

import os , gzip 
import numpy as N
import structutils
import vtk


BIGENDIAN = '>'

DATA_OFFSET = 284

mri_type_enum = ['MRI_UCHAR', 'MRI_INT', 'MRI_LONG', 'MRI_FLOAT',
                 'MRI_SHORT','MRI_BITMAP',]

type2numbytes = {'MRI_FLOAT':4, 'MRI_UCHAR':1, 'MRI_SHORT':2, 'MRI_INT':4}

#page 22 of http:/http://numpy.scipy.org/numpybooksample.pdf
type2dtype = {'MRI_FLOAT':'f', 'MRI_UCHAR':'b', 'MRI_SHORT':'h', 'MRI_INT':'i'}
dtype2type = {'f':'MRI_FLOAT', 'b':'MRI_UCHAR', 'h':'MRI_SHORT', 'i':'MRI_INT'}

class odict(dict):
    """
    This dictionary class extends dict to record the order in which items
    are added.  Calling keys(), values(), items(), etc. will return results in
    this order.
    """

    def __init__(self, items=()):
        self._keys = map(lambda t: t[0], items)
        dict.__init__(self, items)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._keys.remove(key)

    def __setitem__(self, key, item):
        dict.__setitem__(self, key, item)
        if key not in self._keys: self._keys.append(key)

    def clear(self):
        dict.clear(self)
        self._keys = []

    def copy(self):
        dict = dict.copy(self)
        dict._keys = self._keys[:]
        return dict

    def sort( self, keyfunc=None ):
        if keyfunc is None: self._keys.sort()
        else:
            decorated = [(keyfunc(key),key) for key in self._keys]
            decorated.sort()
            self._keys[:] = [t[1] for t in decorated]

    def items(self):
        return zip(self._keys, self.values())

    def keys(self):
        return self._keys

    def popitem(self):
        try:
            key = self._keys[-1]
        except IndexError:
            raise KeyError('dictionary is empty')

        val = self[key]
        del self[key]
        return (key, val)

    def setdefault(self, key, failobj = None):
        if key not in self._keys: self._keys.append(key)
        return dict.setdefault(self, key, failobj)

    def update(self, other):
        dict.update(self, other)
        for key in other.keys():
            if key not in self._keys: self._keys.append(key)

    def values(self):
        return map(self.get, self._keys)


hdr_struct = odict((
    ('version','i'),
    ('width','i'),
    ('height','i'),
    ('depth','i'),
    ('nframes','i'),
    ('type','i'),
    ('dof','i'),
    ('goodRASFlag','h'),
    ('delta','3f'),
    ('Mdc','9f'),
    ('Pxyz_c','3f'),
))

class MGH:

    _field_defaults = {}

    def __init__(self):
        self.header={}

    # ---------------------- load() method ------------------- #
    def load(self, fname, slices=[], frames=[], headeronly=0):
        """ Loads the mgh volume onto an MGH structure
        :Parameters
            fname - string
            The mgh file to be loaded. Can be gzipped mgh also (.mgz) 
        
        """

        # quick way to check if file exists / readable 
        os.stat(fname)
        self.fname = fname
        # mgz => get the gunzipped file object
        if fname.lower().endswith('.mgz') or fname.lower().endswith('.gz'):
            self.fobj = gzip.GzipFile(fname, 'rb');
        else:
            self.fobj = open(fname, 'rb');

        # header is a dict which has fields specific to mgh structure.
        self._fill_header()

        # fill some variables
        _sh = self.header
        self.w = _sh['width']
        self.h = _sh['height']
        self.d = _sh['depth']
        self.n = _sh['nframes']
        
        self.nv = self.w * self.h * self.d * self.n
        self.mri_type = mri_type_enum[ _sh['type' ]]
        self.nbytespervox = type2numbytes[self.mri_type]
        # Numpy dataype descriptor. Takes Endianness into account 
        self.numpy_dtype = BIGENDIAN + type2dtype[self.mri_type] 
        
        # if goodRASFlag is set, additional processing needed.
        # and additional public variables created in MGH class. 
        if _sh['goodRASFlag']:
            self._process_ifgoodRASFlag()
        
        # if headeronly flag is set, process footer and exit before importing any data
        if headeronly:
            self._process_footer()
            self.fobj.close()
            return
       
        if slices or frames:
            # If subset of frames or slices are to be loaded
            self._load_subset_framesslices(slices, frames)
        else:    
            # Otherwise read the entire volume
            self.fobj.seek(0)
            self.fobj.seek(DATA_OFFSET)
            tmpstring = self.fobj.read(self.nv * self.nbytespervox )
            self.vol = N.fromstring(tmpstring, dtype=self.numpy_dtype)
            if not self.vol.shape[0] == self.nv:
                self.fobj.close()
                raise IOError('Voxels that were read dont match voxels supposed to read')
            self.vol = self.vol.reshape(self.w, self.h, self.d, self.n,
                                        order='Fortran')
            
        # nframes most of the time is 1. So squeeze the 4th dimension
        self.vol = self.vol.squeeze()
        self._process_footer()
        self.fobj.close()
        return



    def _fill_header(self):
        """
        PRIVATE method which helps in filling MGH object header from the
        .mgh file
        """
        values = structutils.struct_unpack(self.fobj, BIGENDIAN, hdr_struct.values() )
        for field, val in zip(hdr_struct.keys(), values):
            self.header[field] = val

    def _process_ifgoodRASFlag(self):
        """
        PRIVATE method which gets executed if goodRASFlag is set
        """
        _sm = self.header
        _sm['delta'] = N.array(_sm['delta']) 
        _sm['Mdc'] = N.array(_sm['Mdc']).reshape(3,3,order='f')
        _sm['Pxyz_c'] = N.array(_sm['Pxyz_c']).T
        
        D = N.diag(_sm['delta'])

        _sm['Pcrs_c'] = N.array([self.w/2.0, self.h/2.0, self.d/2.0]).T
        _sm['Pxyz_0'] = _sm['Pxyz_c'] - N.dot(_sm['Mdc'], N.dot(D, _sm['Pcrs_c']))
        
        t1 = N.dot(_sm['Mdc'], D)
        t2 = _sm['Pxyz_0']
        t3 = _sm['Mdc']
        t4 = _sm['Pxyz_c']
        _sm['M'] = N.array([[t1[0,0], t1[0,1], t1[0,2], t2[0]],
                            [t1[1,0], t1[1,1], t1[1,2], t2[1]],
                            [t1[2,0], t1[2,1], t1[2,2], t2[2]],
                            [0., 0., 0., 1.]]);
        _sm['ras_xform'] = N.array([[t3[0,0], t3[0,1], t3[0,2], t4[0]],
                            [t3[1,0], t3[1,1], t3[1,2], t4[1]],
                            [t3[2,0], t3[2,1], t3[2,2], t4[2]],
                            [0., 0., 0., 1.]]);
        
    def _process_footer(self):
        """
        PRIVATE method which fills in mrparms from the .mgh file
        """
        # Go to the footer. It immediately follows the data.
        self.fobj.seek(0)
        self.fobj.seek(DATA_OFFSET + self.nv * self.nbytespervox)
        footer = self.fobj.read(4 * 4) #four float32s.
        if not footer:
            raise IOError('MGH footer (mrparms) missing')
            
        self.mrparms = N.fromstring(footer, dtype=BIGENDIAN+'f')
        if not len(self.mrparms) == 4:
            print 'WARNING: error reading MR params (footer size is not 4 floats)'

    def _load_subset_framesslices(self, slices, frames):
        """
        PRIVATE method.. gets executed when slices or frames argument in load()
        is not empty
        """
        if not frames:
            frames = range(self.n)
        if not slices:
            slices = range(self.d)
            
        nvslice = self.w * self.h
        nvvol = self.w * self.h * self.d
        self.vol = N.zeros( (self.w, self.h, len(slices), len(frames)), dtype=self.numpy_dtype)
        framecount = 0
        for frame in frames:
            slicecount = 0
            for slice in slices:
                filepos = ( frame * nvvol + slice * nvslice ) * self.nbytespervox + DATA_OFFSET
                self.fobj.seek(0)
                self.fobj.seek(filepos)
                tmpstring = self.fobj.read(nvslice * self.nbytespervox)
                tmpslice = N.fromstring(tmpstring, dtype=self.numpy_dtype)
                if not tmpslice.shape[0] == nvslice:
                    self.fobj.close()
                    raise IOError('Error when reading slices/frames')
                self.vol[:,:,slicecount,framecount] = tmpslice.reshape(self.w, self.h, order='f').copy()
                slicecount = slicecount + 1
            framecount = framecount + 1

from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *

def main():
    
    fname="C:\\Users\\Mohamad Shamas\\Desktop\\bert1\\mri\\orig.mgz"
    #fname="C:\Users\Mohamad Shamas\Downloads\\brain.mgz"
    image=MGH()
    image.load(fname)
    h=image.vol
    hdr=image.header
    #h[h<300]=0
    d = np.diag(hdr['delta'])
    pcrs_c = [128,128,128] 
    Mdc = hdr['Mdc'].T
    pxyz_0 = hdr['Pxyz_c'] - np.dot(Mdc, np.dot(d, pcrs_c))
    M = np.eye(4, 4)
    M[0:3, 0:3] = np.dot(Mdc, d)
    M[0:3, 3] = pxyz_0.T
    print M         
    #data.astype(N.float64).tostring('F') == str_io.getvalue() 
               
    #f=copy.deepcopy(h)
    #h[h<100]=0
    
    #h=f-h
    #for i in range(256):        
        #h[:,:,i]=ndimage.rotate(h[:,:,i],90)
    figure()   
    f=contour(h[150,:,:], levels=[1,66,100], origin='image')  
    print f
    show()    
     
    #plt.imshow(h[:,:,100])    
    
##    Volume
##    --------------------------------------------------------------------------
#    h=N.require(h[:,:,:],dtype=N.uint8)
#    dataImporter=vtk.vtkImageImport()
#    data_string=h.tostring()
#    dataImporter.CopyImportVoidPointer(data_string,len(data_string))
#    dataImporter.SetDataScalarTypeToUnsignedChar()
#    dataImporter.SetNumberOfScalarComponents(1)
#    w, d, h = h.shape
#    dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
#    dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
#    
#    alphaChannelFunc = vtk.vtkPiecewiseFunction()
#    colorFunc = vtk.vtkColorTransferFunction()
#    
#    for i in range(1100):                 
#           if i<80:
#               alphaChannelFunc.AddPoint(i, 0)
#           else:
#               alphaChannelFunc.AddPoint(i, 0.1)
#           
#           colorFunc.AddRGBPoint(i,i/255.0,i/255.0,i/255.0)
#    
#    colorFunc.AddRGBPoint(0,0,0,0)    
#    volumeProperty = vtk.vtkVolumeProperty()
#    volumeProperty.SetColor(colorFunc)
#    volumeProperty.SetScalarOpacity(alphaChannelFunc)        
#    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
#    volumeMapper = vtk.vtkVolumeRayCastMapper()
#    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
#    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
#    volume = vtk.vtkVolume()
#    volume.SetMapper(volumeMapper)
#    volume.SetProperty(volumeProperty)                 
#    c=volume.GetCenter()
#    print c
#    # volume cutting
#    
#    normal=N.array([0.04,-0.1,0.088])*10
#   
#    plane1 = vtk.vtkPlane()
#    plane1.SetOrigin([c[0]-110,c[1]-150,c[2]-125])
#    plane1.SetNormal(normal)   
#    plane2 = vtk.vtkPlane()
#    plane2.SetOrigin([c[0]+normal[0]-110,c[1]+normal[1]-150,c[2]+normal[2]-125])
#    plane2.SetNormal(-normal)
#    
#    
#    mapVolume = vtk.vtkVolumeRayCastMapper()
#    
#    
#    
#    funcRayCast = vtk.vtkVolumeRayCastCompositeFunction()
#    funcRayCast.SetCompositeMethodToClassifyFirst()
#    
#    mapVolume.SetVolumeRayCastFunction(funcRayCast)
#    mapVolume.SetInput(dataImporter.GetOutput())
#    planes=vtk.vtkPlaneCollection()
#    #planes.AddItem(plane1)  
#    #planes.AddItem(plane2)  
#    mapVolume.SetClippingPlanes(planes)        
#    actorVolume = vtk.vtkVolume()
#    actorVolume.SetMapper(mapVolume)
#    actorVolume.SetProperty(volumeProperty)
#    renderer = vtk.vtkRenderer()
#    renderWin = vtk.vtkRenderWindow()
#    renderWin.AddRenderer(renderer)
#    renderInteractor = vtk.vtkRenderWindowInteractor()
#    renderInteractor.SetRenderWindow(renderWin)
#    
#    #--------------- read brain file using the given file path---------------------
#    
#    filePath="C:\Users\Mohamad Shamas\Downloads\cortex4000.vtk"
#    brain=vtk.vtkDataSetReader()
#    brain.SetFileName(filePath)
#    brain.ReadAllScalarsOn()
#    brain.Update()
##******************************************************************************
#    #-------------------create a mapper and actor for brain------------------------
#    brainMapper=vtk.vtkPolyDataMapper()
#    brainMapper.SetInputConnection(brain.GetOutputPort())
#    brainMapper.SetScalarModeToUseCellFieldData()
#    brainMapper.GlobalImmediateModeRenderingOn()
#    brainActor = vtk.vtkLODActor()
#    brainActor.GetProperty().SetOpacity(1)
#    brainActor.SetMapper(brainMapper)
#    brainActor.SetPickable(0) #prevent brain from being panned      
#    brainActor.GetProperty().SetOpacity(0.54)    
##******************************************************************************   
#    trans = vtk.vtkTransform()
#    trans.RotateX(-90)
#    trans.RotateZ(15)
#    trans.Translate(-110,-150,-125)
#    actorVolume.SetUserTransform(trans)
#    # We add the volume to the renderer ...
#    renderer.AddActor(actorVolume)
#    renderer.AddActor(brainActor)
#    #renderer.AddVolume(volume)
#    # ... set background color to white ...
#    renderer.SetBackground(0,0,0)
#    # ... and set window size.
#    renderWin.SetSize(400, 400)
#     
#    renderer.ResetCamera()
#    cam = renderer.GetActiveCamera()   
#    cam.Zoom(1.8)
#    renderInteractor.Initialize()
#    # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
#    renderWin.Render()
#    renderInteractor.Start()
#   #--------------------------------------------------------------------------- 
#    
if __name__ == "__main__": main()       
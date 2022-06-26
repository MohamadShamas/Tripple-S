# -*- coding: utf-8 -*-
"""
Created on Mon Jan 05 15:34:47 2015

@author: Mohamad Shamas

Simulating Script / testing
"""
import vtk
import math
import array
import numpy as np
import EEG
import SaveLoad
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import gc
import os
from ReadImageVolume import MGH
from sanstitre0 import ReadSegmentedOrgans
import scipy.io as sio
import time
from threading import Thread 

class Patch():
    def __init__(self,style):
        
        self.PatchMapper=vtk.vtkDataSetMapper()
        self.PatchActor =vtk.vtkActor()
        self.PatchActor.SetMapper(self.PatchMapper)  
        # Actors for normals
        self.glyphMapper = vtk.vtkPolyDataMapper()
        self.glyphActor = vtk.vtkActor()
        self.glyphActor.SetMapper(self.glyphMapper)
        self.glyphActor.SetPickable(1)
        
        if style==1:
            self.PatchActor.GetProperty().SetColor(0,1,0)
            self.triangleFilter=vtk.vtkTriangleFilter()
            self.cellPointIds=vtk.vtkIdList()        
            self.selectionNode=vtk.vtkSelectionNode()
            self.selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
            self.selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
            self.selection=vtk.vtkSelection()        
            self.selection.AddNode(self.selectionNode)
            self.exSelection=vtk.vtkExtractSelection()
            self.exPoly =vtk.vtkDataSetSurfaceFilter()
            self.area=vtk.vtkMassProperties()
            self.area.SetInputConnection(self.exPoly.GetOutputPort())
            self.neighborCellIds =vtk.vtkIdList()
        if style ==2:
            self.CellsMapper =vtk.vtkDataSetMapper()
            self.selectionNode=vtk.vtkSelectionNode()
            self.selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
            self.selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
            self.selection=vtk.vtkSelection()
            self.selection.AddNode(self.selectionNode)
            self.FrSelection=vtk.vtkExtractSelection()
            self.exPoly =vtk.vtkDataSetSurfaceFilter()
            self.exPoly.SetInput(self.FrSelection.GetOutput())
            self.CellsActor =vtk.vtkActor()
        if style==3:
            self.SelectedMapper = vtk.vtkDataSetMapper()
            self.SelectedActor = vtk.vtkActor()
            self.SelectedActor.SetMapper(self.SelectedMapper)
            self.extracted=vtk.vtkExtractSelection()
    def setPatchColor(self,color):
        self.PatchActor.GetProperty().SetColor(color)

class AreaStyle(vtk.vtkInteractorStyleRubberBand3D):
    def __init__(self,brain,ReqArea=1000):
        self.RequiredArea=ReqArea
        self.Brain=brain
        self.picker = vtk.vtkCellPicker()
        #add an observer that starts on left button release event        
        self.AddObserver("LeftButtonReleaseEvent",self.PickArea)
        self.AddObserver('LeftButtonPressEvent',self.set_mmZero)
        self.AddObserver('MouseMoveEvent',self.set_mmOne)
    def SetReqArea(self,ReqArea):
        self.RequiredArea=ReqArea
        
    def Create(self):    
        self.triangleFilter=self.GetInteractor().SinglePatch.triangleFilter
        self.triangleFilter.SetInputConnection(self.Brain.GetOutputPort())
        self.triangleFilter.Update()
        self.cellPointIds=self.GetInteractor().SinglePatch.cellPointIds        
        self.selectionNode=self.GetInteractor().SinglePatch.selectionNode
        self.selection=self.GetInteractor().SinglePatch.selection        
        self.exSelection=self.GetInteractor().SinglePatch.exSelection
        self.exSelection.SetInputConnection(0, self.Brain.GetOutputPort())
        self.exPoly =self.GetInteractor().SinglePatch.exPoly
        self.exPoly.SetInput(self.exSelection.GetOutput())
        self.area=self.GetInteractor().SinglePatch.area
        self.neighborCellIds =self.GetInteractor().SinglePatch.neighborCellIds   
        self.MouseMove=0

      
        
    def set_mmZero(self,obj,event):
      self.MouseMotion=0
      self.OnLeftButtonDown()

    def set_mmOne(self,obj,event):
      self.MouseMotion=1
      self.OnMouseMove()
    
    def  PickArea(self,obj,event):
     self.OnLeftButtonUp()
     if self.addElectrode==0:
         if self.MouseMotion==0:
            x,y=self.GetInteractor().GetEventPosition()
            self.picker.Pick(x,y,0,self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
            cellId=self.picker.GetCellId()
            if cellId==-1:
                pass
            else:
                #Start of neighborhood functionality
                neighbors=[]
                neighbors.append(cellId)
                x=0
                a=0
                
                while (a<self.RequiredArea):  
                
                  self.triangleFilter.GetOutput().GetCellPoints(neighbors[x],self.cellPointIds)
                  for i in range(self.cellPointIds.GetNumberOfIds()):
                    idList=vtk.vtkIdList()
                    #Add one of the edge points
                    idList.InsertNextId(self.cellPointIds.GetId(i))
                    #Add the other edge point
                    if i+1==self.cellPointIds.GetNumberOfIds():
                        idList.InsertNextId(self.cellPointIds.GetId(0))
                    else:
                        idList.InsertNextId(self.cellPointIds.GetId(i+1))
                    self.triangleFilter.GetOutput().GetCellNeighbors(neighbors[x], idList, self.neighborCellIds)                                
                    for j in range(self.neighborCellIds.GetNumberOfIds()):
                        if self.neighborCellIds.GetId(j) not in neighbors:
                           neighbors.append(self.neighborCellIds.GetId(j))
                           #print(self.neighborCellIds.GetId(j))
                #  Create a dataset with the neighbor cells
                  ids=vtk.vtkIdTypeArray()
                  ids.SetNumberOfComponents(1)         
                  for it1 in range(len(neighbors)):
                    ids.InsertNextValue(neighbors[it1])
                
                  self.selectionNode.SetSelectionList(ids)             
                  self.exSelection.SetInput(1, self.selection)
                  self.exSelection.Update()                    
                  self.exPoly.Update()              
                  a=self.area.GetSurfaceArea()+self.Brain.GetOutput().GetCell(cellId).ComputeArea()
                  x=x+1                
            self.GetInteractor().SinglePatch.PatchMapper.SetInputConnection(self.exSelection.GetOutputPort())
            self.GetInteractor().SinglePatch.PatchActor.SetMapper(self.GetInteractor().SinglePatch.PatchMapper)
            act=self.GetInteractor().SinglePatch.PatchActor
            if act not in self.GetInteractor().PatchesCollection:
                self.GetInteractor().PatchesCollection.append(act)
            PatcToRender=len(self.GetInteractor().PatchesCollection)-1
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.GetInteractor().PatchesCollection[PatcToRender])
            self.GetInteractor().Render()
            self.GetInteractor().SinglePatch.PatchActor.SetPickable(1)
     else:
         x,y=self.GetInteractor().GetEventPosition()
         self.picker.Pick(x,y,0,self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
         cellId=self.picker.GetCellId()
         if cellId==-1:
                pass
         else:
             n=[0,0,0]
             elecId=len(self.GetInteractor().ActorSt.actAssembly)
             dCell=self.Brain.GetOutput().GetCell(cellId)
             dPoints=dCell.GetPoints()
             dCell.ComputeNormalDirection(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)   
             self.GetInteractor().ActorSt.actAssembly[elecId-1].SetOrientation(math.degrees(math.atan(n[2]/n[1])),0,-math.degrees(math.atan(n[0]/n[1])))  
             dCell.TriangleCenter(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)
             print n
             self.GetInteractor().ActorSt.actAssembly[elecId-1].SetPosition(n[0],n[1],n[2])         
             self.GetInteractor().Render()
           
class FreeSelection(vtk.vtkInteractorStyleRubberBand3D):
    def __init__(self,brain,parent=None):
      self.Brain=brain
      self.picker = vtk.vtkCellPicker()
      #Remove Classical Observers
      self.RemoveObservers('LeftButtonReleaseEvent')  
      self.RemoveObservers('LeftButtonPressEvent')
      self.RemoveObservers('MouseMoveEvent')
      #Add New Observers
      self.AddObserver("LeftButtonReleaseEvent",self.set_timerOff)
      self.AddObserver('LeftButtonPressEvent',self.set_timerOn)

    def Create(self):
      self.CellsMapper =self.GetInteractor().SinglePatch.CellsMapper
      self.selectionNode=self.GetInteractor().SinglePatch.selectionNode
      self.selection=self.GetInteractor().SinglePatch.selection
      self.selection.AddNode(self.GetInteractor().SinglePatch.selectionNode)
      self.FrSelection=self.GetInteractor().SinglePatch.FrSelection
      self.FrSelection.SetInputConnection(0, self.Brain.GetOutputPort())
      self.exPoly =self.GetInteractor().SinglePatch.exPoly
      self.exPoly.SetInput(self.GetInteractor().SinglePatch.FrSelection.GetOutput())
      self.CellsActor =self.GetInteractor().SinglePatch.CellsActor
    def set_timerOn(self,obj,event):
      self.timer=self.GetInteractor().CreateRepeatingTimer(1)
      self.GetInteractor().AddObserver('TimerEvent',self.FreeSelect)
      self.select=[]
    def FreeSelect(self,obj,event):
        x,y=self.GetInteractor().GetEventPosition()
        self.picker.Pick(x,y,0,self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
        cellId=self.picker.GetCellId()
        if cellId==-1:
            pass
        else:
            if cellId not in self.select:
              self.select.append(cellId)
              print(cellId)
              #  Create a dataset with the selected cells
              ids=vtk.vtkIdTypeArray()
              ids.SetNumberOfComponents(1)         
              for it1 in range(len(self.select)):
                ids.InsertNextValue(self.select[it1])
                
             
              self.selectionNode.SetSelectionList(ids)             
              self.FrSelection.SetInput(1, self.selection)
              self.FrSelection.Update()                    
              self.exPoly.Update()
        self.CellsMapper.SetInputConnection(self.FrSelection.GetOutputPort())
        self.CellsActor.SetMapper(self.CellsMapper)
        self.CellsActor.GetProperty().SetColor(0,1,1)
        
        act=self.GetInteractor().SinglePatch.CellsActor
        if act not in self.GetInteractor().PatchesCollection:
          self.GetInteractor().PatchesCollection.append(act)
        PatchToRender=len(self.GetInteractor().PatchesCollection)-1
        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.GetInteractor().PatchesCollection[PatchToRender])
        self.GetInteractor().Render()     
        self.CellsActor.SetPickable(1)
    def set_timerOff(self,obj,event):
      self.GetInteractor().DestroyTimer(self.timer)
      
class HighlightStyle(vtk.vtkInteractorStyleRubberBand3D):
   def __init__(self,parent=None):

      #add an observer that starts on left button release event
      self.RemoveObservers('LeftButtonReleaseEvent')
      self.AddObserver("LeftButtonReleaseEvent",self.LeftButtonUp)
   def Create(self):
      self.SelectedMapper = self.GetInteractor().SinglePatch.SelectedMapper
      self.SelectedActor =  self.GetInteractor().SinglePatch.SelectedActor
      self.extracted=self.GetInteractor().SinglePatch.extracted
   def LeftButtonUp(self,obj,event):
      self.OnLeftButtonUp() # terminates the rubber band when left button release event is reached
      
      # Add HArdware Selector
      hsel = vtk.vtkHardwareSelector()
      hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
      hsel.SetRenderer(self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
      #Get start and end position of rubber band rectangle
      x0,x1=self.GetStartPosition()
      y0,y1=self.GetEndPosition()
      
      #select the covererd area
      hsel.SetArea(x0,y1,y0,x1)
      res = hsel.Select()
      #Extract the Selection)
      self.extracted.SetInput(0,self.PolyData)
      self.extracted.SetInput(1,res)         
      poly =vtk.vtkDataSetSurfaceFilter()
      poly.SetInput(self.extracted.GetOutput())
      poly.Update()      
      area=vtk.vtkMassProperties()
      area.SetInputConnection(poly.GetOutputPort())
      print "%.2f mm²" %area.GetSurfaceArea() 
      #Draw the selected section along with the unselected
      self.SelectedMapper.SetInputConnection(self.extracted.GetOutputPort())
      self.SelectedMapper.ScalarVisibilityOff()
      self.SelectedActor.GetProperty().SetEdgeColor(0, 0, 0) # (R,G,B)
      self.SelectedActor.GetProperty().SetColor(0, 1, 0) # (R,G,B)
      #self.SelectedActor.GetProperty().EdgeVisibilityOn()
      self.SelectedActor.GetProperty().SetOpacity(0.5)     
      self.SelectedActor.GetProperty().SetPointSize(5)
      
              
      act=self.GetInteractor().SinglePatch.SelectedActor
      if area.GetSurfaceArea()!=0:
          if act not in self.GetInteractor().PatchesCollection:
              self.GetInteractor().PatchesCollection.append(act)
              PatchToRender=len(self.GetInteractor().PatchesCollection)-1
              self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.GetInteractor().PatchesCollection[PatchToRender])
      self.GetInteractor().GetRenderWindow().Render()
      self.HighlightProp(None)     
      self.SelectedActor.SetPickable(1)

      return
      
   def SetPolyData(self,PolyData):
      self.PolyData=PolyData
      return

def DrawNormals(iStyle,poly,i):
      normalsCalc = vtk.vtkPolyDataNormals()
      normalsCalc.SetInputConnection(poly.GetOutputPort())
      # Disable normal calculation at cell vertices
      normalsCalc.ComputePointNormalsOff()
      # Enable normal calculation at cell centers
      normalsCalc.ComputeCellNormalsOn()
      # Disable splitting of sharp edges
      normalsCalc.SplittingOff()
      # Disable global flipping of normal orientation
      normalsCalc.FlipNormalsOn()
      # Disable automatic determination of correct normal orientation
      normalsCalc.AutoOrientNormalsOff()
      # Perform calculation
      normalsCalc.Update()
#      array=normalsCalc.GetOutput().GetCellData().GetNormals()
#      v=[0,0,0]
#      array.GetTupleValue(0,v)
#      print v
      cellCenters=vtk.vtkCellCenters()
      cellCenters.VertexCellsOn()
      cellCenters.SetInputConnection(normalsCalc.GetOutputPort())
      cellCenters.Update()
      
      # Create a new 'default' arrow to use as a glyph
      arrow = vtk.vtkArrowSource()
      
      # Create a new 'vtkGlyph3D'
      glyph = vtk.vtkGlyph3D()
      # Set its 'input' as the cell-center normals 
      glyph.SetInputConnection(cellCenters.GetOutputPort())
      # Set its 'source', i.e., the glyph object, as the 'arrow'
      glyph.SetSourceConnection(arrow.GetOutputPort())
      # Enforce usage of normals for orientation
      glyph.SetVectorModeToUseNormal()
      # Set scale for the arrow object
      glyph.SetScaleModeToScaleByVector()
      glyph.SetScaleFactor(1.3)
      
      actor=iStyle.GetInteractor().GlyphCollection[i]
      # Create a mapper for all the arrow-glyphs
      actor.GetMapper().SetInputConnection(glyph.GetOutputPort())
            
      # Create an actor for the arrow-glyphs      
      actor.GetProperty().SetColor([0,0,1])
      # Add actor
      iStyle.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(actor)

class InteractorStyleTrackballActor(vtk.vtkInteractorStyleTrackballActor):
   def __init__(self,Actors,actAssembly):      
      self.Actors=Actors
      self.actAssembly=actAssembly
      self.AddObserver("MouseMoveEvent",self.MouseMove)
      #self.AddObserver("MouseMoveEvent",self.Print1)
      self.AddObserver("MouseWheelForwardEvent",self.MouseWheelForward)  
      self.AddObserver("MouseWheelBackwardEvent",self.MouseWheelBackward) 
      #self.AddObserver("MouseWheelForwardEvent",self.Print2)  
      #self.AddObserver("MouseWheelBackwardEvent",self.Print2) 
      self.AddObserver("MiddleButtonPressEvent",self.MiddleButtonOn) 
      self.AddObserver("MiddleButtonReleaseEvent",self.MiddleButtonOff) 
      self.AddObserver('LeftButtonPressEvent',self.MouseMoveOK)
      self.AddObserver('LeftButtonReleaseEvent',self.MouseMoveNot)
      
      self.AddObserver("KeyPressEvent",self.ConfirmPatch)
      
      self.DetectMouseMove=0
      # create cutter
      self.cutter=vtk.vtkCutter()
      self.mapper=vtk.vtkPolyDataMapper()
      self.mapper.SetInputConnection(self.cutter.GetOutputPort())
      self.mapper.ScalarVisibilityOff()
      self.cutterActor=vtk.vtkActor()
      self.cutterActor.GetProperty().SetColor(0,1,1)
      self.cutterActor.SetMapper(self.mapper)
      self.cutterActor.SetPickable(1)

      # create cutter
      self.cutter2=vtk.vtkCutter()
      self.mapper2=vtk.vtkPolyDataMapper()
      self.mapper2.SetInputConnection(self.cutter2.GetOutputPort())
      self.mapper2.ScalarVisibilityOff()
      self.cutterActor2=vtk.vtkActor()
      self.cutterActor2.GetProperty().SetColor(1,1,0)
      self.cutterActor2.SetMapper(self.mapper2)
      self.cutterActor2.SetPickable(0)      
      self.picker=vtk.vtkPicker()
      self.elec=[]    
   def MiddleButtonOn(self,obj,event):
        self.OnMiddleButtonDown()
        self.DetectMouseMove = 1
   def MiddleButtonOff(self,obj,event):
        self.OnMiddleButtonUp()
        self.DetectMouseMove = 0        
   def ConfirmPatch(self,obj,event):
       if self.GetInteractor().GetKeyCode()=="n":
          self.TemPnormals, self.TemPdipCenters, self.TemPcellArea, self.TemPnumOfCells=PrepareData(self.GetInteractor())
          self.TemPdata=EEG.EEGData(self.TemPdipCenters,self.TemPnormals,self.TemPcellArea,self.GetInteractor().TemporalData[0:len(self.TemPdipCenters)],self.TemPnumOfCells,3)
          self.figure=plt.figure()
          plt.hold(False)
          plt.ylabel('Amplitude mV')
          plt.xlabel('Time s')
   def Print1(self,obj,event):
       global modeEEG
       if self.DetectMouseMove:
          #start_time = time.time()
          self.TemPdata.setSenCenters(self.GetInteractor().centers[0][5])
          V = self.TemPdata.CalculateOne()
          plt.plot(V)
          plt.show()    
          #44plt.ylim(-0.03,0.03)
          # print("--- %s seconds ---" % (time.time() - start_time))
   def Print2(self,obj,event):
       global modeEEG
       #start_time = time.time()
       self.TemPdata.setSenCenters(self.GetInteractor().centers[0][5])
       V = self.TemPdata.CalculateOne()
       plt.plot(V)
       plt.show()    
       #44plt.ylim(-0.03,0.03)
       # print("--- %s seconds ---" % (time.time() - start_time))
      
      
   def MouseMoveOK(self,obj,event):
        self.OnLeftButtonDown()
        self.DetectMouseMove = 1
   def MouseMoveNot(self,obj,event):
        self.OnLeftButtonUp()
        self.DetectMouseMove = 0
   def MouseWheelForward(self,obj,event):
            centers=[]
            electrode= self.actAssembly[self.elec]
            a=np.asarray(self.GetInteractor().centers[self.elec][2])-np.asarray(self.GetInteractor().centers[self.elec][3])   
            position=electrode.GetPosition()
            electrode.SetPosition(position[0]+a[0]/7,position[1]+a[1]/7,position[2]+a[2]/7)
            self.GetInteractor().GetRenderWindow().Render()
            
            # update centers
            g=self.actAssembly[self.elec].GetMatrix()             
            h = vtk.vtkTransform()
            h.SetMatrix(g)     
            v=self.Actors
            for i in range(0,len(v)):
              v[i].SetUserTransform(h)
              centers.append(v[i].GetCenter())
            k=h.Inverse()
            for i in range(0,len(v)):          
              v[i].SetUserTransform(k)
            if self.elec<len(self.GetInteractor().centers) :
              self.GetInteractor().centers[self.elec]=centers
            else:
              self.GetInteractor().centers.append(centers)
                        
    
   def MouseWheelBackward(self,obj,event):
            centers=[]
            electrode= self.actAssembly[self.elec]
            a=np.asarray(self.GetInteractor().centers[self.elec][3])-np.asarray(self.GetInteractor().centers[self.elec][2])   
            position=electrode.GetPosition()
            electrode.SetPosition(position[0]+a[0]/7,position[1]+a[1]/7,position[2]+a[2]/7)
            self.GetInteractor().GetRenderWindow().Render()
            
            # update centers
            g=self.actAssembly[self.elec].GetMatrix()             
            h = vtk.vtkTransform()
            h.SetMatrix(g)     
            v=self.Actors
            for i in range(0,len(v)):
              v[i].SetUserTransform(h)
              centers.append(v[i].GetCenter())
            k=h.Inverse()
            for i in range(0,len(v)):          
              v[i].SetUserTransform(k)
            if self.elec<len(self.GetInteractor().centers) :
              self.GetInteractor().centers[self.elec]=centers
            else:
              self.GetInteractor().centers.append(centers)
            
   def MouseMove(self,obj,event):
      global modeEEG                     
      if self.DetectMouseMove:
          centers=[]
          self.OnMouseMove()
          x,y=self.GetInteractor().GetEventPosition()
          self.picker.Pick(x, y, 0, self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
          
          if hasattr(self.picker.GetAssembly(),'id'):
              #elec=getattr(self.picker.GetAssembly(),'id')
              elec=self.actAssembly.index(self.picker.GetAssembly())
              self.elec=elec
              self.actAssembly[elec].SetScale([1,1,1])      
              g=self.actAssembly[elec].GetMatrix()             
              h = vtk.vtkTransform()
              h.SetMatrix(g)     
              v=self.Actors
              for i in range(0,len(v)):
                  v[i].SetUserTransform(h)
                  centers.append(v[i].GetCenter())
              k=h.Inverse()
              for i in range(0,len(v)):          
                  v[i].SetUserTransform(k)
              if elec<len(self.GetInteractor().centers) :
                  self.GetInteractor().centers[elec]=centers
              else:
                  self.GetInteractor().centers.append(centers)
              #self.actAssembly.SetUserTransform(h)
              #print centers
              """center=centers[1]
              a=np.asarray(centers[1])-np.asarray(centers[2])
              
              # construct first plane
              normal=[-a[1],a[0],0] 
              normal=normal/np.linalg.norm(normal)
              planeSource=vtk.vtkPlane()
              planeSource.SetOrigin(center)
              planeSource.SetNormal(normal)     
              self.cutter.SetInput(self.GetInteractor().PolyData)
              self.cutter.SetCutFunction(planeSource)
              #self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.cutterActor)
              
              w=1.01
              # add Volumetric Planes
              plane3 = vtk.vtkPlane()
              #c=self.GetInteractor().VolumeMapper.GetCenter()
              plane3.SetOrigin(center)
              plane3.SetNormal([normal[0],normal[1],normal[2]])        
              plane4 = vtk.vtkPlane()
              plane4.SetOrigin([center[0]+normal[0]*w,center[1]+normal[1]*w,center[2]+normal[2]*w])
              plane4.SetNormal([-normal[0],-normal[1],-normal[2]])
            
              #Volume Planes
              planes=vtk.vtkPlaneCollection()
              planes.AddItem(plane3)  
              planes.AddItem(plane4)  
              self.GetInteractor().VolumeMapper2.SetClippingPlanes(planes) 
                 
              # construct second plane 
              normal=[-a[2]*a[0],-a[1]*a[2],a[0]**2+a[1]**2] 
              normal=normal/np.linalg.norm(normal)
              planeSource2=vtk.vtkPlane()
              planeSource2.SetOrigin(center)
              planeSource2.SetNormal(normal)
              self.cutter2.SetInput(self.GetInteractor().PolyData)
              self.cutter2.SetCutFunction(planeSource2)
              #self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.cutterActor2)     
              
              # add Volumetric Planes
              plane1 = vtk.vtkPlane()
              #c=self.GetInteractor().VolumeMapper.GetCenter()
              plane1.SetOrigin(center)
              plane1.SetNormal([normal[0],normal[1],normal[2]])   
              plane2 = vtk.vtkPlane()
              plane2.SetOrigin([center[0]+normal[0]*w,center[1]+normal[1]*w,center[2]+normal[2]*w])
              plane2.SetNormal([-normal[0],-normal[1],-normal[2]])
            
              #Volume Planes
              planes=vtk.vtkPlaneCollection()
              planes.AddItem(plane1)  
              planes.AddItem(plane2)  
              self.GetInteractor().VolumeMapper.SetClippingPlanes(planes) """
              
              # add some effects
              self.GetInteractor().BA.GetProperty().SetOpacity(1)
              #self.GetInteractor().BA.GetProperty().SetRepresentationToWireframe()
              self.GetInteractor().GetRenderWindow().Render()
              self.HighlightProp(None)   
              #thread = Thread(target = self.CalculationThread, args = (centers, ))
              #thread.start()
              #thread.join()
   def CalculationThread(self,centers):
        normals, dipCenters, cellArea, numOfCells=PrepareData(self.GetInteractor())
        data=EEG.EEGData(dipCenters,normals,cellArea,self.GetInteractor().TemporalData[0:len(dipCenters)],numOfCells,modeEEG)
        data.setSenCenters(centers[1])
        v=data.Vj()
        return v
class Electrode():
    def __init__(self,NumOfSensors=15,Diameter=0.8,Resolution=36):
        self.NOS=NumOfSensors
        self.Diameter=Diameter
        self.Resolution=Resolution
    def Create(self):  
        SensorLength=2
        mm=1
        cylActors=[]
        
        # design tube holding electrodes
        tube=vtk.vtkCylinderSource()
        tube.SetHeight(((self.NOS*2+(self.NOS-1)*1.5)-0.1)*mm)
        tube.SetCenter(0,(self.NOS*2+(self.NOS-1)*1.5)*mm/2,0)
        tube.SetRadius((self.Diameter-0.05)*mm/2)
        tube.SetResolution(self.Resolution)
        tubePoly=tube.GetOutput()                       
        tubePoly.Update() 
        TubeMT=vtk.vtkPolyDataMapper()
        TubeMT.SetInput(tubePoly)
        TubeMT.GlobalImmediateModeRenderingOn()
        TubeA=vtk.vtkLODActor()
        TubeA.VisibilityOn()
        TubeA.SetMapper(TubeMT)
        TubeA.GetProperty().SetColor(0,0,1)
        cylActors.append(TubeA)
        
        # create the Electrodes
        for i in xrange(self.NOS):
            # create cylinder
            cyl=vtk.vtkCylinderSource()
            cyl.SetHeight(SensorLength*mm)            
            cyl.SetCenter(0,(1+i*(2+1.5))*mm,0)
            cyl.SetRadius(self.Diameter/2*mm)
            cyl.SetResolution(self.Resolution)           
            cylPoly=cyl.GetOutput()                       
            cylPoly.Update() 
            # create mappers and actors for the sensors
            cMT=vtk.vtkPolyDataMapper()
            cMT.SetInput(cylPoly)
            cMT.GlobalImmediateModeRenderingOn()
            cA=vtk.vtkLODActor()
            cA.VisibilityOn()
            cA.SetMapper(cMT)
            if i==0:
                cA.GetProperty().SetColor(0,1,0)

            else:
                cA.GetProperty().SetColor(1,0,0)

            cylActors.append(cA)           
        return cylActors


def SelectStyle(self,obj):
  global modeEEG
  if self.GetKeyCode()=="1":   
    self.PatchConfirmed=not(self.PatchConfirmed)
    self.BM.SetScalarModeToUsePointData()
    self.BA.SetPickable(1)
    #self.BA.GetProperty().SetOpacity(1)
    self.SetInteractorStyle(self.highlight)
    self.SinglePatch=Patch(3)
    self.GlyphCollection.append(self.SinglePatch.glyphActor)
    self.highlight.Create()   
    self.DrawN[1]=0
    self.BA.GetProperty().SetOpacity(1)
    self.BA.GetProperty().SetRepresentationToSurface()

  if self.GetKeyCode()=="0":
    self.BM.SetScalarModeToUseCellFieldData()
    self.BA.SetPickable(0)
    self.SetInteractorStyle(self.ActorSt)  

  if self.GetKeyCode()=="2":
    self.PatchConfirmed=not(self.PatchConfirmed)

    self.BM.SetScalarModeToUseCellFieldData()  
    self.BA.SetPickable(1) 
    self.SinglePatch=Patch(1)
    #self.BA.GetProperty().SetOpacity(1)
 
    #self.AreaFirstTime = not(self.AreaFirstTime)
    if not self.AreaFirstTime:
        self.AreaStyle.SetReqArea(170)
        self.SinglePatch.setPatchColor((0,1,1))
    else:
        self.AreaStyle.SetReqArea(170)
        self.SinglePatch.setPatchColor((0,1,1))
    if self.AreaFirstTime:
        self.AreaFirstTime = 0
        
    self.SetInteractorStyle(self.AreaStyle)
    self.GlyphCollection.append(self.SinglePatch.glyphActor)
    self.AreaStyle.Create()
    self.DrawN[0]=0
  if self.GetKeyCode()=="4":
    self.PatchConfirmed=not(self.PatchConfirmed)
    self.BM.SetScalarModeToUseCellFieldData()
    self.BA.SetPickable(1)
    #self.BA.GetProperty().SetOpacity(1)
    self.SetInteractorStyle(self.free)
    self.SinglePatch=Patch(2)    
    self.GlyphCollection.append(self.SinglePatch.glyphActor)
    self.free.Create()
    self.DrawN[2]=0
  if self.GetKeyCode()=="3":      
    d=self.GetInteractorStyle()
    self.DrawN[0]= not self.DrawN[0]
    if self.DrawN[0]:
          for i in range(len(self.PatchesCollection)):
              poly =vtk.vtkDataSetSurfaceFilter()
              poly.SetInput(self.PatchesCollection[i].GetMapper().GetInput())
              poly.Update()
              DrawNormals(d,poly,i)
         
    else:
          for i in range(len(self.PatchesCollection)):
              self.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.GlyphCollection[i])
 
  if self.GetKeyCode()=="8":

      normals, dipCenters, cellArea, numOfCells, vertices =PrepareData(self)
      
      sio.savemat('Ni.mat', {'Ni':normals})
      sio.savemat('DipoleCenters.mat', {'DipoleCenters':dipCenters})
      sio.savemat('Area.mat', {'Area':cellArea})
      sio.savemat('ElectrodeCenters.mat', {'ElectrodeCenters':self.centers})
      sio.savemat('vertices.mat', {'vertices':vertices})
      print len(vertices)
      print numOfCells
      print sum(cellArea)
      if len(dipCenters)>len(self.TemporalData):
          print "insufficient temporal data ..."
      else:
          data=EEG.EEGData(dipCenters,normals,cellArea,self.TemporalData[0:len(dipCenters)],numOfCells,modeEEG)          
          print len(self.TemporalData[0:len(dipCenters)])
          #for x in range(len(self.centers)): 
          for x in range(1):
              data.setSenCenters(self.centers[x])
              v=data.Vj()
              a=np.linspace(0,1,2*8192)
              fig=plt.figure()
              mycmap = cm.get_cmap('prism')
              fig.text(0.5, 0.98, 'Electrode '+str(x+1), horizontalalignment='center')
              for i in range(15):
                  c = mycmap(float(i)/(14))
                  fig.add_subplot(15,1,i+1)
                  plt.plot(a,v[i+1],color=c,label= 'E: '+repr(i+1))
                  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                  SaveLoad.SaveSig('Electrode '+str(x+1)+'mode'+str(data.mode)+str(i),v)
              plt.show()
              plt.legend()
            # collect garbage explicitly          
              gc.collect()
          data.job_server.destroy()  
  if self.GetKeyCode()=="5":
          self.BA.GetProperty().SetOpacity(0)
          modeEEG=1
  if self.GetKeyCode()=="6":
          modeEEG=2 
  if self.GetKeyCode()=="7":
          modeEEG=3
  if self.GetKeyCode()=="9":
          modeEEG=0     
  if self.GetKeyCode()=="+":     
     if  self.ActorVolume.GetVisibility()==0:
         self.ActorVolume.SetVisibility(1)      
     else:
         self.ActorVolume.SetVisibility(0)  
     self.GetRenderWindow().Render()
   
  if self.GetKeyCode()=="-":   
     if  self.ActorVolume2.GetVisibility()==1:
         self.ActorVolume2.SetVisibility(0) 
     else:
         self.ActorVolume2.SetVisibility(1) 
     self.GetRenderWindow().Render()
    
  if self.GetKeyCode()=="*": 
       self.AreaStyle.addElectrode=not(self.AreaStyle.addElectrode)
       if self.AreaStyle.addElectrode:
         self.BA.SetPickable(1)
         self.SetInteractorStyle(self.AreaStyle)
         elec1=Electrode(15,0.8,36)
         Actors=elec1.Create()
         mm=1
         sph=vtk.vtkSphereSource()       
         sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
         sph.SetRadius(0)
         sphPoly=sph.GetOutput()                       
         sphPoly.Update() 
         sMT=vtk.vtkPolyDataMapper()
         sMT.SetInput(sphPoly)
         sMT.GlobalImmediateModeRenderingOff()
         sA=vtk.vtkLODActor()
         sA.VisibilityOn()
         sA.SetMapper(sMT)
         actAssembly=vtk.vtkAssembly()
         for i in range(elec1.NOS+1):
                actAssembly.AddPart(Actors[i])
         actAssembly.AddPart(sA)
         
         actAssembly.id=len(self.ActorSt.actAssembly)             
         self.ActorSt.actAssembly.append(actAssembly)  
         self.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.ActorSt.actAssembly[actAssembly.id])
#         if len(self.ActorSt.actAssembly) >0:
#             self.ActorSt.cutterActor.SetVisibility(1)
#             self.ActorSt.cutterActor2.SetVisibility(1)
#             if  self.ActorVolume.GetVisibility()==0: 
#                 self.ActorVolume.SetVisibility(0)
#             else:
#                 self.ActorVolume.SetVisibility(1)
#             if  self.ActorVolume2.GetVisibility()==0: 
#                 self.ActorVolume2.SetVisibility(0)
#             else:
#                 self.ActorVolume2.SetVisibility(1) 
         self.labelPoints.InsertNextPoint(actAssembly.GetCenter()) 
         self.labelScalars.InsertNextTuple1(actAssembly.id) 
         self.BA.GetProperty().SetOpacity(1)


       else:
          self.SetInteractorStyle(self.ActorSt)          
          self.BA.SetPickable(0)
  if  self.GetKeyCode()=="l":
      # toggle the visibilty off the Labels actors 
      for i in xrange(len(self.isoLabels)):
          labelActor=self.isoLabels[i]
          labelActor.SetVisibility(not(self.isoLabels.GetVisibility()))
      self.GetRenderWindow().Render()

  if  self.GetKeyCode()=="d":
      if hasattr(self.ActorSt.picker.GetAssembly(),'id'):
          elec=self.ActorSt.actAssembly.index(self.ActorSt.picker.GetAssembly())
          self.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.ActorSt.actAssembly[elec])
          self.ActorSt.actAssembly.remove(self.ActorSt.actAssembly[elec])
          self.centers.remove(self.centers[elec][:])
          if len(self.ActorSt.actAssembly) ==0:
              self.ActorSt.cutterActor.SetVisibility(0)
              self.ActorSt.cutterActor2.SetVisibility(0)
              #self.ActorVolume.SetVisibility(0)
              #self.ActorVolume2.SetVisibility(0)
          self.GetRenderWindow().Render()
      
      self.BA.SetPickable(0)
        
      for a in range(len(self.PatchesCollection)):
          self.PatchesCollection[a].SetPickable(0)
      x,y=self.GetEventPosition()
      self.ActorSt.picker.Pick(x, y, 0, self.GetRenderWindow().GetRenderers().GetFirstRenderer())
      if self.ActorSt.picker.GetActor() in self.PatchesCollection:
          patch=self.PatchesCollection.index(self.ActorSt.picker.GetActor())
          self.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.PatchesCollection[patch])
          if self.DrawN[0]:
              self.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.GlyphCollection[patch])
              self.GlyphCollection.remove(self.GlyphCollection[patch])   
          self.PatchesCollection.remove(self.PatchesCollection[patch])
      self.GetRenderWindow().Render()   
      for a in range(len(self.PatchesCollection)):
          self.PatchesCollection[a].SetPickable(0)
  if self.GetKeyCode()==".":
     if  self.headActor.GetVisibility()==1:
         self.headActor.SetVisibility(0) 
         self.LhippoActor.SetVisibility(0)
         self.RhippoActor.SetVisibility(0)
     else:
         self.headActor.SetVisibility(1)
         self.LhippoActor.SetVisibility(1)
         self.RhippoActor.SetVisibility(1)
     self.GetRenderWindow().Render()
def PrepareData(iren):
    poly =vtk.vtkDataSetSurfaceFilter()
    # Compute Normals of selected patch
    normalsCalc = vtk.vtkPolyDataNormals()
    cellCenters=vtk.vtkCellCenters()
    normals=[]
    dipCenters=[]
    cellArea=[]
    vertices = []
    numOfCells=[0]*(len(iren.PatchesCollection)+1)
    for j in range (len(iren.PatchesCollection)):
       poly.SetInput(iren.PatchesCollection[j].GetMapper().GetInput())
       poly.Update()
       normalsCalc.SetInputConnection(poly.GetOutputPort())
       normalsCalc.ComputePointNormalsOn()
       normalsCalc.ComputeCellNormalsOff()
       normalsCalc.SplittingOff()
       normalsCalc.FlipNormalsOn()
       normalsCalc.AutoOrientNormalsOff()
       normalsCalc.Update()
       cellCenters.VertexCellsOn()
       cellCenters.SetInputConnection(normalsCalc.GetOutputPort())
       cellCenters.Update()
       array=normalsCalc.GetOutput().GetPointData().GetNormals()
       for k in range(poly.GetOutput().GetPoints().GetNumberOfPoints ()):
           vertices.append(poly.GetOutput().GetPoint(k))           
       for i in range(array.GetNumberOfTuples()):
           numOfCells[j+1]=numOfCells[j+1]+1
           normals.append(array.GetTuple(i))
           dipCenters.append( cellCenters.GetOutput().GetPoint(i))
           #cellArea.append(poly.GetOutput().GetCell(i).ComputeArea())
           
    return normals, dipCenters, cellArea, numOfCells, vertices
def ReadEEGfile(filename):
    name, ext=os.path.splitext(filename)    
    # read descriptive file (.des)  into fs, nChannels, nSamples
    data=open(name+'.des','r').readlines()[4]
    fs=float(data.split()[1])
    data=open(name+'.des','r').readlines()[7]
    nSamples=int(data.split()[1])
    data=open(name+'.des','r').readlines()[10]
    nChannels=int(data.split()[1])
    T0=0
    # read (.bin)
    if ext=='.bin':
        fin=open(filename,'rb')         
        x=array.array('f')
        x.read(fin,nSamples*nChannels)
        start=T0*fs*nChannels
        sig=[0 for i in range(nSamples-T0*fs)]
        for i in range(nChannels):
            sig[i]=x[start+i::nChannels]             
        return sig   
    # read (.dat)
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

def SelectPolygons(object, event):
    # object will be the boxWidget
    global selectActor, planes
    rep=object.GetRepresentation()
    rep.GetPlane(planes)
    selectActor.VisibilityOn()
    selectActor.GetProperty().SetColor(0.9, 0.9, 0.9)
    selectActor.SetScale(1., 1., 1.)
    object.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(selectActor)
    
def main():
    global modeEEG
    modeEEG=0
#--------------- read brain file using the given file path---------------------
    
    #filePath="C:\Users\Mohamad Shamas\Downloads\These\mesh\RET_mid_iso2mesh_10001V_bis_rotz90y-17.vtk"
    #filePath= "C:\Users\Mohamad Shamas\Desktop\scalp-HFO-simulations\mesh\MFM_mid_iso2mesh_15000V.vtk"
    #filePath= "C:\Users\Mohamad Shamas\Desktop\scalp-HFO-simulations\RET_sim_mid_100001V_iso2mesh.vtk"
    filePath = "D:\LEN-simul-new\mesh\LEN_white_100000_new.vtk"
    #filePath="C:\Users\Mohamad Shamas\Downloads\These\mesh\RET_mid_iso2mes1h_10001V.vtk"
    #filePath="C:\Users\Mohamad Shamas\Desktop\EMBC\MET_cortex_293095V.vtk"
    #filePath="C:\Users\Mohamad Shamas\Downloads\These\mesh\RET_sim_cortex_15002V_rotz90y-17.vtk"
    brain=vtk.vtkDataSetReader()
    brain.SetFileName(filePath)
    brain.ReadAllScalarsOn()
    brain.Update()
    
    
    
#    transformFilter = vtk.vtkTransformPolyDataFilter()
#    transformFilter.SetInputConnection(brain.GetOutputPort())
#    h = vtk.vtkTransform()
#    m=np.array([[1, 0, 0, 0],[0, 1, 0, -255],[0, 0, -1, 255],[0, 0, 0, 1]])
#    mat=vtk.vtkMatrix4x4()
#    for i in np.ndindex(4,4):
#        mat.SetElement(i[0],i[1],m[i])
#    h.SetMatrix(mat)
#    transformFilter.SetTransform(h)
#    transformFilter.Update()
#    brain=transformFilter
    
    
#    filePath="C:\\Users\\Mohamad Shamas\\Desktop\\999\\mesh\\ref_t1mri\\999_cortex_gray_pial_left.vtk"
#    brainL=vtk.vtkDataSetReader()
#    brainL.SetFileName(filePath)
#    brainL.ReadAllScalarsOn()
#    brainL.Update()
#    filePath="C:\\Users\\Mohamad Shamas\\Desktop\\999\\mesh\\ref_t1mri\\999_cortex_gray_pial_right.vtk"
#    brainR=vtk.vtkDataSetReader()
#    brainR.SetFileName(filePath)
#    brainR.ReadAllScalarsOn()
#    brainR.Update()
#    
#    brain =vtk.vtkAppendPolyData()
#    brain.AddInput(brainL.GetOutput())
#    brain.AddInput(brainR.GetOutput())
#    brain.Update()
    

    
#******************************************************************************
 
# subdivide the mesh into a smoother one
#    v = vtk.vtkLoopSubdivisionFilter()
#    v.SetNumberOfSubdivisions(1)
#    v.SetInputConnection(brain.GetOutputPort())
#    v.Update()
#    brain=v
 

   
#----------------------------------Id Filter-----------------------------------
    idFilter=vtk.vtkIdFilter()
    idFilter.SetInput(brain.GetOutput())
    idFilter.SetIdsArrayName("OriginalIds")
    idFilter.Update()
#******************************************************************************    

    
#-------------------------------Surface Filter---------------------------------
    surfaceFilter=vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputConnection(idFilter.GetOutputPort())
    surfaceFilter.Update()
    inputs = surfaceFilter.GetOutput()
#******************************************************************************    


#-------------------create a mapper and actor for brain------------------------
    brainMapper=vtk.vtkPolyDataMapper()
    brainMapper.SetInputConnection(brain.GetOutputPort())
    brainMapper.SetScalarModeToUseCellFieldData()
    brainMapper.GlobalImmediateModeRenderingOn()
    brainActor = vtk.vtkLODActor()
    brainActor.GetProperty().SetOpacity(0.1)
    #brainActor.GetProperty().SetRepresentationToWireframe()
    brainActor.SetMapper(brainMapper)
    brainActor.SetPickable(0) #prevent brain from being panned 
       
#******************************************************************************
    

#-------------------create electrode with balancing sphere---------------------
    elec1=Electrode(15)
    Actors=elec1.Create()
    mm=1
    sph=vtk.vtkSphereSource()       
    sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
    sph.SetRadius(0)
    sphPoly=sph.GetOutput()                       
    sphPoly.Update() 
    sMT=vtk.vtkPolyDataMapper()
    sMT.SetInput(sphPoly)
    sMT.GlobalImmediateModeRenderingOff()
    sA=vtk.vtkLODActor()
    sA.VisibilityOn()
    sA.SetMapper(sMT)
#******************************************************************************    


#---------------- fill up the actAssembly with electrodes actors---------------
    actAssembly=vtk.vtkAssembly()
    for i in range(elec1.NOS+1):
        actAssembly.AddPart(Actors[i])
    actAssembly.AddPart(sA)
    actAssembly.id=0
    actAssemblys=[]
    actAssemblys.append(actAssembly)
#******************************************************************************    


#---------------set default position of electrode and orientation--------------
    n=[0,0,0]
    dCell=brain.GetOutput().GetCell(5600)
    dPoints=dCell.GetPoints()
    dCell.ComputeNormalDirection(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)   
    actAssembly.SetOrientation(math.degrees(math.atan(n[2]/n[1])),0,-math.degrees(math.atan(n[0]/n[1])))  
    #n=[0,-19,55]
    actAssembly.SetOrientation(math.degrees(math.atan(n[2]/(n[1]+0.000001)))+180,0,math.degrees(math.atan(n[0]/(n[1]+0.000001)) ))  
    dCell.TriangleCenter(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)
    actAssembly.SetPosition(n[0],n[1],n[2])
    #actAssembly.SetPosition(97,162,154)
    
#******************************************************************************   
    
   
#---------------- create renderer, render window and interactor----------------
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()    
    renWin.AddRenderer(ren)
    iren.SetRenderWindow(renWin)
    
#******************************************************************************      



#------------------------------ Highlight Style--------------------------------
    st=HighlightStyle()
    st.SetPolyData(inputs)
    iren.highlight=st
#******************************************************************************       
    
#---------------------------------Free Style-----------------------------------
    free=FreeSelection(brain)  
    iren.free=free       
#******************************************************************************

#---------------------------------Area Style----------------------------------
    AS=AreaStyle(brain,1000)
    AS.addElectrode=0
    iren.AreaStyle=AS
    
#******************************************************************************    

#--------------------------------Actor Style-----------------------------------
    style=InteractorStyleTrackballActor(Actors,actAssemblys)
    iren.ActorSt=style 
    iren.centers=[]
    labelPolyData=vtk.vtkPolyData()    
    labelPoints=vtk.vtkPoints()
    labelScalars=vtk.vtkDoubleArray()
    labelScalars.SetNumberOfComponents(1)
    labelScalars.SetName("ElecNum")
    
    iren.labelPolyData = labelPolyData
    iren.labelPoints=labelPoints
    iren.labelScalars=labelScalars
    
    iren.labelPolyData.SetPoints(iren.labelPoints)
    iren.labelPolyData.GetPointData().SetScalars(iren.labelScalars)
    
    iren.labelPoints.InsertNextPoint(actAssemblys[0].GetCenter())
    iren.labelScalars.InsertNextTuple1(1)
    

    
    labelMapper=vtk.vtkLabeledDataMapper()
    labelMapper.SetInput(iren.labelPolyData)
    labelMapper.SetLabelModeToLabelScalars()
    labelMapper.SetLabelFormat("E:%6.0f")
    isoLabels = vtk.vtkActor2D()
    isoLabels.SetMapper(labelMapper)
    iren.isoLabels=[]
    iren.isoLabels.append(isoLabels)
#******************************************************************************


    
#----------add the actors to the renderer, set the background and size---------
    
    ren.AddActor(brainActor)
    ren.AddActor(actAssemblys[0])      
    ren.SetBackground(1, 1, 1)
   #ren.AddActor(isoLabels)
    renWin.SetSize(1000, 800)
#******************************************************************************    
    
    
#---------------- get the camera and zoom in closer to the image---------------
    ren.ResetCamera()
    cam = ren.GetActiveCamera()   
    cam.Zoom(1)
#******************************************************************************    


#---------------------------------Add iren attributes--------------------------
    iren.DrawN=[0,0,0]
    iren.BM=brainMapper
    iren.BA=brainActor
    iren.PolyData=inputs
    iren.SetInteractorStyle( style )
    iren.AddObserver("KeyPressEvent",SelectStyle)
    iren.PatchConfirmed=0 
    iren.PatchesCollection=[] 
    iren.GlyphCollection=[]     
    iren.AreaFirstTime=1
#******************************************************************************


#----------------------- Read Volume File (.mgz) ------------------------------
    fname="C:\Users\Mohamad Shamas\Downloads\These\\brain.mgz"
    image=MGH()
    image.load(fname)
    h=image.vol
    h=np.require(h[:,:,:],dtype=np.uint8)
    dataImporter=vtk.vtkImageImport()
    data_string=h.tostring()
    dataImporter.CopyImportVoidPointer(data_string,len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    w, d, hi = h.shape
    dataImporter.SetDataExtent(0, hi-1, 0, d-1, 0, w-1)
    dataImporter.SetWholeExtent(0, hi-1, 0, d-1, 0, w-1)
    
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    colorFunc = vtk.vtkColorTransferFunction()
    
    for i in range(np.max(h)):                 
        if i<40:
           alphaChannelFunc.AddPoint(i, 0)
        else:
           alphaChannelFunc.AddPoint(i, 0.8)
    
    colorFunc.AddRGBPoint(40,0.0,0,0.5)  
    colorFunc.AddRGBPoint(50,0.,0.,0.9444)
    colorFunc.AddRGBPoint(60,0.0,0.39,1)  
    colorFunc.AddRGBPoint(70,0.833,0,1)
    colorFunc.AddRGBPoint(80,0.27,1,0.72)    
    colorFunc.AddRGBPoint(100,0.72,1.0,0.27)  
    colorFunc.AddRGBPoint(110,1,0.833,0)  
    colorFunc.AddRGBPoint(120,1,0.39,0)
    colorFunc.AddRGBPoint(130,0.94,0,0.0)  
    colorFunc.AddRGBPoint(140,0.5,0,0)
    colorFunc.AddRGBPoint(256,1.0,1.0,1.0)  
    colorFunc.AddRGBPoint(0,0,0,0)    
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    mapVolume = vtk.vtkVolumeRayCastMapper()
    funcRayCast = vtk.vtkVolumeRayCastCompositeFunction()
    funcRayCast.SetCompositeMethodToClassifyFirst()    
    mapVolume.SetVolumeRayCastFunction(funcRayCast)
    mapVolume.SetInput(dataImporter.GetOutput())
    actorVolume = vtk.vtkVolume()
    actorVolume.SetMapper(mapVolume)
    actorVolume.SetProperty(volumeProperty)
    trans = vtk.vtkTransform()
    rotx=-90
    rotz=17.5
    x=-112; y=-150; z=-125.54
    trans.RotateX(rotx)
    trans.RotateZ(rotz)
    trans.Translate(x,y,z)
    actorVolume.SetUserTransform(trans)
    actorVolume.SetPickable(0)
    ren.AddActor(actorVolume)
    actorVolume.SetVisibility(0)
    iren.ActorVolume=actorVolume
    
    #second plane
    mapVolume2 = vtk.vtkVolumeRayCastMapper()
    mapVolume2.SetVolumeRayCastFunction(funcRayCast)
    mapVolume2.SetInput(dataImporter.GetOutput())
    actorVolume2 = vtk.vtkVolume()
    actorVolume2.SetMapper(mapVolume2)
    actorVolume2.SetProperty(volumeProperty)
    trans = vtk.vtkTransform()
    trans.RotateX(rotx)
    trans.RotateZ(rotz)
    trans.Translate(x,y,z)
    actorVolume2.SetUserTransform(trans)
    actorVolume2.SetPickable(0)
    ren.AddActor(actorVolume2)
    actorVolume2.SetVisibility(0)
    iren.ActorVolume2=actorVolume2
    iren.VolumeMapper=mapVolume
    iren.VolumeMapper2=mapVolume2
#******************************************************************************
#--------------------------- Read Segmented Organs ----------------------------
    ReadSegmentedOrgans(iren)
    ren.AddActor(iren.headActor)
    ren.AddActor(iren.LhippoActor)
    ren.AddActor(iren.RhippoActor)

#------------------------------------------------------------------------------    

#--------------------------- Read Binary File ---------------------------------    
    #filename='D:\\ZZZ_JournalSimulations\\AutoSimulations6\\jitter130\\SIM_A_0.05_J_1_I2_1_I3_6.dat'
    #filename='C:\Users\\Mohamad Shamas\\Desktop\\Simulation -Sesame\\Gaussian\\GaussianSpikes41.dat'
    filename='D:\Simulated Signals\\500-LowDeSync-FullRandomness-1024Hz\\500-LowDeSync-FullRandomness-1024Hz.dat'
    filename2='D:\Simulated Signals\\500-MediumSync-FullRandomness-1024Hz\\500-MediumSync-FullRandomness-1024Hz.dat'
    filename3='D:\Simulated Signals\\500-ExtremelyDesencronized-FullRandomness-1024Hz\\500-ExtremelyDesencronized-FullRandomness-1024Hz.dat'
    s=ReadEEGfile(filename)
#    g=[]
#    for i in xrange(500): 
#        g.append(np.array(s[i],dtype=np.float32))
#    g=np.array(g)
    iren.TemporalData=s.T
    
    import scipy.io as sio
    sio.savemat('np_vector.mat', {'vect':s.T})
    
#******************************************************************************
#   
##***********************Add 4 electrodes temporal part*************************
#    elec1=Electrode(15)
#    Actors=elec1.Create()
#    mm=1
#    sph=vtk.vtkSphereSource()       
#    sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
#    sph.SetRadius(0)
#    sphPoly=sph.GetOutput()                       
#    sphPoly.Update() 
#    sMT=vtk.vtkPolyDataMapper()
#    sMT.SetInput(sphPoly)
#    sMT.GlobalImmediateModeRenderingOff()
#    sA=vtk.vtkLODActor()
#    sA.VisibilityOn()
#    sA.SetMapper(sMT)
#    actAssembly=vtk.vtkAssembly()
#    for i in range(elec1.NOS+1):
#        actAssembly.AddPart(Actors[i])
#    actAssembly.AddPart(sA)
#    actAssembly.id=len(iren.ActorSt.actAssembly) 
#    #---------------set default position of electrode and orientation--------------
#    n=[0,0,0]
#    dCell=brain.GetOutput().GetCell(5000)
#    dPoints=dCell.GetPoints()
#    dCell.ComputeNormalDirection(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)   
#    #actAssembly.SetOrientation(math.degrees(math.atan(n[2]/n[1])),0,-math.degrees(math.atan(n[0]/n[1])))  
#    n=[-133+86,-161+161,-187+187]
#    actAssembly.SetOrientation((180),0,-90)  
#   
#    dCell.TriangleCenter(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)
#    #actAssembly.SetPosition(n[0],n[1],n[2])
#    actAssembly.SetPosition(133,161,187)
##******************************************************************************   
#    iren.ActorSt.actAssembly.append(actAssembly)
#    ren.AddActor(iren.ActorSt.actAssembly[1])
##------------------------------------------------------------------------------  
#  #***********************Add 4 electrodes temporal part*************************
#    elec2=Electrode(15)
#    Actors=elec2.Create()
#    mm=1
#    sph=vtk.vtkSphereSource()       
#    sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
#    sph.SetRadius(0)
#    sphPoly=sph.GetOutput()                       
#    sphPoly.Update() 
#    sMT=vtk.vtkPolyDataMapper()
#    sMT.SetInput(sphPoly)
#    sMT.GlobalImmediateModeRenderingOff()
#    sA=vtk.vtkLODActor()
#    sA.VisibilityOn()
#    sA.SetMapper(sMT)
#    actAssembly=vtk.vtkAssembly()
#    for i in range(elec1.NOS+1):
#        actAssembly.AddPart(Actors[i])
#    actAssembly.AddPart(sA)
#    actAssembly.id=len(iren.ActorSt.actAssembly) 
#    #---------------set default position of electrode and orientation--------------
#    n=[0,0,0]
#    dCell=brain.GetOutput().GetCell(5000)
#    dPoints=dCell.GetPoints()
#    dCell.ComputeNormalDirection(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)   
#    #actAssembly.SetOrientation(math.degrees(math.atan(n[2]/n[1])),0,-math.degrees(math.atan(n[0]/n[1])))  
#    n=[-127+92,-161+161,-196+197]
#    actAssembly.SetOrientation((180),0,-90)  
#   
#    dCell.TriangleCenter(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)
#    #actAssembly.SetPosition(n[0],n[1],n[2])
#    actAssembly.SetPosition(127,161,196)
##******************************************************************************   
#    iren.ActorSt.actAssembly.append(actAssembly)
#    ren.AddActor(iren.ActorSt.actAssembly[2])
##------------------------------------------------------------------------------    
##------------------------------------------------------------------------------  
#  #***********************Add 4 electrodes temporal part*************************
#    elec2=Electrode(15)
#    Actors=elec2.Create()
#    mm=1
#    sph=vtk.vtkSphereSource()       
#    sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
#    sph.SetRadius(0)
#    sphPoly=sph.GetOutput()                       
#    sphPoly.Update() 
#    sMT=vtk.vtkPolyDataMapper()
#    sMT.SetInput(sphPoly)
#    sMT.GlobalImmediateModeRenderingOff()
#    sA=vtk.vtkLODActor()
#    sA.VisibilityOn()
#    sA.SetMapper(sMT)
#    actAssembly=vtk.vtkAssembly()
#    for i in range(elec1.NOS+1):
#        actAssembly.AddPart(Actors[i])
#    actAssembly.AddPart(sA)
#    actAssembly.id=len(iren.ActorSt.actAssembly) 
#    #---------------set default position of electrode and orientation--------------
#    n=[0,0,0]
#    dCell=brain.GetOutput().GetCell(5000)
#    dPoints=dCell.GetPoints()
#    dCell.ComputeNormalDirection(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)   
#    #actAssembly.SetOrientation(math.degrees(math.atan(n[2]/n[1])),0,-math.degrees(math.atan(n[0]/n[1])))  
#    n=[-132+80,-138+139,-189+191]
#    actAssembly.SetOrientation((180),0,-90)  
#   
#    dCell.TriangleCenter(dPoints.GetPoint(0),dPoints.GetPoint(1),dPoints.GetPoint(2),n)
#    #actAssembly.SetPosition(n[0],n[1],n[2])
#    actAssembly.SetPosition(132,138,189)
#    
#    
##******************************************************************************   
#    
#    
#    iren.ActorSt.actAssembly.append(actAssembly)
#    ren.AddActor(iren.ActorSt.actAssembly[3])
# 
##**********************make clippers ******************************************
#    global planes, selectActor
#    planes = vtk.vtkPlane()
#    clipper = vtk.vtkClipPolyData()
#    clipper.SetInput(brain.GetOutput())
#    clipper.SetClipFunction(planes)
#    clipper.InsideOutOn()
#    selectMapper = vtk.vtkPolyDataMapper()
#    selectMapper.SetInputConnection(clipper.GetOutputPort())
#    selectActor = vtk.vtkLODActor()
#    selectActor.SetMapper(selectMapper)
#    selectActor.GetProperty().SetColor(0.95, 0.95, 1)
#    selectActor.VisibilityOff()
#    selectActor.SetScale(1., 1., 1.)
#    selectActor.SetPickable(0)
#    boxWidget = vtk.vtkImplicitPlaneRepresentation()
#    #boxWidget.SetInteractor(iren)
#    boxWidget.SetPlaceFactor(1)
#    boxWidget.SetNormal(planes.GetNormal())
#    boxWidget.PlaceWidget(brainActor.GetBounds())
#    boxWidget.SetOrigin(brainActor.GetOrigin()[0]+128,brainActor.GetOrigin()[1]+128,brainActor.GetOrigin()[2]+128)
#    
#    wid = vtk.vtkImplicitPlaneWidget2()
#    #boxWidget.SetInput(brain.GetOutput())
#    wid.SetInteractor(iren)
#    wid.SetRepresentation(boxWidget)
#    wid.AddObserver("EndInteractionEvent", SelectPolygons) 
#    brainActor.GetProperty().SetOpacity(1)
    
##------------------------------------------------------------------------------  
    
#----------------------enable user interface interactor------------------------
    iren.Initialize()
    renWin.Render()
    iren.Start()
#******************************************************************************


    
if __name__ == "__main__": main()
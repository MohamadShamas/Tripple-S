# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:19:01 2015

@author: Mohamad Shamas
"""

import vtk

class AreaStyle(vtk.vtk.vtkInteractorStyleTrackballCamera):
    """ this class represent the selection style that depend on required area """   
    def __init__(self,brain,ReqArea=1000):
        
        # Needed Data
        self.RequiredArea=ReqArea
        self.Brain=brain
        self.MouseMove=0

        # Created objects for selection
        self.picker = vtk.vtkCellPicker()
        self.triangleFilter=vtk.vtkTriangleFilter()
        self.triangleFilter.SetInputConnection(self.Brain.GetOutputPort());
        self.triangleFilter.Update();
        self.cellPointIds=vtk.vtkIdList()
        self.neighborCellsMapper =vtk.vtkDataSetMapper()
        self.neighborCellsActor =vtk.vtkActor()
        self.selectionNode=vtk.vtkSelectionNode()
        self.selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL);
        self.selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES);
        self.selection=vtk.vtkSelection()
        self.selection.AddNode(self.selectionNode)
        self.exSelection=vtk.vtkExtractSelection()
        self.exSelection.SetInputConnection(0, self.Brain.GetOutputPort())
        self.exPoly =vtk.vtkDataSetSurfaceFilter()
        self.exPoly.SetInput(self.exSelection.GetOutput())
        self.area=vtk.vtkMassProperties()
        self.area.SetInputConnection(self.exPoly.GetOutputPort())      
        self.neighborCellIds =vtk.vtkIdList()            
        
        # Created objects for normals
        self.glyphMapper = vtk.vtkPolyDataMapper()
        self.glyphActor = vtk.vtkActor()
        self.glyphActor.SetMapper(self.glyphMapper)
        
        # Add observers for mouse button events        
        self.AddObserver("LeftButtonReleaseEvent",self.PickArea)
        self.AddObserver('LeftButtonPressEvent',self.set_mmZero)
        self.AddObserver('MouseMoveEvent',self.set_mmOne)
#------------------------------------------------------------------------------        
   
    def set_mmZero(self,obj,event):
      self.MouseMotion=0
      self.OnLeftButtonDown()
#------------------------------------------------------------------------------

    def set_mmOne(self,obj,event):
      self.MouseMotion=1
      self.OnMouseMove()
#------------------------------------------------------------------------------   

    def  PickArea(self,obj,event):
      self.OnLeftButtonUp()
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
                #Add the other edge points
                if i+1==self.cellPointIds.GetNumberOfIds():
                    idList.InsertNextId(self.cellPointIds.GetId(0))
                else:
                    idList.InsertNextId(self.cellPointIds.GetId(i+1))
                self.triangleFilter.GetOutput().GetCellNeighbors(neighbors[x], idList, self.neighborCellIds)                                
                for j in range(self.neighborCellIds.GetNumberOfIds()):
                    if self.neighborCellIds.GetId(j) not in neighbors:
                       neighbors.append(self.neighborCellIds.GetId(j))                
            #  Create a dataset with the neighbor cells
              ids=vtk.vtkIdTypeArray()
              ids.SetNumberOfComponents(1)         
              for it1 in range(len(neighbors)):
                ids.InsertNextValue(neighbors[it1])            
              self.selectionNode.SetSelectionList(ids);             
              self.exSelection.SetInput(1, self.selection)
              self.exSelection.Update()                    
              self.exPoly.Update()        
              a=self.area.GetSurfaceArea()+self.Brain.GetOutput().GetCell(cellId).ComputeArea()
              x=x+1               
        self.neighborCellsMapper.SetInputConnection(self.exSelection.GetOutputPort())
        self.neighborCellsActor.SetMapper(self.neighborCellsMapper)
        self.neighborCellsActor.GetProperty().SetColor(0,1,0)
        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.neighborCellsActor)
        self.GetInteractor().Render()
#------------------------------------------------------------------------------          
        
        
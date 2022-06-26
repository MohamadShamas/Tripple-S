#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __main__ import vtk, qt, ctk, slicer
import PythonQt
import math
import time
import numpy as np
import cPickle
import array
import os
import gc


#
# SimulatingSEEG
#

#------------------------------- GUI building ---------------------------------
#------------------------------------------------------------------------------

class SimulatingSEEG:
  def __init__(self, parent):
    parent.title = "SSS ---> |S|imulate |S|eeg |S|ignals"
    parent.categories = ["Simulation"]
    parent.dependencies = []
    parent.contributors = ["This tool was developed by Mohamad Shamas and supervised by: Isabelle Merlet ",
                           "Pascal Benquet",
                           "Fabrice Wendling. "] 
    parent.helpText = """
This program was developed in the aim of simulating EEG signals as collected from intracerebral electrodes. It consist of 3 main controls:
    1) Electrodes: Add electrodes, specify name, position and orientation either by entering coordinates or interactively.
    2) Neural Population Patch: Specify the neural population patch by 3 differnt manners.
    3) SEEG Simulation: Add neural population activity file, calculate Lead Field matrix and display simulated data.
    """
    parent.acknowledgementText = """ This work received support from the French ANR and GDOS (Project Vibrations, Programme de Recherche Translationelle en Sante, ANR-13-PRTS-0011).It was also supported by AZM and SAADE Association (Tripoli, Lebanon). """ 
    self.parent = parent

#
# SimulatingSEEGWidget
#

class SimulatingSEEGWidget():
  def __init__(self, parent = None):
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()
      
  def setup(self):
    global ElectrodeTable , NbPatches
    # Choose suitable slicer layout
    lns = slicer.mrmlScene.GetNodesByClass('vtkMRMLLayoutNode')
    lns.InitTraversal()
    ln = lns.GetNextItemAsObject()
    ln.SetViewArrangement(24)
    
    # Get the Chart View Node
    cvns = slicer.mrmlScene.GetNodesByClass('vtkMRMLChartViewNode')
    cvns.InitTraversal()
    self.cvn = cvns.GetNextItemAsObject()
    
    # Cortex mesh Collapsible button
    self.CortexMeshCollapsibleButton = ctk.ctkCollapsibleButton()
    self.CortexMeshCollapsibleButton.text = "CortexMesh"
    self.layout.addWidget(self.CortexMeshCollapsibleButton)

    # Layout within the Cortex collapsible button
    self.CortexFormLayout = qt.QFormLayout(self.CortexMeshCollapsibleButton)

    self.CortexMeshForm = qt.QFrame(self.CortexMeshCollapsibleButton)
    self.CortexMeshForm.setLayout(qt.QHBoxLayout())
    self.CortexFormLayout.addWidget(self.CortexMeshForm)
    inputModelSelectorLabel = qt.QLabel("Input Model: ",self.CortexMeshForm)
    inputModelSelectorLabel.setToolTip( "Select the input model")
    self.CortexMeshForm.layout().addWidget(inputModelSelectorLabel)
	
    self.inputModelSelector = slicer.qMRMLNodeComboBox(self.CortexMeshForm)
    self.inputModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.inputModelSelector.selectNodeUponCreation = False
    self.inputModelSelector.addEnabled = False
    self.inputModelSelector.removeEnabled = False
    self.inputModelSelector.noneEnabled = True
    self.inputModelSelector.showHidden = False
    self.inputModelSelector.showChildNodeTypes = False
    self.inputModelSelector.setMRMLScene( slicer.mrmlScene )
    self.CortexMeshForm.layout().addWidget(self.inputModelSelector)
    self.inputModelSelector.currentNodeChanged.connect(self.onNodeSelected)
    
    # Electrode Collapsible button
    self.electrodesCollapsibleButton = ctk.ctkCollapsibleButton()
    self.electrodesCollapsibleButton.text = "Electrodes"
    self.layout.addWidget(self.electrodesCollapsibleButton)
    # Layout within the Electrodes collapsible button
    self.electrodesFormLayout = qt.QFormLayout(self.electrodesCollapsibleButton)
    
    #Table of electrodes
    ElectrodeTable = MyTableWidget("ElectrodeTable")
    self.ElectrodeTable = ElectrodeTable
    ElectrodeTable.setRowCount(0)
    ElectrodeTable.setColumnCount(6)
    ElectrodeTable.setShowGrid(1)    
    ElectrodeTable.horizontalHeader().setStretchLastSection(0)
    ElectrodeTable.horizontalHeader().setResizeMode( 0, qt.QHeaderView.Stretch)
    ElectrodeTable.setColumnWidth(0, 80)
    ElectrodeTable.verticalHeader().setStretchLastSection(0)
    # old version: slicer 4.3.1
    #ElectrodeTable.setItem(0,0,qt.QTableWidgetItem("Electrode 1"))
    #for i in range(6):
    #     ElectrodeTable.setItem(0,i+1,qt.QTableWidgetItem("0"))
    #     ElectrodeTable.setColumnWidth(i+1, 40)
    #     ElectrodeTable.horizontalHeader().setResizeMode( i+1, qt.QHeaderView.Stretch)
    ElectrodeTable.setHorizontalHeaderLabels(('Name','X','Y','Z',u' θ',u'ψ'))
    self.electrodesFormLayout.addWidget(ElectrodeTable)
    
    #Add Electrode Button
    addElectrodeButton = qt.QPushButton("Add Electrode")
    addElectrodeButton.toolTip="Add Electrode to the workspace."
    self.electrodesFormLayout.addWidget(addElectrodeButton)
    addElectrodeButton.connect('clicked(bool)',self.onAddElectrode)
    self.addElectrodeButton = addElectrodeButton
    addElectrodeButton.setEnabled(False)
    self.Electrodes = 0
    
    # add save Electrodes button 
    saveElectrodeButton = qt.QPushButton("Save")
    saveElectrodeButton.toolTip="Save Electrodes Coordinates"
    saveElectrodeButton.setMaximumWidth(100)
    saveElectrodeButton.connect('clicked(bool)',self.onSaveElectrodes)
    self.electrodesFormLayout.addWidget(saveElectrodeButton)

    #
    # Patch Collapsible button
    #
    self.patchCollapsibleButton = ctk.ctkCollapsibleButton()
    self.patchCollapsibleButton.text = "Neural Population Patch"
    self.layout.addWidget(self.patchCollapsibleButton)
    
    # Layout within the Patch collapsible button
    NbPatches = 1
    self.Patches =1
    self.patchFormLayout = qt.QFormLayout(self.patchCollapsibleButton)
    #Set radio button group
    groupBox = qt.QGroupBox("Delineation Method:")
    radio1 = qt.QRadioButton("Ruberband Selection")
    radio2 = qt.QRadioButton("Predefined Area Selection")
    radio2.toggled.connect(self.radio2_clicked)
    radio3 = qt.QRadioButton("Free Selection")
    
    # Set a buttonGroup for the radiobbutons       
    self.ButtonGroup = qt.QButtonGroup()
    self.ButtonGroup.addButton(radio1,1)
    self.ButtonGroup.addButton(radio2,2)
    self.ButtonGroup.addButton(radio3,3)
    
    radio1.setChecked(True)
    self.areaText = qt.QLineEdit("500")
    self.areaText.setFixedWidth(120)
    self.areaText.setDisabled(1)
    areaLabel = qt.QLabel("mm<sup>2</sup>")
    vbox = qt.QGridLayout()
    vbox.addWidget(self.areaText,1,1,1,1)
    vbox.addWidget(areaLabel,1,2,1,1)
    vbox.addWidget(radio1,0,0)
    vbox.addWidget(radio2,1,0)
    vbox.addWidget(radio3,2,0)
    groupBox.setLayout(vbox)
    self.patchFormLayout.addWidget(groupBox)
    
    #Patch table
    PatchTable = MyTableWidget("PatchTable")
    self.PatchTable = PatchTable
    PatchTable.setRowCount(0)
    PatchTable.setColumnCount(4)
    PatchTable.setShowGrid(1)    
    PatchTable.horizontalHeader().setStretchLastSection(0)
    PatchTable.horizontalHeader().setResizeMode( 0, qt.QHeaderView.Stretch)
    PatchTable.setColumnWidth(0, 80)
    PatchTable.verticalHeader().setStretchLastSection(0)
    
    
    # add patchTable widget
    PatchTable.setHorizontalHeaderLabels(('Patch ID','Color','Transperancy','Nb of Populations'))
    self.patchFormLayout.addWidget(PatchTable)
    
    # add Patch Button
    addPatchButton = qt.QPushButton("Add Patch")
    addPatchButton.toolTip="delineate patch on the cerebral cortex."
    self.patchFormLayout.addWidget(addPatchButton)
    addPatchButton.connect('clicked(bool)',self.onAddPatch)
    self.addPatchButton = addPatchButton
    addPatchButton.setEnabled(False)
    
    # add show dipoles checkbox
    self.showDipoleCheckBox = qt.QCheckBox("Show Dipoles")
    self.showDipoleCheckBox.stateChanged.connect(self.showNormalTogglled)
    # add dipole size slider
    self.dipoleSlider = qt.QSlider(qt.Qt.Horizontal)
    self.dipoleSlider.setTickInterval(1)
    self.dipoleSlider.setSingleStep(1)
    self.dipoleSlider.setMinimum(1)
    self.dipoleSlider.setMaximum(10)
    self.dipoleSlider.setValue(5)
    self.dipoleSlider.setTickPosition(qt.QSlider.TicksBelow)
    self.dipoleSlider.valueChanged.connect(self.dipoleSizeChanged)
    #add space labels
    spaceLabel1 = qt.QLabel("     ")  
    spaceLabel2 = qt.QLabel("     ")      
    # add size label
    sizeLabel = qt.QLabel("Size:")  
    # add size label value
    self.sizeLabelValue = qt.QLabel("5")  
    # add dipole color button
    self.dipoleColorButton = qt.QPushButton("Color")
    self.dipoleColorButton.toolTip="Choose dipoles color"
    self.dipoleColorButton.setStyleSheet('QPushButton {background-color: rgb(0,0,255); color: white;}')
    self.dipoleColorButton.connect('clicked(bool)',self.onDipoleColorButton)
    # add gropbox for dipole management 
    dipoleBox = qt.QGroupBox("Dipole Controls:")    
    dbox = qt.QGridLayout()
    dbox.addWidget(self.showDipoleCheckBox,0,0)
    dbox.addWidget(spaceLabel1,0,1)
    dbox.addWidget(sizeLabel,0,2)
    dbox.addWidget(self.dipoleSlider,0,3)
    dbox.addWidget(self.sizeLabelValue,0,4)
    dbox.addWidget(spaceLabel2,0,5)
    dbox.addWidget(self.dipoleColorButton,0,6)
    dipoleBox.setLayout(dbox)    

    self.patchFormLayout.addWidget(dipoleBox)
    
    # add save patches button 
    savePatchButton = qt.QPushButton("Save")
    savePatchButton.toolTip="save patches information"
    savePatchButton.setMaximumWidth(100)
    savePatchButton.connect('clicked(bool)',self.onSavePatch)
    self.patchFormLayout.addWidget(savePatchButton)    
  
    #
    # Simulation Collapsible button
    #
    self.SimulationCollapsibleButton = ctk.ctkCollapsibleButton()
    self.SimulationCollapsibleButton.text = "Simulation"
    self.layout.addWidget(self.SimulationCollapsibleButton)
    # Layout within the Simulation collapsible button
    self.SimulationGridLayout = qt.QGridLayout(self.SimulationCollapsibleButton)
    
    self.combo = qt.QComboBox()
    self.combo.addItems(('None','Spatial Averaging','Temporal Averaging','Spati-Temporal Averaging'))
    chooseAveragingLabel = qt.QLabel("Averaging Method:  ")
    self.addFiletxt = qt.QLineEdit()
    addFileLabel = qt.QLabel("Neural Activity File:  ")    
    self.browseButton = qt.QPushButton("Browse")
    calculateLFButton = qt.QPushButton("Calculate LF")
    simulateButton = qt.QPushButton("Simulate")
    self.SimulationGridLayout.addWidget(chooseAveragingLabel, 0, 0)
    self.SimulationGridLayout.addWidget(self.combo, 0, 1, 1, 1)
    self.SimulationGridLayout.addWidget(addFileLabel, 1, 0)
    self.SimulationGridLayout.addWidget(self.addFiletxt, 1, 1, 1, 3)
    self.SimulationGridLayout.addWidget(self.browseButton, 1, 4)
    self.SimulationGridLayout.addWidget(calculateLFButton, 2, 3)
    self.SimulationGridLayout.addWidget(simulateButton, 2, 4)
    # Add functionality to buttons
    self.browseButton.connect('clicked(bool)',self.onBrowse)
    calculateLFButton.connect('clicked(bool)',self.onCalculateLF)
    simulateButton.connect('clicked(bool)',self.onSimulate)

    #Stretch 
    self.layout.addStretch(1)
    
  def onSimulate(self):     
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
      if not(hasattr(iren,'TemporalData')):
          msg = qt.QMessageBox()
          msg.setIcon(qt.QMessageBox().Critical)
          msg.setText('Load a neural population activty file in order to proceed')
          msg.setWindowTitle("Error")
          msg.exec_()
      elif hasattr(iren,'centers') and hasattr(iren,'PatchesCollection'):
          nbElec = len(iren.ActorSt.actAssembly)
          diag = qt.QDialog()
          # groupbox for electrodes
          groupBox = qt.QGroupBox("Choose Electrodes:")
          vbox = qt.QGridLayout()
          GridLayout = qt.QGridLayout(diag)
          hSim=[]
          for i in range (nbElec):
             hSim.append(qt.QRadioButton(self.ElectrodeTable.item(i,0).text(),diag))
             vbox.addWidget(hSim[i],i-(i/4)*4,i/4)
          groupBox.setLayout(vbox)           
          # groupbox for contacts
          groupBox2 = qt.QGroupBox("Choose Contacts:")
          Cbox = qt.QGridLayout()
          hSimC=[]
          for i in range (15):
             hSimC.append(qt.QCheckBox('Contact '+str(i+1),diag))
             Cbox.addWidget(hSimC[i],i-(i/4)*4,i/4)
          groupBox2.setLayout(Cbox)   
          
          displayButtonSim = qt.QPushButton("Display")
          cancelButtonSim = qt.QPushButton("Cancel")
          saveButtonSim = qt.QPushButton("Save")
          
          GridLayout.addWidget(groupBox,0,0,1,5)
          GridLayout.addWidget(groupBox2,1,0,1,5)
          
          GridLayout.addWidget(saveButtonSim,2,0,1,1)
          GridLayout.addWidget(displayButtonSim,2,2,1,1)
          GridLayout.addWidget(cancelButtonSim,2,4,1,1)
          diag.setWindowTitle("Simulate")
          diag.setWindowModality(qt.Qt.ApplicationModal)
          cancelButtonSim.clicked.connect(lambda: qt.QDialog.reject(diag))
          saveButtonSim.clicked.connect(lambda: self.onSaveSim(diag,hSim))   
          displayButtonSim.clicked.connect(lambda: self.onDisplaySim(diag,hSim,hSimC)) 
          diag.exec_()
          
  def onDisplaySim(self,diag,h,hSimC):
    modeEEG = self.combo.currentIndex
    iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
    matLF = self.calLeadField(h)      
    iren.LeadField = matLF 
    if modeEEG==2 or modeEEG ==3:
         # average temporal signals 
         averageSignal=np.mean(iren.TemporalData,axis=0)
         iren.TemporalData = np.tile(averageSignal,(np.array(iren.LeadField[0]).shape[1],1))
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    print iren.TemporalData.shape      
    for i in range(len(matLF)):
         V = np.dot(np.array(iren.LeadField[i]),np.array(iren.TemporalData)[0:np.array(iren.LeadField[i]).shape[1]]) 
    dn=[]
    a= []
    checkedE = []
    checkedC = []
    for i in range(len(h)):
       if h[i].isChecked():
          checkedE.append(i)      
    for i in range(len(hSimC)):
        if hSimC[i].checkState():
           checkedC.append(i)         
    # Create a Chart Node.
    cn = slicer.mrmlScene.AddNode(slicer.vtkMRMLChartNode())
    # Create an Array Node and add some data
    for j in range(len(checkedC)):
        dn.append(slicer.mrmlScene.AddNode(slicer.vtkMRMLDoubleArrayNode()))
        a.append(dn[j].GetArray())
        a[j].SetNumberOfTuples(V.shape[1])
        x = range(0, V.shape[1])    
        for i in range(len(x)):
            a[j].SetComponent(i, 0, x[i]/iren.fs)
            a[j].SetComponent(i, 1, V[checkedC[j]][i]*10**6+j*0.01)
            a[j].SetComponent(i, 2, 0)
         
        # Add the Array Nodes to the Chart. The first argument is a string used for the legend and to refer to the Array when setting properties.
        cn.AddArray(hSimC[checkedC[j]].text, dn[j].GetID())
    
    # Set a few properties on the Chart. The first argument is a string identifying which Array to assign the property. 
    # 'default' is used to assign a property to the Chart itself (as opposed to an Array Node).
    cn.SetProperty('default', 'title', 'LFPs of '+ str(len(checkedC))+ ' contacts of '+ h[checkedE[0]].text)
    cn.SetProperty('default', 'xAxisLabel', 'Time in s')
    cn.SetProperty('default', 'yAxisLabel', 'Amplitude in V')
    
    # Tell the Chart View which Chart to display
    self.cvn.SetChartNodeID(cn.GetID())  
    qt.QDialog.accept(diag) 

  def onSaveSim(self,diag,h):
      modeEEG = self.combo.currentIndex
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
      s1=PythonQt.QtGui.QFileDialog.getSaveFileName(self.browseButton,'Save File') 
      matLF = self.calLeadField(h)      
      iren.LeadField = matLF 
      if modeEEG==2 or modeEEG ==3:
         # average temporal signals 
         averageSignal=np.mean(iren.TemporalData,axis=0)
         iren.TemporalData = np.tile(averageSignal,(np.array(iren.LeadField[0]).shape[1],1))
      np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
      print type(np.array(iren.LeadField[0]))
      print type(np.array(iren.TemporalData)[0:np.array(iren.LeadField[0]).shape[1]])
      for i in range(len(matLF)):
          V = np.dot(np.array(iren.LeadField[i]),np.array(iren.TemporalData)[0:np.array(iren.LeadField[i]).shape[1]])          
          with open(s1 + h[i].text + '.txt', 'w') as f:
               f.write(np.array2string(V, separator=', '))
      qt.QDialog.accept(diag) 

  def onCalculateLF(self):     
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
      if hasattr(iren,'centers') and hasattr(iren,'PatchesCollection'):
          nbElec = len(iren.ActorSt.actAssembly)
          diag = qt.QDialog()
          groupBox = qt.QGroupBox("Choose Electrodes:")
          vbox = qt.QGridLayout()
          GridLayout = qt.QGridLayout(diag)
          h=[]
          for i in range(nbElec):
             h.append(qt.QCheckBox(self.ElectrodeTable.item(i,0).text(),diag))
             vbox.addWidget(h[i],i-(i/4)*4,i/4)
          groupBox.setLayout(vbox) 
          okButtonLF = qt.QPushButton("OK")
          cancelButtonLF = qt.QPushButton("Cancel")
          saveButtonLF = qt.QPushButton("Save")
          GridLayout.addWidget(groupBox,0,0,1,5)
          GridLayout.addWidget(okButtonLF,1,0,1,1)
          GridLayout.addWidget(cancelButtonLF,1,2,1,1)
          GridLayout.addWidget(saveButtonLF,1,4,1,1)
          diag.setWindowTitle("Calculate LF")
          diag.setWindowModality(qt.Qt.ApplicationModal)
          cancelButtonLF.clicked.connect(lambda: qt.QDialog.reject(diag))
          saveButtonLF.clicked.connect(lambda: self.onSaveLF(diag,h))   
          okButtonLF.clicked.connect(lambda: self.onOkLF(diag,h)) 
          diag.exec_()
      
  def onOkLF(self,diag,h):
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
      matLF = self.calLeadField(h)
      iren.LeadField = matLF   
      qt.QDialog.accept(diag)  
      
  def onSaveLF(self,diag,h):
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
      s1=PythonQt.QtGui.QFileDialog.getSaveFileName(self.browseButton,'Save File') 
      matLF = self.calLeadField(h)      
      iren.LeadField = matLF 
      np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
      for i in range(len(matLF)):
          with open(s1 + h[i].text + '.txt', 'w') as f:
               f.write(np.array2string(np.array(matLF[i]), separator=', '))
      qt.QDialog.accept(diag) 

      
  def calLeadField(self,h):
      modeEEG = self.combo.currentIndex
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
      checked = []
      if hasattr(h[0],'checkState():'):
          for i in range(len(h)):
              if h[i].checkState():
                 checked.append(i)   
      else:  
          for i in range(len(h)):
              if h[i].isChecked():
                 checked.append(i)            
      normals, dipCenters, cellArea, numOfCells=PrepareData(iren) 
      data=EEG(dipCenters,normals,cellArea,numOfCells,modeEEG)          
      #for x in range(len(self.centers)): 
      matLF = []
      for x in checked:
         data.setSenCenters(iren.centers[x])
         matLF.append(data.CalculateLF())
         # collect garbage explicitly          
         gc.collect()
      return  matLF 
  def onSavePatch(self):
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor() 
      if hasattr(iren,'PatchesCollection'):       
          normals, dipCenters, cellArea, numOfCells=PrepareData(iren)
          s1=PythonQt.QtGui.QFileDialog.getSaveFileName(self.browseButton,'Save File')
          np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
          with open(s1+'_normals.txt', 'w') as f:
               f.write(np.array2string(np.array(normals), separator=', '))
          with open(s1+'_centers.txt', 'w') as f:
               f.write(np.array2string(np.array(dipCenters), separator=', '))   
          with open(s1+'_areas.txt', 'w') as f:
               f.write(np.array2string(np.array(cellArea), separator=', '))     
  def onSaveElectrodes(self):
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()        
      if hasattr(iren,'centers'):
          s1=PythonQt.QtGui.QFileDialog.getSaveFileName(self.browseButton,'Save File')
          np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
          with open(s1+'.txt', 'w') as f:
               f.write(np.array2string(np.array(iren.centers), separator=', '))
  def onDipoleColorButton(self):
     iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()        
     if hasattr(iren,'DrawN'):       
          color = qt.QColorDialog.getColor()
          s1 = "QPushButton {background-color:"
          s2= "; color: white;}" 
          self.dipoleColorButton.setStyleSheet(s1 + color.name() + s2)
          color = color.toRgb()
          color = (color.red()/255.,color.green()/255.,color.blue()/255.)
          size = self.dipoleSlider.value
          self.plotNormals(color,size)
          
  def dipoleSizeChanged(self):
     iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()  
     self.sizeLabelValue.setText(str(self.dipoleSlider.value))
     color=self.dipoleColorButton.palette.color(1)
     color = color.toRgb()
     color = (color.red()/255.,color.green()/255.,color.blue()/255.)
     size = self.dipoleSlider.value      
     if hasattr(iren,'DrawN'): 
          self.plotNormals(color,size)
      
  def plotNormals(self,color,size):
    iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()  
    d=iren.GetInteractorStyle()
    if hasattr(iren,'DrawN'): 
        if iren.DrawN[0]:
              for i in range(len(iren.PatchesCollection)):
                  poly =vtk.vtkDataSetSurfaceFilter()
                  poly.SetInputData(iren.PatchesCollection[i].GetMapper().GetInput())
                  poly.Update()
                  DrawNormals(d,poly,i,color,size)         
        else:
              for i in range(len(iren.PatchesCollection)):
                  iren.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(iren.GlyphCollection[i])  
        iren.GetRenderWindow().Render()
               
  def showNormalTogglled(self):      
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()  
      if hasattr(iren,'DrawN'): 
          iren.DrawN[0]= self.showDipoleCheckBox.checkState()
          color=self.dipoleColorButton.palette.color(1)
          color = color.toRgb()
          color = (color.red()/255.,color.green()/255.,color.blue()/255.)
          size = self.dipoleSlider.value
          self.plotNormals(color,size)
          
  def onNodeSelected(self):
      if self.inputModelSelector.currentNode() is None:
          return
      else:  
          # Get vtkAlgorithmOutput
          brain1 = self.inputModelSelector.currentNode().GetDisplayNode().GetInputPolyDataConnection()
          # Get PolyData
          brain = self.inputModelSelector.currentNode().GetPolyData()          
          # Get Interactor
          iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
          # Get Renderer
          ren = iren.GetRenderWindow().GetRenderers().GetFirstRenderer()               
          # Initialize Highlight Interaction Style
         
          #----------------------------------Id Filter-------------------------
          idFilter = vtk.vtkIdFilter()
          idFilter.SetInputConnection(brain1)
          idFilter.SetIdsArrayName("OriginalIds")
          idFilter.Update()
    
          #-------------------------------Surface Filter-----------------------
          surfaceFilter = vtk.vtkDataSetSurfaceFilter()
          surfaceFilter.SetInputConnection(idFilter.GetOutputPort())
          surfaceFilter.Update()
          inputs = surfaceFilter.GetOutput()
          
          st=HighlightStyle()
          st.SetPolyData(brain1)
          iren.highlight=st
          # Innitialize Free Interaction Style
          free=FreeSelection(brain1)  
          iren.free=free  
          # Initialize Area Interaction Style
          AS=AreaStyle(brain1,inputs,500)
          AS.addElectrode=0
          iren.AreaStyle=AS
          # Place first electrode at default position
          elec1=Electrode(15)
          Actors=elec1.Create()
          mm=1
          sph=vtk.vtkSphereSource()       
          sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
          sph.SetRadius(0)
          #sphPoly=sph.GetOutput()                       
          #sphPoly.Update() 
          sphPoly=sph.GetOutputPort()
          sMT=vtk.vtkPolyDataMapper()
          sMT.SetInputConnection(sphPoly)
          sMT.GlobalImmediateModeRenderingOff()
          sA=vtk.vtkLODActor()
          sA.VisibilityOn()
          sA.SetMapper(sMT)
          # fill up assembly with electrode actors
          actAssembly=vtk.vtkAssembly()
          for i in range(elec1.NOS+1):
              actAssembly.AddPart(Actors[i])
          actAssembly.AddPart(sA)
          actAssembly.id=0
          actAssemblys=[]
          actAssemblys.append(actAssembly)
          # Set default orientation and position
          actAssembly.SetPosition(0,0,0)
       
          #Initialize Actor Interaction Style
          style = InteractorStyleTrackballActor(Actors,actAssemblys)
          iren.ActorSt=style 
          collecActors=ren.GetActors()
          collecActors.InitTraversal()
          for i in range(collecActors.GetNumberOfItems()):
              tempActor=collecActors.GetNextActor()
              if tempActor is None:
                  print("mesh loaded successfully")
              else:
                  if tempActor.GetMapper().GetInput().GetNumberOfCells () == brain.GetNumberOfCells ():
                     print(tempActor.GetMapper().GetInput().GetNumberOfCells ())
                     brainActor = tempActor
          brainActor.SetPickable(0)           
          iren.BA=brainActor
          iren.BM = tempActor.GetMapper()
          iren.centers=[]  
          iren.patchConfirmed = 2
          #ren.AddActor(actAssemblys[0])      
          iren.PatchesCollection=[] 
          iren.GlyphCollection=[]
          iren.DrawN=[0,0,0]
          iren.GetRenderWindow().Render()
          iren.SetInteractorStyle(style)  
          iren.AddObserver("KeyPressEvent",SelectStyle)
          self.addPatchButton.setEnabled(True)
          self.addElectrodeButton.setEnabled(True)
          iren.modelPatches =[]
          iren.displayModelPatches =[]
          iren.AddFirstElectrode = 0
  def onAddElectrode(self):
      
      iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
      print self.ElectrodeTable.rowCount
      if iren.AddFirstElectrode >0 :
    
          iren.BA.SetPickable(1)
          iren.SetInteractorStyle(iren.AreaStyle)
          elec1=Electrode(15,0.8,36)
          Actors=elec1.Create()
          mm=1
          sph=vtk.vtkSphereSource()       
          sph.SetCenter(0,-(1+elec1.NOS*3.5)*mm,0)    
          sph.SetRadius(0)
          sphPoly=sph.GetOutputPort()                       
          sMT=vtk.vtkPolyDataMapper()
          sMT.SetInputConnection(sphPoly)
          sMT.GlobalImmediateModeRenderingOff()
          sA=vtk.vtkLODActor()
          sA.VisibilityOn()
          sA.SetMapper(sMT)
          actAssembly=vtk.vtkAssembly()
          for i in range(elec1.NOS+1):
             actAssembly.AddPart(Actors[i])
          actAssembly.AddPart(sA)
             
          actAssembly.id=len(iren.ActorSt.actAssembly)             
          iren.ActorSt.actAssembly.append(actAssembly)  
          iren.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(iren.ActorSt.actAssembly[actAssembly.id])   
          iren.SetInteractorStyle(iren.ActorSt)
          iren.BA.SetPickable(0) 

          self.Electrodes = self.Electrodes+1
          self.ElectrodeTable.insertRow(self.ElectrodeTable.rowCount)
          self.ElectrodeTable.setItem(self.ElectrodeTable.rowCount-1,0,qt.QTableWidgetItem("Electrode " +str(self.Electrodes)))
          for i in range(self.ElectrodeTable.columnCount):
             self.ElectrodeTable.setItem(self.ElectrodeTable.rowCount-1,i+1,qt.QTableWidgetItem("0"))                          
          iren.GetRenderWindow().Render()  
          
          # try to add node            
          model = slicer.vtkMRMLModelNode()    
          model.SetAndObservePolyData(Actors[1].GetMapper().GetInput())
               
          modelDisplay = slicer.vtkMRMLModelDisplayNode()
          modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
          modelDisplay.SetVisibility(True) 
          slicer.mrmlScene.AddNode(modelDisplay)
          model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
          modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
          slicer.mrmlScene.AddNode(model)           
          
      else:
          iren.AddFirstElectrode = 1
          self.ElectrodeTable.insertRow(self.ElectrodeTable.rowCount)
          self.ElectrodeTable.setItem(0,0,qt.QTableWidgetItem("Electrode 1"))
          for i in range(self.ElectrodeTable.columnCount):
              self.ElectrodeTable.setItem(0,i+1,qt.QTableWidgetItem("0"))
              self.ElectrodeTable.setColumnWidth(i+1, 40)
              self.ElectrodeTable.horizontalHeader().setResizeMode( i+1, qt.QHeaderView.Stretch)
          self.Electrodes = self.Electrodes+1
          iren.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(iren.ActorSt.actAssembly[0])          
          iren.SetInteractorStyle(iren.ActorSt)
          iren.BA.SetPickable(0) 
          iren.GetRenderWindow().Render()
  def SelectStyle(self):
        iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
        #read style of patch selection
        chosenRadio = self.ButtonGroup.checkedId()
        if chosenRadio == 1:
            iren.highlight.highlightON =1                
            iren.BM.SetScalarModeToUsePointData()
            iren.BA.SetPickable(1)
            iren.SetInteractorStyle(iren.highlight)
            iren.SinglePatch=Patch(3)
            iren.GlyphCollection.append(iren.SinglePatch.glyphActor)
            iren.highlight.Create()   
            iren.DrawN[1]=0               
            iren.BA.GetProperty().SetOpacity(1)
            iren.BA.GetProperty().SetRepresentationToSurface()
        elif  chosenRadio == 2:            
            iren.BM.SetScalarModeToUseCellFieldData()  
            iren.BA.SetPickable(1) 
            iren.SinglePatch=Patch(1)
            reqArea = int(self.areaText.text)
            iren.AreaStyle.SetReqArea(reqArea)
            iren.SetInteractorStyle(iren.AreaStyle)            
            iren.GlyphCollection.append(iren.SinglePatch.glyphActor)
            iren.AreaStyle.Create()
            iren.DrawN[0]=0
        elif chosenRadio == 3:
            iren.BM.SetScalarModeToUseCellFieldData()
            iren.BA.SetPickable(1)
            iren.SetInteractorStyle(iren.free)
            iren.SinglePatch=Patch(2)    
            iren.GlyphCollection.append(iren.SinglePatch.glyphActor)
            iren.free.Create()
            iren.DrawN[2]=0
  def onAddPatch(self):
      global NbPatches
      iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
      if iren.patchConfirmed == 2:#add patch is pressed for first time
          iren.PatchSelected = 0
          iren.patchConfirmed = 0 
          self.addPatchButton.setText("Confirm")
          self.SelectStyle()
      else:
          if iren.patchConfirmed ==1:# add patch is pressed
             iren.PatchSelected = 0
             iren.patchConfirmed = 0
             self.addPatchButton.setText("Confirm")
             self.SelectStyle()

          elif iren.PatchSelected == 1: # confirm patch is pressed
            iren.patchConfirmed = 1
            self.addPatchButton.setText("Add Patch")
            iren.highlight.highlightON = 0            
            iren.SetInteractorStyle(iren.highlight)
            if self.PatchTable.rowCount >0 :    
               self.Patches = self.Patches +1
               self.PatchTable.insertRow(self.PatchTable.rowCount)
               self.PatchTable.setItem(self.PatchTable.rowCount-1,0,qt.QTableWidgetItem("Patch " + str(self.Patches)))
               normals, dipCenters, cellArea, numOfCells = PrepareData(iren)
               item1 = qt.QTableWidgetItem(str(numOfCells[len(iren.PatchesCollection)]))
               item1.setFlags(PythonQt.QtCore.Qt.ItemIsEnabled)
               self.PatchTable.setItem(self.PatchTable.rowCount-1,3,item1)
               # add color button in color column
               ColorButton = qt.QPushButton()
               ColorButton.setFixedWidth(20)
               if iren.SinglePatch.style == 3: 
                  ColorButton.setStyleSheet("background-color: rgb(255,0,0);")
               elif iren.SinglePatch.style == 1:
                  ColorButton.setStyleSheet("background-color: rgb(0,255,0);")
               elif iren.SinglePatch.style == 2:
                  ColorButton.setStyleSheet("background-color: rgb(255,153,51);")                   
               ColorButton.clicked.connect(lambda: self.onColorButton(ColorButton))
               ColorButton.setObjectName("Patch " +str(self.Patches))
               pWidget = qt.QWidget(self.parent)
               TempLayout = qt.QHBoxLayout(pWidget)
               TempLayout.addWidget(ColorButton)
               TempLayout.setAlignment(PythonQt.QtCore.Qt.AlignCenter)
               TempLayout.setContentsMargins(10,0,0,0)
               pWidget.setLayout(TempLayout)
               self.PatchTable.setCellWidget(self.PatchTable.rowCount-1,1,pWidget) 
              
               # add combo button  for transperancy
               combo = qt.QComboBox()
               combo.addItems(('1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0'))
               combo.currentIndexChanged.connect(lambda: self.onComboChanged(combo,ColorButton))              
               self.PatchTable.setCellWidget(self.PatchTable.rowCount-1,2,combo)
               self.createPatchNode()
                              
            else:
               self.PatchTable.insertRow(self.PatchTable.rowCount)
               self.PatchTable.setItem(0,0,qt.QTableWidgetItem("Patch 1"))
               
               normals, dipCenters, cellArea, numOfCells = PrepareData(iren)
               item1 = qt.QTableWidgetItem(str(numOfCells[1]))
               item1.setFlags(PythonQt.QtCore.Qt.ItemIsEnabled)
               self.PatchTable.setItem(self.PatchTable.rowCount-1,3,item1)
               # add color button in color column
               ColorButton = qt.QPushButton()
               ColorButton.setFixedWidth(20)
               if iren.SinglePatch.style == 3: 
                  ColorButton.setStyleSheet("background-color: rgb(255,0,0);")
               elif iren.SinglePatch.style == 1:
                  ColorButton.setStyleSheet("background-color: rgb(0,255,0);")
               elif iren.SinglePatch.style == 2:
                  ColorButton.setStyleSheet("background-color: rgb(255,153,51);")  
                  
               ColorButton.clicked.connect(lambda: self.onColorButton(ColorButton))
               ColorButton.setObjectName("Patch " +str(1))
               pWidget = qt.QWidget(self.parent);
               TempLayout = qt.QHBoxLayout(pWidget)
               TempLayout.addWidget(ColorButton)
               TempLayout.setAlignment(PythonQt.QtCore.Qt.AlignCenter)
               TempLayout.setContentsMargins(10,0,0,0)
               pWidget.setLayout(TempLayout)
               self.PatchTable.setCellWidget(0,1,pWidget) 
                
               # add combo button  for transperancy
               combo = qt.QComboBox()
               combo.addItems(('1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0'))
               combo.currentIndexChanged.connect(lambda: self.onComboChanged(combo,ColorButton))
               self.PatchTable.setCellWidget(0,2,combo)
               self.createPatchNode()
              # disable numberof population column  
    #          self.PatchTable.setItem(0,3,item1)
               
  def createPatchNode(self):       
       iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
       model = slicer.vtkMRMLModelNode()
       # convert to vtkpolydata
       geometryFilter =  vtk.vtkGeometryFilter()
       geometryFilter.SetInputConnection(iren.PatchesCollection[len(iren.PatchesCollection)-1].GetMapper().GetInputConnection(0,0));
       geometryFilter.Update()
       polydata = vtk.vtkPolyData()
       polydata = geometryFilter.GetOutput()
       model.SetAndObservePolyData(polydata)       
       modelDisplay = slicer.vtkMRMLModelDisplayNode()
       modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view               
       modelDisplay.SetVisibility(True)
       modelDisplay.AddViewNodeID('vtkMRMLSliceNodeRed')
       modelDisplay.AddViewNodeID('vtkMRMLSliceNodeGreen')
       modelDisplay.AddViewNodeID('vtkMRMLSliceNodeYellow')
       modelDisplay.SetSliceIntersectionThickness(2)
       if iren.SinglePatch.style == 3: 
          modelDisplay.SetColor(1,0,0) 
       elif iren.SinglePatch.style == 1:
          modelDisplay.SetColor(0,1,0) 
       elif iren.SinglePatch.style == 2:
          modelDisplay.SetColor(1,0.6,0.2) 
       modelDisplay.SetColor(1,0,0) 
       modelDisplay.SetName( "PatchDisplay "+str(len(iren.PatchesCollection)-1))
       model.SetName( "PatchModel "+str(len(iren.PatchesCollection)-1))
       slicer.mrmlScene.AddNode(modelDisplay)
       model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
       modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
       slicer.mrmlScene.AddNode(model) 
       iren.modelPatches.append(model)
       iren.displayModelPatches.append(modelDisplay)
  def onBrowse(self):
      iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
      s1=PythonQt.QtGui.QFileDialog.getOpenFileName(self.browseButton,'Choose neural population activity file', 'C:\\','Binary file(*.bin);; ASCII file (*.dat)')
      self.addFiletxt.setText(s1)
      s=ReadEEGfile(s1)
      iren.TemporalData=s.T
  def radio2_clicked(self, enabled):
      if enabled:
          self.areaText.setEnabled(1)
      else:
          self.areaText.setDisabled(1)
  def onColorButton(self,button):
          #patchID = button.parent().pos.y()/30
          index = self.PatchTable.indexAt(button.parent().pos)
          patchID = index.row()
          iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
          color = qt.QColorDialog.getColor()
          s = "background-color: "
          button.setStyleSheet(s + color.name())
          #change corresponding patch
          color = color.toRgb()
          iren.PatchesCollection[patchID].GetProperty().SetColor((color.red()/255.,color.green()/255.,color.blue()/255.))        
          iren.displayModelPatches[patchID].SetColor((color.red()/255.,color.green()/255.,color.blue()/255.))           
  def onComboChanged (self,combo,button):        
          patchID = button.parent().pos.y()/30 
          iren = slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()
          iren.PatchesCollection[patchID].GetProperty().SetOpacity(float(combo.currentText))
          iren.GetRenderWindow().Render()
class MyTableWidget(qt.QTableWidget):
    def __init__(self,name, parent=None):
        super(MyTableWidget, self).__init__(parent)
        self.itemDoubleClicked.connect(self.onItemDoubleClicked)
        self.itemChanged.connect(self.onItemChanged)
        self.InternalChange=1
        self.name = name
        
    def keyPressEvent(self, event):
         key = event.key()

         if key == PythonQt.QtCore.Qt.Key_Delete :
             self.onDeleteKeyPressed()

    def onDeleteKeyPressed(self):
        rows = self.selectionModel().selectedRows() 
        iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()  
        if  self.name=="ElectrodeTable":       
            for r in rows:
               self.removeRow(r.row())
               iren.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(iren.ActorSt.actAssembly[r.row()])
               iren.ActorSt.actAssembly.remove(iren.ActorSt.actAssembly[r.row()])
               iren.centers.remove(iren.centers[r.row()][:])
               if len(iren.ActorSt.actAssembly) ==0:
                  iren.ActorSt.cutterActor.SetVisibility(0)
                  iren.ActorSt.cutterActor2.SetVisibility(0)                 
               iren.GetRenderWindow().Render()          
               iren.BA.SetPickable(0)  
        elif self.name=="PatchTable":             
            for r in rows:
               self.removeRow(r.row())
               iren.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(iren.PatchesCollection[r.row()])
               if iren.DrawN[0]:
                   iren.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(iren.GlyphCollection[r.row()])
                   iren.GlyphCollection.remove(iren.GlyphCollection[r.row()])   
               iren.PatchesCollection.remove(iren.PatchesCollection[r.row()])
               iren.GetRenderWindow().Render()   
               slicer.mrmlScene.RemoveNode(iren.modelPatches[r.row()])
               slicer.mrmlScene.RemoveNode(iren.displayModelPatches[r.row()])
               iren.modelPatches.remove(iren.modelPatches[r.row()])
               iren.displayModelPatches.remove(iren.displayModelPatches[r.row()])
    def onItemChanged(self,item):
        if  self.name=="ElectrodeTable": 
            if self.InternalChange == 0:
                centers=[]
                iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()     
                rowCh = item.row()
                print(rowCh)
                iren.ActorSt.actAssembly[rowCh].SetScale([1,1,1])      
                g=iren.ActorSt.actAssembly[rowCh].GetMatrix()             
                h = vtk.vtkTransform()
                h.SetMatrix(g)     
                v=iren.ActorSt.Actors
                for i in range(0,len(v)):
                    v[i].SetUserTransform(h)
                    centers.append(v[i].GetCenter())
                k=h.Inverse()
                for i in range(0,len(v)):          
                    v[i].SetUserTransform(k)
                if rowCh<len(iren.centers) :
                    iren.centers[rowCh]=centers
                else:
                    iren.centers.append(centers)
                
                electrode= iren.ActorSt.actAssembly[rowCh]
                if self.item(rowCh,2) is not None and self.item(rowCh,3) is not None and self.item(rowCh,4) is not None and self.item(rowCh,4) is not None and self.item(rowCh,5) is not None: 
                    a1=np.array([float(self.item(rowCh,1).text()),float(self.item(rowCh,2).text()),float(self.item(rowCh,3).text())])  
                    electrode.SetPosition(a1[0],a1[1],a1[2])                    
                    a=np.array([float(self.item(rowCh,4).text()),float(self.item(rowCh,5).text())]) 
                    print electrode.GetOrientation()
                    electrode.SetOrientation(a[0],0,a[1])  
                iren.GetRenderWindow().Render() 
            
    def onItemDoubleClicked(self,event):    
        self.InternalChange = 0
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

           
#------------------------------Operating Core --------------------------------- 
#------------------------------------------------------------------------------                              
                                # Classes #
class Patch():
    def __init__(self,style):
        
        self.PatchMapper = vtk.vtkDataSetMapper()
        self.PatchActor = vtk.vtkActor()
        self.PatchActor.SetMapper(self.PatchMapper) 
        self.PatchActor.style = style
        # Actors for normals
        self.glyphMapper = vtk.vtkPolyDataMapper()
        self.glyphActor = vtk.vtkActor()
        self.glyphActor.SetMapper(self.glyphMapper)
        self.glyphActor.SetPickable(0)
        self.style = style
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
            self.exPoly.SetInputData(self.FrSelection.GetOutput())
            self.CellsActor =vtk.vtkActor()
        if style==3:
            self.SelectedMapper = vtk.vtkDataSetMapper()
            self.SelectedActor = vtk.vtkActor()
            self.SelectedActor.SetMapper(self.SelectedMapper)
            self.extracted=vtk.vtkExtractSelection()
    def setPatchColor(self,color):
        self.PatchActor.GetProperty().SetColor(color)

class AreaStyle(vtk.vtkInteractorStyleRubberBand3D):
    def __init__(self,brainAlgo,brain,ReqArea=1000):
        self.RequiredArea=ReqArea
        self.Brain=brain
        self.brainAlgo=brainAlgo
        self.picker = vtk.vtkCellPicker()
        #add an observer that starts on left button release event        
        self.AddObserver("LeftButtonReleaseEvent",self.PickArea)
        self.AddObserver('LeftButtonPressEvent',self.set_mmZero)
        self.AddObserver('MouseMoveEvent',self.set_mmOne)

    def SetReqArea(self,ReqArea):
        self.RequiredArea=ReqArea
        
    def Create(self):    
        self.triangleFilter = self.GetInteractor().SinglePatch.triangleFilter
        self.triangleFilter.SetInputConnection(self.brainAlgo)
        self.triangleFilter.Update()
        self.cellPointIds = self.GetInteractor().SinglePatch.cellPointIds        
        self.selectionNode = self.GetInteractor().SinglePatch.selectionNode
        self.selection = self.GetInteractor().SinglePatch.selection        
        self.exSelection = self.GetInteractor().SinglePatch.exSelection
        self.exSelection.SetInputConnection(0, self.brainAlgo)
        self.exPoly = self.GetInteractor().SinglePatch.exPoly
        self.exPoly.SetInputData(self.exSelection.GetOutput())
        self.area = self.GetInteractor().SinglePatch.area
        self.neighborCellIds = self.GetInteractor().SinglePatch.neighborCellIds   
        self.MouseMove=0
         
    def set_mmZero(self,obj,event):
      self.MouseMotion = 0
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
                    
                #  Create a dataset with the neighbor cells
                  ids=vtk.vtkIdTypeArray()
                  ids.SetNumberOfComponents(1)         
                  for it1 in range(len(neighbors)):
                    ids.InsertNextValue(neighbors[it1])
                
                  self.selectionNode.SetSelectionList(ids)             
                  self.exSelection.SetInputData(1, self.selection)
                  self.exSelection.Update()                    
                  self.exPoly.Update()              
                  a=self.area.GetSurfaceArea()+self.Brain.GetCell(cellId).ComputeArea()
                  x=x+1       
            
            self.GetInteractor().SinglePatch.PatchMapper.SetInputConnection(self.exSelection.GetOutputPort())
            self.GetInteractor().SinglePatch.PatchActor.SetMapper(self.GetInteractor().SinglePatch.PatchMapper)
            act=self.GetInteractor().SinglePatch.PatchActor
            if act not in self.GetInteractor().PatchesCollection:
                self.GetInteractor().PatchesCollection.append(act)
                self.GetInteractor().PatchSelected = 1
            PatcToRender=len(self.GetInteractor().PatchesCollection)-1
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.GetInteractor().PatchesCollection[PatcToRender])

            self.GetInteractor().Render()
            self.GetInteractor().SinglePatch.PatchActor.SetPickable(0)
            self.GetInteractor().PatchSelected = 1

     else:
         x,y=self.GetInteractor().GetEventPosition()
         self.picker.Pick(x,y,0,self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
         cellId=self.picker.GetCellId()
         if cellId==-1:
                pass
         else:
             n=[0,0,0]
             elecId=len(self.GetInteractor().ActorSt.actAssembly)
             dCell=self.Brain.GetCell(cellId)
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
      self.AddObserver("MiddleButtonPressEvent",self.MiddleButtonOn) 

    def MiddleButtonOn(self,obj,event):
       self.GetInteractor().SetInteractorStyle(self.GetInteractor().ActorSt)
       self.GetInteractor().BA.SetPickable(0)
    def Create(self):
      self.CellsMapper =self.GetInteractor().SinglePatch.CellsMapper
      self.selectionNode=self.GetInteractor().SinglePatch.selectionNode
      self.selection=self.GetInteractor().SinglePatch.selection
      self.selection.AddNode(self.GetInteractor().SinglePatch.selectionNode)
      self.FrSelection=self.GetInteractor().SinglePatch.FrSelection
      self.FrSelection.SetInputConnection(0, self.Brain)
      self.exPoly =self.GetInteractor().SinglePatch.exPoly
      self.exPoly.SetInputData(self.GetInteractor().SinglePatch.FrSelection.GetOutput())
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
              #  Create a dataset with the selected cells
              ids=vtk.vtkIdTypeArray()
              ids.SetNumberOfComponents(1)         
              for it1 in range(len(self.select)):
                ids.InsertNextValue(self.select[it1])             
             
              self.selectionNode.SetSelectionList(ids)             
              self.FrSelection.SetInputData(1, self.selection)
              self.FrSelection.Update()                    
              self.exPoly.Update()
        self.CellsMapper.SetInputConnection(self.FrSelection.GetOutputPort())
        self.CellsActor.SetMapper(self.CellsMapper)
        self.CellsActor.GetProperty().SetColor(1,0.6,0.2)
        
        act=self.GetInteractor().SinglePatch.CellsActor
        if act not in self.GetInteractor().PatchesCollection:
          self.GetInteractor().PatchesCollection.append(act)
          self.GetInteractor().PatchSelected = 1            
        PatchToRender=len(self.GetInteractor().PatchesCollection)-1
        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.GetInteractor().PatchesCollection[PatchToRender])
        self.GetInteractor().Render()     
        self.CellsActor.SetPickable(0)
    def set_timerOff(self,obj,event):
      self.GetInteractor().DestroyTimer(self.timer)

class HighlightStyle(vtk.vtkInteractorStyleRubberBand3D):
   def __init__(self,parent=None):

      #add an observer that starts on left button release event
      self.RemoveObservers('LeftButtonReleaseEvent')
      self.AddObserver("LeftButtonReleaseEvent",self.LeftButtonUp)
      self.AddObserver("MiddleButtonPressEvent",self.MiddleButtonOn) 
      self.highlightON = 0
   def Create(self):
      self.SelectedMapper = self.GetInteractor().SinglePatch.SelectedMapper
      self.SelectedActor =  self.GetInteractor().SinglePatch.SelectedActor
      self.extracted=self.GetInteractor().SinglePatch.extracted
   def MiddleButtonOn(self,obj,event):
       self.GetInteractor().SetInteractorStyle(self.GetInteractor().ActorSt)
       #self.GetInteractor().BA.SetPickable(0)
   def LeftButtonUp(self,obj,event):
      self.OnLeftButtonUp() # terminates the rubber band when left button release event is reached
      if  self.highlightON:
          
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
          self.extracted.SetInputConnection(0,self.PolyData)
          self.extracted.SetInputData(1,res)  
          self.extracted.Update()
          poly =vtk.vtkDataSetSurfaceFilter()
          poly.SetInputConnection(self.extracted.GetOutputPort())
          poly.Update()      
          area=vtk.vtkMassProperties()
          area.SetInputConnection(poly.GetOutputPort())
          print "%.2f mm²" %area.GetSurfaceArea() 
    
          #Draw the selected section along with the unselected
          self.SelectedMapper.SetInputConnection(self.extracted.GetOutputPort())
          self.SelectedMapper.ScalarVisibilityOff()
          self.SelectedActor.GetProperty().SetEdgeColor(0, 0, 0) # (R,G,B)
          self.SelectedActor.GetProperty().SetColor(1, 0, 0) # (R,G,B)
          #self.SelectedActor.GetProperty().EdgeVisibilityOn()
          self.SelectedActor.GetProperty().SetOpacity(0.5)     
          self.SelectedActor.GetProperty().SetPointSize(5)
          
                  
          act=self.GetInteractor().SinglePatch.SelectedActor
          if area.GetSurfaceArea()!=0:
              if act not in self.GetInteractor().PatchesCollection:
                  self.GetInteractor().PatchesCollection.append(act)
                  PatchToRender=len(self.GetInteractor().PatchesCollection)-1
                  self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.GetInteractor().PatchesCollection[PatchToRender])
                  self.GetInteractor().PatchSelected = 1
          self.GetInteractor().GetRenderWindow().Render()
          self.HighlightProp(None)     
          self.SelectedActor.SetPickable(1)    
      
   def SetPolyData(self,PolyData):
      self.PolyData=PolyData
      return

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
      
      self.AddObserver('RightButtonPressEvent',self.RightButtonPressed)
      
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
      self.cutterActor.SetPickable(0)

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
   def RightButtonPressed(self,obj,event):
       self.GetInteractor().SetInteractorStyle(self.GetInteractor().highlight)
   def MiddleButtonOn(self,obj,event):
        self.OnMiddleButtonDown()
        self.DetectMouseMove = 1
   def MiddleButtonOff(self,obj,event):
        self.OnMiddleButtonUp()
        self.DetectMouseMove = 0        
   def ConfirmPatch(self,obj,event):
       if self.GetInteractor().GetKeyCode()=="n":
          self.TemPnormals, self.TemPdipCenters, self.TemPcellArea, self.TemPnumOfCells=PrepareData(self.GetInteractor())
          self.TemPdata=EEG(self.TemPdipCenters,self.TemPnormals,self.TemPcellArea,self.GetInteractor().TemporalData[0:len(self.TemPdipCenters)],self.TemPnumOfCells,3)

   def Print1(self,obj,event):
       if self.DetectMouseMove:
          #start_time = time.time()
          self.TemPdata.setSenCenters(self.GetInteractor().centers[0][5])
          V = self.TemPdata.CalculateOne()

   def Print2(self,obj,event):
       #start_time = time.time()
       self.TemPdata.setSenCenters(self.GetInteractor().centers[0][5])
       V = self.TemPdata.CalculateOne()
            
   def MouseMoveOK(self,obj,event):
        self.OnLeftButtonDown()
        self.DetectMouseMove = 1
   def MouseMoveNot(self,obj,event):
        self.OnLeftButtonUp()
        self.DetectMouseMove = 0
   def MouseWheelForward(self,obj,event):
            centers=[]
            if len(self.actAssembly) >0 and len(self.GetInteractor().centers[self.elec])>0:
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
                
                ElectrodeTable.InternalChange = 1   
      
                for i in range(3):
                  ElectrodeTable.setItem(self.elec,i+1,qt.QTableWidgetItem(str(centers[1][i])))         
    
   def MouseWheelBackward(self,obj,event):
            centers=[]
            if len(self.actAssembly) and len(self.GetInteractor().centers[self.elec])>0:
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
                
                ElectrodeTable.InternalChange = 1   
      
                for i in range(3):
                  ElectrodeTable.setItem(self.elec,i+1,qt.QTableWidgetItem(str(centers[1][i])))
 
   def MouseMove(self,obj,event):
      global ElectrodeTable                    
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
              ElectrodeTable.InternalChange = 1   
              for i in range(3):
                 ElectrodeTable.setItem(self.elec,i+1,qt.QTableWidgetItem(str(centers[1][i])))
              # add some effects
              self.GetInteractor().BA.GetProperty().SetOpacity(1)
              self.GetInteractor().GetRenderWindow().Render()
              self.HighlightProp(None)  
              
              #calculate angles
              dirVect = np.array(centers[0]) - np.array(centers[1])
              try:
                theta = math.degrees(math.atan(dirVect[2]/math.sqrt(dirVect[1]*dirVect[1]+dirVect[0]*dirVect[0])))
              except ZeroDivisionError:
                print "divide by zero error"
              ElectrodeTable.setItem(self.elec,4,qt.QTableWidgetItem(str(theta)))
              try:
                psi = math.degrees(math.atan(-dirVect[0]/dirVect[1]))
              except ZeroDivisionError:
                print "divide by zero error"              
              ElectrodeTable.setItem(self.elec,5,qt.QTableWidgetItem(str(psi)))
              
class EEG():
   
   def __init__(self,dipCenters,normals,area,numOfCells,mode,temporalData=0):
        self.dipCenters=dipCenters
        self.normals=normals
        self.area=area
        self.temporalData=temporalData
        self.numOfCells=numOfCells
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
        if self.mode==2 or self.mode==3 :
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
    
           start_time = time.time()    
           for j in range(np.matrix(self.senCenters).shape[0]):
               print "this is J" , j
               vn=[0]*np.matrix(self.dipCenters).shape[0]
               matLF = [0]*np.matrix(self.dipCenters).shape[0]
               for i in range(np.matrix(self.dipCenters).shape[0]):
                   print i
                   vn[i] = dProduct[j][i]*self.temporalData[i]*self.area[i]*0.525/((4*np.pi*sigma)*r2[j][i]**2) 
                   matLF[i] = dProduct[j][i]*self.area[i]*0.525/((4*np.pi*sigma)*r2[j][i]**2) 
               v.append([sum(x) for x in zip(*vn)])  
           print "Time elapsed: ", time.time() - start_time, "s" 
           return v, matLF
           
   def CalculateLF(self):
       # Spacial averaging (orientation only)
       sigma=30*10**-5
       if self.mode==1 or self.mode ==3:
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
       start_time = time.time()
       matLF = [[None] * np.matrix(self.dipCenters).shape[0] for i in range(np.matrix(self.senCenters).shape[0])]
       print np.matrix(matLF).shape
       for j in range(np.matrix(self.senCenters).shape[0]):
           for i in range(np.matrix(self.dipCenters).shape[0]):
               matLF[j][i] = dProduct[j][i]*self.area[i]*0.525*10**-9/((4*np.pi*sigma)*r2[j][i]**2) 
       print "Time elapsed: ", time.time() - start_time, "s" 
       return matLF  
       
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

class Electrode():
    def __init__(self,NumOfSensors=15,Diameter=0.8,Resolution=36):
        self.NOS=NumOfSensors
        self.Diameter=Diameter
        self.Resolution=Resolution
    def Create(self):  
        SensorLength=2
        InsulatorLength = 1.5
        mm=1        
        cylActors=[]
        
        # design tube holding electrodes
        tube=vtk.vtkCylinderSource()
        tube.SetHeight(((self.NOS*2+(self.NOS-1)*InsulatorLength)-0.1)*mm)
        tube.SetCenter(0,(self.NOS*2+(self.NOS-1)*InsulatorLength)*mm/2,0)
        tube.SetRadius((self.Diameter-0.05)*mm/2)
        tube.SetResolution(self.Resolution)
        #tubePoly=tube.GetOutput()                       
        #tubePoly.Update() 
        tubePoly=tube.GetOutputPort()
        TubeMT=vtk.vtkPolyDataMapper()
        TubeMT.SetInputConnection(tubePoly)
        TubeMT.GlobalImmediateModeRenderingOn()
        TubeA=vtk.vtkLODActor()
        TubeA.VisibilityOn()
        TubeA.SetMapper(TubeMT)
        TubeA.GetProperty().SetColor(1,0,0)
        cylActors.append(TubeA)
        
        # create the Electrodes
        for i in xrange(self.NOS):
            # create cylinder
            cyl=vtk.vtkCylinderSource()
            cyl.SetHeight(SensorLength*mm)            
            cyl.SetCenter(0,(1+i*(2+InsulatorLength))*mm,0)
            cyl.SetRadius(self.Diameter/2*mm)
            cyl.SetResolution(self.Resolution)           
            #cylPoly=cyl.GetOutput()                       
            #cylPoly.Update()
            cylPoly=cyl.GetOutputPort()
            # create mappers and actors for the sensors
            cMT=vtk.vtkPolyDataMapper()
            cMT.SetInputConnection(cylPoly)
            cMT.GlobalImmediateModeRenderingOn()
            cA=vtk.vtkLODActor()
            cA.VisibilityOn()
            cA.SetMapper(cMT)
            if i==0:
                cA.GetProperty().SetColor(0,1,0)
            else:
                cA.GetProperty().SetColor(0,0,1)
            cylActors.append(cA)           
        return cylActors

        
                           # Global Functions #
        
def DrawNormals(iStyle,poly,i,color,size):
      normalsCalc = vtk.vtkPolyDataNormals()
      normalsCalc.SetInputConnection(poly.GetOutputPort())
      # Disable normal calculation at cell vertices
      normalsCalc.ComputePointNormalsOff()
      # Enable normal calculation at cell centers
      normalsCalc.ComputeCellNormalsOn()
      # Disable splitting of sharp edges
      normalsCalc.SplittingOff()
      # Disable global flipping of normal orientation
      #normalsCalc.FlipNormalsOn()
      # Disable automatic determination of correct normal orientation
      normalsCalc.AutoOrientNormalsOff()
      # Perform calculation
      normalsCalc.Update()
#      array=normalsCalc.GetOutput().GetCellData().GetNormals()
#      v=[0,0,0]
#      array.GetTupleValue(0,v)
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
      glyph.SetScaleFactor(size/4.)
      
      actor=iStyle.GetInteractor().GlyphCollection[i]
      # Create a mapper for all the arrow-glyphs
      actor.GetMapper().SetInputConnection(glyph.GetOutputPort())
            
      # Create an actor for the arrow-glyphs      
      actor.GetProperty().SetColor(color)
      # Add actor
      iStyle.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(actor)

def PrepareData(iren):
    poly =vtk.vtkDataSetSurfaceFilter()
    # Compute Normals of selected patch
    normalsCalc = vtk.vtkPolyDataNormals()
    cellCenters=vtk.vtkCellCenters()
    normals=[]
    dipCenters=[]
    cellArea=[]
    numOfCells=[0]*(len(iren.PatchesCollection)+1)
    for j in range (len(iren.PatchesCollection)):
       poly.SetInputData(iren.PatchesCollection[j].GetMapper().GetInput())
       poly.Update()
       normalsCalc.SetInputConnection(poly.GetOutputPort())
       normalsCalc.ComputePointNormalsOff()
       normalsCalc.ComputeCellNormalsOn()
       normalsCalc.SplittingOff()
       normalsCalc.FlipNormalsOn()
       normalsCalc.AutoOrientNormalsOff()
       normalsCalc.Update()
       cellCenters.VertexCellsOn()
       cellCenters.SetInputConnection(normalsCalc.GetOutputPort())
       cellCenters.Update()
       array=normalsCalc.GetOutput().GetCellData().GetNormals()
       for i in range(array.GetNumberOfTuples()):
           numOfCells[j+1]=numOfCells[j+1]+1
           normals.append(array.GetTuple(i))
           dipCenters.append( cellCenters.GetOutput().GetPoint(i))
           cellArea.append(poly.GetOutput().GetCell(i).ComputeArea())
    return normals, dipCenters, cellArea, numOfCells   
def SaveSig(filename,obj):
        fileObj=open(filename,'wb')
        cPickle.dump(obj,fileObj)
        fileObj.close()       

def LoadSig(filename):
        fileObj=open(filename,'rb')
        sig=cPickle.load(fileObj)
        fileObj.close()
        return sig
def SelectStyle(self,obj):
  global  ElectrodeTable
  if self.GetKeyCode()=="1":   
    #self.PatchConfirmed=not(self.PatchConfirmed)
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
    #self.BA.GetProperty().SetOpacity(1)
    
    self.SetInteractorStyle(self.AreaStyle)
    self.SinglePatch=Patch(1)
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

      normals, dipCenters, cellArea, numOfCells=PrepareData(self)
      
      print numOfCells
      if len(dipCenters)>len(self.TemporalData):
          print "insufficient temporal data ..."
      else:

          data=EEG(dipCenters,normals,cellArea,self.TemporalData[0:len(dipCenters)],numOfCells,modeEEG)
          
          print len(self.TemporalData[0:len(dipCenters)])
          #for x in range(len(self.centers)): 
          for x in range(1):
              data.setSenCenters(self.centers[x])
              v=data.Vj()
              a=np.linspace(0,1,2*8192)
              for i in range(15):
                  SaveSig('Electrode '+str(x+1)+'mode'+str(data.mode)+str(i),v)
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
          ElectrodeTable.removeRow(elec)

      self.BA.SetPickable(0)
      
      if hasattr(self,'PatchesCollection'):  
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
  
def ReadEEGfile(filename):
    name, ext=os.path.splitext(filename)  
    iren=slicer.app.layoutManager().threeDWidget(0).threeDView().interactorStyle().GetInteractor()     
    # read descriptive file  into fs, nChannels, nSamples
    data=open(name+'.des','r').readlines()[4]
    fs=float(data.split()[1])
    iren.fs =fs
    data=open(name+'.des','r').readlines()[7]
    nSamples=int(data.split()[1])
    data=open(name+'.des','r').readlines()[10]
    nChannels=int(data.split()[1])
    T0=0
    print nChannels
    print nSamples
    print fs
    if ext=='.bin':
        fin=open(filename,'rb')         
        x=array.array('f')
        x.read(fin,nSamples*nChannels)
        sig = np.array(x).reshape(nChannels,nSamples)
#        start=int(T0*fs*nChannels)
#        sig=[0 for i in range(int(nSamples-T0*fs))]
#        for i in range(nChannels):
#            sig[i]=np.array(x[start+i::nChannels])    
        return np.array(sig).T
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
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------        
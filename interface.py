from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton,QLCDNumber
from pyqtgraph import PlotWidget
import functions
#from mplwidget import MplWidget
import pyqtgraph
#from classes import channelLine




def initConnectors(self):
    
    self.tabWidget=self.findChild(QtWidgets.QTabWidget , "tabWidget") 
    self.tabWidget.setStyleSheet("background-color: #666666")
    
    self.tab =self.findChild(QtWidgets.QWidget, "tab") 
    
    self.tab_2 = self.findChild(QtWidgets.QWidget,"tab_2")
    
    self.gridLayout=self.findChild(QtWidgets.QGridLayout , "gridLayout") 
    
    
    self.signalsBoxComposer=self.findChild(QtWidgets.QComboBox , "signalsBoxComposer") 
    
    self.graph2Composer=self.findChild(PlotWidget , "graph2Composer") 
    self.graph2Composer.setBackground('black')

    self.graph1Composer=self.findChild(PlotWidget , "graph1composer") 
    self.graph1Composer.setBackground('black')
    
    self.frequencySliderComposer=self.findChild(QtWidgets.QSlider , "frequencySliderComposer") 
    self.frequencySliderComposer.valueChanged.connect(lambda: functions.compose_signal(self))
    self.frequencySliderComposer.setRange(1, 100)    
    self.frequencySliderComposer.setValue(1)
    
    
    self.frequencyBoxComposer=self.findChild(QtWidgets.QLCDNumber , "frequencyBoxComposer") 
    self.frequencyBoxComposer.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
    
    self.magnitudeSliderComposer=self.findChild(QtWidgets.QSlider , "magnitudeSliderComposer") 
    self.magnitudeSliderComposer.valueChanged.connect(lambda: functions.compose_signal(self))
    self.magnitudeSliderComposer.setRange(1, 200)  # Magnitude range
    self.magnitudeSliderComposer.setValue(1)
    
    self.magnitudeBoxComposer=self.findChild(QtWidgets.QLCDNumber , "magnitudeBoxComposer") 
    
    self.phaseSliderComposer=self.findChild(QtWidgets.QSlider , "phaseSliderComposer") 
    self.phaseSliderComposer.valueChanged.connect(lambda: functions.compose_signal(self))
    self.phaseSliderComposer.setRange(0, 360)  # Phase range
    
    self.phaseBoxComposer=self.findChild(QtWidgets.QLCDNumber , "phaseBoxComposer") 
        
    self.deleteButtonComposer=self.findChild(QtWidgets.QPushButton , "deleteButtonComposer") 
    self.deleteButtonComposer.clicked.connect(lambda: functions.delete_signal(self))
    self.addButtonComposer = self.findChild(QtWidgets.QPushButton, "addButtonComposer")
    self.addButtonComposer.clicked.connect(lambda: functions.add_signal(self))
    
    self.saveButtonComposer = self.findChild(QtWidgets.QPushButton, "saveButtonComposer")
    #saveButtonComposer.clicked.connect()
    
    self.graph1Viewer = self.findChild(QtWidgets.QWidget, "graph1Viewer")
    self.graph1Viewer.setBackground('black')
    self.saveButtonComposer.clicked.connect(lambda: functions.updateSaved(self))
    self.saveButtonComposer.clicked.connect(lambda: functions.view_graph(self))
    self.saveButtonComposer.clicked.connect(lambda: functions.interpolate_graph(self))
    #self.saveButtonComposer.clicked.connect(lambda: functions.interpolate_graph(self))
    
    self.graph2Viewer = self.findChild(QtWidgets.QWidget, "graph2Viewer")
    self.graph2Viewer.setBackground('black')
    
    self.graph3Viewer = self.findChild(QtWidgets.QWidget, "graph3Viewer")
    self.graph3Viewer.setBackground('black')
    
    self.SNRSlider = self.findChild(QtWidgets.QSlider, "SNRSlider")
    
    self.FMaxLabel=self.findChild(QtWidgets.QLabel,"FMaxLabel")
    self.loadButtonViewer=self.findChild(QtWidgets.QPushButton, "loadButtonViewer")
    self.loadButtonViewer.clicked.connect(lambda: functions.Load_graph(self))
    self.loadButtonViewer.clicked.connect(lambda: functions.interpolate_graph(self))
    self.Save=self.findChild(QtWidgets.QAction ,"Save")
    self.Save.triggered.connect(lambda: functions.save(self))
    self.samplingSliderViewer = self.findChild(QtWidgets.QSlider, "samplingSliderViewer")
    self.samplingSliderViewer.valueChanged.connect(lambda: functions.view_graph(self))
    self.samplingSliderViewer.valueChanged.connect(lambda: functions.interpolate_graph(self))
    self.samplingSliderViewer.setValue(2)
    self.samplingSliderViewer.setRange(2,500)
    
    self.normalizedSliderViewer = self.findChild(QtWidgets.QSlider, "normalizedSliderViewer")
    self.normalizedSliderViewer.setValue(0)
    self.normalizedSliderViewer.setRange(0,5)
    self.normalizedSliderViewer.valueChanged.connect(lambda:functions.normalize(self))

    
    
    self.normalizedBoxViewer=self.findChild(QtWidgets.QLCDNumber,"normalizedBoxViewer")
    
    
    
    self.SNRSlider = self.findChild(QtWidgets.QSlider, "SNRSlider")
    self.SNRSlider.setValue(50)
    self.SNRSlider.setRange(1,50)
    self.SNRSlider.valueChanged.connect(lambda: functions.update_saved_signal(self))
    self.SNRSlider.valueChanged.connect(lambda: functions.view_graph(self))
    self.SNRSlider.valueChanged.connect(lambda: functions.interpolate_graph(self))
    
    self.frequencyBoxViewer=self.findChild(QtWidgets.QLCDNumber,"frequencyBoxViewer")
    
    self.FMaxBox=self.findChild(QtWidgets.QLCDNumber,"FMaxBox")
    self.FMaxBox.setSegmentStyle(QtWidgets.QLCDNumber.Flat)

# Connect the "Save" button to the save_signal function

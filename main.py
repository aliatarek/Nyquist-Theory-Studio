# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
import interface
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget

import sys

class UI(QMainWindow): 
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi("uifinal.ui",self)
        interface.initConnectors(self)
        self.show()


app=QApplication(sys.argv)
UIWindow= UI()
app.exec_()



from PyQt6 import uic, QtWidgets
from PyQt6.QtWidgets import QMainWindow, QListWidgetItem, QButtonGroup, QFileDialog
from PyQt6.QtCore import QAbstractTableModel, Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel

# from PySide6.QtCore import Qt

import os, time, random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from python import load, optimisation, sensitivity_funcs
from python.optimisation_funcs import optimisation_mainscript
from python.windows import optimisation_results, optimisation_setup, developer, prediction

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

# class Worker(QObject):
#     def __init__(self, num_iterations, window):
#         super().__init__()
#     # def __init__ (self, num_iterations=100, *args, **kwargs):
#         self.num_iterations = num_iterations
#         self.window = window
#         self.cancelled = False
#         self.window.cancel.connect(self.stop)

#     finished = pyqtSignal(str)
#     progress = pyqtSignal(int)

#     def run (self):
#         """Long-running task."""
#         for i in range(self.num_iterations):
#             time.sleep(0.1)
#             self.progress.emit(i+1)
#             if self.cancelled:
#                 self.finished.emit("Stopped early!")
#                 break
#             if i+1 == self.num_iterations:
#                 self.finished.emit("Finished!")

#     def stop (self):
#         self.cancelled = True


class Window(QMainWindow):
    def __init__ (self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        uic.loadUi('ui/sensitivity.ui', self)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.grid.addWidget(self.canvas)
        # time.sleep(1)
        # self.canvas.axes.cla()
        # self.canvas.axes.plot([0,1,2,3,4], [5,1,20,3,4])
        # self.canvas.draw_idle()

        self.component_dropdown.addItems(["U-bending"])
        self.component_dropdown.currentTextChanged.connect(self.update_component)
        self.update_component()

        self.load_button.pressed.connect(self.load_sensitivity)

        self.action_prediction.triggered.connect(self.open_prediciton_window)
        self.action_newoptimisation.triggered.connect(self.new_optimisaiton_window)
        self.action_optimisation.triggered.connect(self.open_optimisaiton_window)
        self.action_developer.triggered.connect(self.open_developer_window)

    def update_component (self):
        component = self.component_dropdown.currentText().lower()
        if component == "u-bending":
            self.var1_dropdown.clear()
            self.var2_dropdown.clear()
            self.var1_dropdown.addItems(["BHF", "Friction", "Clearance", "Thickness"])
            self.var2_dropdown.addItems(["Max Thinning"])

    def load_sensitivity (self):
        component = self.component_dropdown.currentText().lower()
        var1 = self.var1_dropdown.currentText().lower()
        var2 = self.var2_dropdown.currentText().lower()
        sensitivity_funcs.load(component, var1, var2, self)
            
    def open_prediciton_window (self):
        self.prediciton_window = prediction.Window()
        self.close()
        self.prediciton_window.show()

    def new_optimisaiton_window (self):
        self.optimisation_window = optimisation_setup.Window()
        self.close()
        self.optimisation_window.show()   

    def open_optimisaiton_window (self):
        self.optimisation_window = optimisation_results.Window()
        self.close()
        self.optimisation_window.show()

    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.close()
        self.developer_window.show()
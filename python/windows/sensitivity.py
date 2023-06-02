
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
        self.canvas = MplCanvas(self, width=6, height=4, dpi=100)
        # self.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.grid.addWidget(self.canvas)
        # time.sleep(1)
        # self.canvas.axes.cla()
        # self.canvas.axes.plot([0,1,2,3,4], [5,1,20,3,4])
        # self.canvas.draw_idle()

        self.component_dropdown.addItems(["U-bending"])
        self.component_dropdown.currentTextChanged.connect(self.update_component)
        self.update_component()

        self.var1_dropdown.currentTextChanged.connect(self.update_sliders)
        self.update_sliders()

        self.load_button.pressed.connect(self.load_sensitivity)

        self.slider1.sliderReleased.connect(self.load_sensitivity)
        self.slider2.sliderReleased.connect(self.load_sensitivity)
        self.slider3.sliderReleased.connect(self.load_sensitivity)
        self.slider4.sliderReleased.connect(self.load_sensitivity)

        self.slider1.valueChanged.connect(self.update_sliders)
        self.slider2.valueChanged.connect(self.update_sliders)
        self.slider3.valueChanged.connect(self.update_sliders)
        self.slider4.valueChanged.connect(self.update_sliders)

        self.action_prediction.triggered.connect(self.open_prediciton_window)
        self.action_newoptimisation.triggered.connect(self.new_optimisaiton_window)
        self.action_optimisation.triggered.connect(self.open_optimisaiton_window)
        self.action_developer.triggered.connect(self.open_developer_window)

    def update_component (self):
        component = self.component_dropdown.currentText()
        if component.lower() == "u-bending":
            self.var1_dropdown.clear()
            self.var2_dropdown.clear()
            self.var1_dropdown.addItems(["Blank Holding Force (kN)", "Friction Coefficient", "Clearance (%)", "Thickness (mm)"])
            self.var2_dropdown.addItems(["Max Thinning (%)"])

            maxBHF = 59
            maxFriction = 0.20
            maxClearance = 1.49
            maxThickness = 2.99

            minBHF = 5.2
            minFriction = 0.1
            minClearance = 1.1
            minThickness = 0.51

            midBHF = minBHF + (maxBHF-minBHF)/2
            midFriction = minFriction + (maxFriction-minFriction)/2
            midClearance = minClearance + (maxClearance-minClearance)/2
            midThickness = minThickness + (maxThickness-minThickness)/2

            self.slider1.setMinimum(minBHF * 100)
            self.slider1.setMaximum(maxBHF * 100)
            self.slider1.setValue(midBHF * 100)
            self.slider2.setMinimum(minFriction * 100)
            self.slider2.setMaximum(maxFriction * 100)
            self.slider2.setValue(midFriction * 100)
            self.slider3.setMinimum(minClearance * 100)
            self.slider3.setMaximum(maxClearance * 100)
            self.slider3.setValue(midClearance * 100)
            self.slider4.setMinimum(minThickness * 100)
            self.slider4.setMaximum(maxThickness * 100)
            self.slider4.setValue(midThickness * 100)

            self.label1.setText(f"Blank Holding Force: {self.slider1.value() / 100} kN")
            self.label2.setText(f"Friction Coefficient: {self.slider2.value() / 100}")
            self.label3.setText(f"Clearance: {self.slider3.value() / 100} %")
            self.label4.setText(f"Thickness: {self.slider4.value() / 100} mm")

    def load_sensitivity (self):
        component = self.component_dropdown.currentText()
        var1 = self.var1_dropdown.currentText()
        var2 = self.var2_dropdown.currentText()
        bhf = self.slider1.value() / 100
        friction = self.slider2.value() / 100
        clearance = self.slider3.value() / 100
        thickness = self.slider4.value() / 100
        sensitivity_funcs.load(component, var1, var2, self, bhf, friction, clearance, thickness)

    def update_sliders (self):
        var1 = self.var1_dropdown.currentText()
        bhf = self.slider1.value() / 100
        friction = self.slider2.value() / 100
        clearance = self.slider3.value() / 100
        thickness = self.slider4.value() / 100
        self.label1.setText(f"Blank Holding Force: {bhf} kN")
        self.label2.setText(f"Friction Coefficient: {friction}")
        self.label3.setText(f"Clearance: {clearance} %")
        self.label4.setText(f"Thickness: {thickness} mm")
        if "blank holding force" in var1.lower():
            self.label1.setEnabled(False)
            self.slider1.setEnabled(False)
            self.label2.setEnabled(True)
            self.slider2.setEnabled(True)
            self.label3.setEnabled(True)
            self.slider3.setEnabled(True)
            self.label4.setEnabled(True)
            self.slider4.setEnabled(True)
        if "friction" in var1.lower():
            self.label1.setEnabled(True)
            self.slider1.setEnabled(True)
            self.label2.setEnabled(False)
            self.slider2.setEnabled(False)
            self.label3.setEnabled(True)
            self.slider3.setEnabled(True)
            self.label4.setEnabled(True)
            self.slider4.setEnabled(True)
        if "clearance" in var1.lower():
            self.label1.setEnabled(True)
            self.slider1.setEnabled(True)
            self.label2.setEnabled(True)
            self.slider2.setEnabled(True)
            self.label3.setEnabled(False)
            self.slider3.setEnabled(False)
            self.label4.setEnabled(True)
            self.slider4.setEnabled(True)
        if "thickness" in var1.lower():
            self.label1.setEnabled(True)
            self.slider1.setEnabled(True)
            self.label2.setEnabled(True)
            self.slider2.setEnabled(True)
            self.label3.setEnabled(True)
            self.slider3.setEnabled(True)
            self.label4.setEnabled(False)
            self.slider4.setEnabled(False)
            
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
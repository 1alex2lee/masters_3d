
from PyQt6 import uic, QtWidgets
from PyQt6.QtWidgets import QMainWindow, QListWidgetItem, QButtonGroup, QFileDialog
from PyQt6.QtCore import QAbstractTableModel, Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel

# from PySide6.QtCore import Qt

import os, time, random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from python import load, optimisation
from python.optimisation_funcs import optimisation_mainscript
from python.windows import optimisation_results

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
    def __init__ (self, num_iterations=100, file="temp/DieForImage_1002.STL", active_component="Bulkhead", *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        uic.loadUi('ui/optimisation_setup2.ui', self)

        self.num_iterations = num_iterations
        self.file = file
        self.component = active_component

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.grid.addWidget(self.canvas)
        # time.sleep(1)
        # self.canvas.axes.cla()
        # self.canvas.axes.plot([0,1,2,3,4], [5,1,20,3,4])
        # self.canvas.draw_idle()

        self.runsno_label.setText("Currently on iteration 1 out of " + str(self.num_iterations))

        self.next_button.pressed.connect(self.show_result)

        self.begin_optimisation()

    cancel = pyqtSignal()

    def update_material (self):
        self.material_dropdown.clear()
        self.material_dropdown.addItems(load.materials(self.process_dropdown.currentText()))

    def update_model (self):
        self.model_dropdown.clear()
        self.model_dropdown.addItems(load.models(self.process_dropdown.currentText(), self.material_dropdown.currentText()))

    def update_goals (self, item):
        text = item.text()
        if item.checkState() == Qt.CheckState.Checked:
            if self.goal_dropdown.findText(text) == -1:
                self.goal_dropdown.addItem(text)

        elif item.checkState() == Qt.CheckState.Unchecked:
            self.goal_dropdown.removeItem(self.goal_dropdown.findText(text))

    def load_mesh (self):
        file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="STL file (*.stl);; STEP file (*.step)")
        optimisation.load_mesh(file[0], self)

    def begin_optimisation (self):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = optimisation_mainscript.worker(self.num_iterations, self.file, self.component, self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.report_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.report_progress)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        # self.longRunningBtn.setEnabled(False)
        # self.thread.finished.connect(
        #     lambda: self.longRunningBtn.setEnabled(True)
        # )
        
        self.worker.finished.connect(self.report_finished)
        self.cancel_button.pressed.connect(self.cancel.emit)

    def report_progress (self, progress):
        self.runsno_label.setText("Currently on iteration " + str(progress) + " out of " + str(self.num_iterations))
        self.progressBar.setValue(100 * (progress-1)/self.num_iterations)

    def report_finished (self, component):
        self.component = component
        self.runsno_label.setText(f"{component} optimisation completed with {self.num_iterations} iterations")
        self.progressBar.setValue(100)
        self.next_button.setEnabled(True)

    def show_result (self):
        self.next_window = optimisation_results.Window(self.component)
        self.close()
        self.next_window.show()
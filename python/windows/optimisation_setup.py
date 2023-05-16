
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QListWidgetItem, QButtonGroup, QFileDialog
from PyQt6.QtCore import QAbstractTableModel, Qt, QThread
from PyQt6.QtGui import QStandardItem, QStandardItemModel

# from PySide6.QtCore import Qt

import os

from python import load, optimisation
from python.windows import optimisation_setup2, developer, optimisation_results, prediction, sensitivity

class Window(QMainWindow):
    def __init__ (self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        uic.loadUi('ui/optimisation_setup.ui', self)
        self.file = "temp/ubending.STL"

        # self.component_dropdown.addItems(load.components())
        self.component_dropdown.addItems(["U-bending", "Bulkhead"])

        self.component_dropdown.currentTextChanged.connect(self.update_process_dropdown)
        self.process_dropdown.currentTextChanged.connect(self.update_material_dropdown)
        self.material_dropdown.currentTextChanged.connect(self.update_indicator_dropdown)
        # self.indicator_dropdown.currentTextChanged.connect(self.select_model)

        self.load_mesh_button.pressed.connect(lambda: self.load_mesh())

        self.objfunc_dropdown.addItems(["Chamfer Distance"])

        self.optimiser_dropdown.addItems(["ADAM Gradient Descent"])

        self.next_button.pressed.connect(self.begin_optimisation)

        # add items to a checkable items list
        # stringlist = ["one","two","three"]
        # self.variables_model = QStandardItemModel()
        # for i in range(len(stringlist)):
        #     item = QStandardItem(stringlist[i])
        #     item.setCheckable(True)
        #     # check = Qt.Checked if checked else QtCore.Qt.Unchecked
        #     # item.setCheckState(check)
        #     self.variables_model.appendRow(item)
        #     item = QListWidgetItem(self.variables_listwidget)
        #     item.setText(stringlist[i])
        #     # item.setFlags(item.flags())
        #     item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        #     item.setCheckState(Qt.CheckState.Unchecked)
        #     # item.setCheckable(True)
        # self.variables_listwidget.itemClicked.connect(self.update_goals)

        self.goal_buttongroup = QButtonGroup()
        self.goal_buttongroup.addButton(self.minimise_checkbox)
        self.goal_buttongroup.addButton(self.maximise_checkbox)
        self.goal_buttongroup.addButton(self.setto_checkbox)

        # Set the button group to be exclusive
        self.goal_buttongroup.setExclusive(True)
        
        self.update_process_dropdown()

        self.action_prediction.triggered.connect(self.open_prediction_window)
        self.action_optimisation.triggered.connect(self.open_optimisaiton_window)
        self.action_sensitivity.triggered.connect(self.open_sensitivity_window)
        self.action_developer.triggered.connect(self.open_developer_window)

    def update_process_dropdown (self):
        self.process_dropdown.clear()
        self.process_dropdown.addItems(load.processes(self.component_dropdown.currentText()))

    def update_material_dropdown (self):
        self.material_dropdown.clear()
        self.material_dropdown.addItems(load.materials(self.component_dropdown.currentText(), self.process_dropdown.currentText()))

    def update_indicator_dropdown (self):
        self.indicator_dropdown.clear()
        self.indicator_dropdown.addItems(load.indicators(self.component_dropdown.currentText(), self.process_dropdown.currentText(), self.material_dropdown.currentText()))

    def update_goals (self, item):
        text = item.text()
        if item.checkState() == Qt.CheckState.Checked:
            if self.goal_dropdown.findText(text) == -1:
                self.goal_dropdown.addItem(text)

        elif item.checkState() == Qt.CheckState.Unchecked:
            self.goal_dropdown.removeItem(self.goal_dropdown.findText(text))

    def load_mesh (self):
        self.file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="STL file (*.stl);; STEP file (*.step)")[0]
        selected_component = self.component_dropdown.currentText()
        self.progress_label.setText("Geometry loading...")
        # optimisation.load_mesh(self.file, self)

        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = optimisation.load_mesh_worker(self.file, self, selected_component)
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

    def report_finished (self):
        self.progress_label.setText("Geometry loaded successfully!")
        self.progress_bar.setValue(100)
        self.next_button.setEnabled(True)

    def report_progress (self, progress):
        self.progress_bar.setValue(progress)

    def begin_optimisation (self):
        self.next_window = optimisation_setup2.Window(num_iterations=self.runsno_value.value(), file=self.file, active_component=self.component_dropdown.currentText())
        self.close()
        self.next_window.show()
            
    def open_prediction_window (self):
        self.prediction_window = prediction.Window()
        self.close()
        self.prediction_window.show()   

    def open_optimisaiton_window (self):
        self.optimisation_window = optimisation_results.Window()
        self.close()
        self.optimisation_window.show()

    def open_sensitivity_window (self):
        self.sensitivity_window = sensitivity.Window()
        self.close()
        self.sensitivity_window.show()

    def open_developer_window (self):
        self.developer_window = developer.DeveloperWindow()
        self.close()
        self.developer_window.show()
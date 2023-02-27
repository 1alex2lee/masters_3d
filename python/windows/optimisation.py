
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QListWidgetItem, QButtonGroup
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel

# from PySide6.QtCore import Qt

import os

from python import load

class SetupWindow(QMainWindow):
    def __init__ (self, *args, **kwargs):
        super(SetupWindow, self).__init__(*args, **kwargs)
        uic.loadUi('ui/optimisation_setup.ui', self)

        self.process_dropdown.addItems(load.processes())
        self.material_dropdown.addItems(load.materials(self.process_dropdown.currentText()))
        self.process_dropdown.currentIndexChanged.connect(self.update_material)

        self.model_dropdown.addItems(load.modeltypes(self.process_dropdown.currentText(), self.material_dropdown.currentText()))
        self.material_dropdown.currentIndexChanged.connect(self.update_model)

        stringlist = ["one","two","three"]

        self.variables_model = QStandardItemModel()
        
        for i in range(len(stringlist)):
            item = QStandardItem(stringlist[i])
            item.setCheckable(True)
            # check = Qt.Checked if checked else QtCore.Qt.Unchecked
            # item.setCheckState(check)
            self.variables_model.appendRow(item)
            item = QListWidgetItem(self.variables_listwidget)
            item.setText(stringlist[i])
            # item.setFlags(item.flags())
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            # item.setCheckable(True)

        self.variables_listwidget.itemClicked.connect(self.update_goals)

        self.goal_buttongroup = QButtonGroup()
        self.goal_buttongroup.addButton(self.minimise_checkbox)
        self.goal_buttongroup.addButton(self.maximise_checkbox)
        self.goal_buttongroup.addButton(self.setto_checkbox)

        # Set the button group to be exclusive
        self.goal_buttongroup.setExclusive(True)

    def update_material (self):
        self.material_dropdown.clear()
        self.material_dropdown.addItems(load.materials(self.process_dropdown.currentText()))

    def update_model (self):
        self.model_dropdown.clear()
        self.model_dropdown.addItems(load.modeltypes(self.process_dropdown.currentText(), self.material_dropdown.currentText()))

    def update_goals (self, item):
        text = item.text()
        if item.checkState() == Qt.CheckState.Checked:
            if self.goal_dropdown.findText(text) == -1:
                self.goal_dropdown.addItem(text)

        elif item.checkState() == Qt.CheckState.Unchecked:
            self.goal_dropdown.removeItem(self.goal_dropdown.findText(text))


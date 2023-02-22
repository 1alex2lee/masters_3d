
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QAbstractTableModel

from PySide6.QtCore import Qt

import os

from windows.train_new import TrainNewWindow

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])


class DeveloperWindow(QMainWindow):
    def __init__ (self, *args, **kwargs):
        super(DeveloperWindow, self).__init__(*args, **kwargs)
        uic.loadUi('developer.ui', self)

        # self.model_tree.setHeaderHidden(True)
        # self.trained_models_model = QStandardItemModel()
        # root_node = self.trained_models_model.invisibleRootItem()

        trained_models = []

        for process in os.listdir("models"):
            # process_item = TreeItem(process)
            # root_node.appendRow(process_item)
            for material in os.listdir(os.path.join("models", process)):
                if material[0] != ".":
                    # material_item = TreeItem(material)
                    # process_item.appendRow(material_item)
                    for target in os.listdir(os.path.join("models", process, material)):
                        # material_item.appendRow(TreeItem(target))
                        trained_models.append([process, material, target])

        # self.model_tree.setModel(self.trained_models_model)
        # self.model_tree.expandAll()
        # self.model_tree.doubleClicked.connect(self.rename_treeitem)
        # self.model_tree.currentItemChanged.connect(self.rename_treeitem)

        self.trained_models_model = TableModel(trained_models)
        self.model_table.setModel(self.trained_models_model)
        self.model_table.setShowGrid(False)

        self.delete_button.pressed.connect(lambda: self.delete_treeitem(self.model_tree.selectedIndexes()))

        self.train_new_button.pressed.connect(self.open_trainnew_window)

    def delete_treeitem (self, index):
        item = index[0].model().itemFromIndex(index[0]).text()
        print(index[0].column())
        self.trained_models_model.removeRow(index[0].row())
        self.model_tree.setModel(self.trained_models_model)
        self.model_tree.expandAll()
        # index = index[0]
        # print("delete ", index.model().itemFromIndex(index))
        
    def open_trainnew_window (self):
        self.trainnew_window = TrainNewWindow()
        self.trainnew_window.show()
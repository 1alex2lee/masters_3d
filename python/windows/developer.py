
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QProgressDialog, QTreeWidgetItem
from PyQt6.QtCore import QAbstractTableModel
from PyQt6.QtGui import QStandardItemModel, QIcon

from PySide6.QtCore import Qt

import os

from python import model_control

# from windows.train_new import TrainNewWindow

class TrainNewWindow (QMainWindow):
    def __init__(self):
        super(TrainNewWindow, self).__init__()
        uic.loadUi("ui/trainnew.ui", self)

        self.input_dir = self.output_dir = ""

        self.input_button.pressed.connect(self.load_input)
        self.output_button.pressed.connect(self.load_output)

        self.epochs_slider.valueChanged.connect(self.epochs_changed)
        self.batchsize_slider.valueChanged.connect(self.batchsize_changed)

        self.cancel_button.pressed.connect(self.close)

        self.begin_button.pressed.connect(lambda: self.begin_training(self.name_edit.text(), 
        self.material_edit.text(), self.target_edit.text(), self.epochs_slider.value(),
        self.batchsize_slider.value()))
        
    def load_input (self):
        self.input_dir = QFileDialog().getExistingDirectory(self, "Choose input directory")
        self.input_dir_label.setText(self.input_dir+" selected")
        print(self.input_dir)

    def load_output (self):
        self.output_dir = QFileDialog().getExistingDirectory(self, "Choose output directory")
        self.output_dir_label.setText(self.output_dir+" selected")
        print(self.output_dir)
    
    def begin_training (self, name, material, target, epochs, batch_size):
        print(name, material, target, epochs, batch_size)
        # if name == "":
        #     error = QMessageBox()
        #     error.setText("No name entered")
        #     error.exec()
        # elif material == "":
        #     error = QMessageBox()
        #     error.setText("No material entered")
        #     error.exec()
        # elif target == "":
        #     error = QMessageBox()
        #     error.setText("No target entered")
        #     error.exec()
        # elif self.input_dir == "":
        #     error = QMessageBox()
        #     error.setText("No input directory selected")
        #     error.exec()
        # elif self.output_dir == "":
        #     error = QMessageBox()
        #     error.setText("No output directory entered")
        #     error.exec()
        # else:
        self.progress = QProgressDialog("Training Model", "Stop Training", 0, epochs, self)
        self.progress.open()

        model_control.begin_training(self, name, material, target, epochs, batch_size, self.input_dir, self.output_dir)

    def epochs_changed (self, val):
        self.epochs_number.setText(str(val))

    def batchsize_changed (self, val):
        self.batchsize_number.setText(str(val))
        

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
        uic.loadUi('ui/developer.ui', self)

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

        self.load_project_structure ("models", self.model_tree)

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

    def load_project_structure (self, startpath, tree):
        for element in os.listdir(startpath):
            if element[0] != ".":
                path_info = startpath + "/" + element
                parent_itm = QTreeWidgetItem(tree, [os.path.basename(element)])
                if os.path.isdir(path_info):
                    self.load_project_structure(path_info, parent_itm)


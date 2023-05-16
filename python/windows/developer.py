
import typing
from PyQt6 import QtCore, uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QProgressDialog, QTreeWidgetItem, QWidget
from PyQt6.QtCore import QAbstractTableModel, QThread, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QIcon

from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import os

from python import model_control
from python.developer_funcs import script1, script2, die_images, ReSEUNet_training
from python.windows import optimisation_results, optimisation_setup, prediction, sensitivity

# from windows.train_new import TrainNewWindow

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class TrainProgressWindow (QMainWindow):
    def __init__(self, component, process, material, indicator, input_dir, target_dir, epochs_num, batch_size):
        super(TrainProgressWindow, self).__init__()
        uic.loadUi("ui/trainnew_progress.ui", self)
        
        self.component = component
        self.process = process
        self.material = material
        self.indicator = indicator
        self.input_dir = input_dir
        self.epochs_num = epochs_num
        self.batch_size = batch_size

        self.begin_script1(component, target_dir)

        self.cancel_button.pressed.connect(self.cancel)
        self.done_button.pressed.connect(self.close)

        self.canvas = MplCanvas(self, width=5, height=5, dpi=100)
        # self.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.grid.addWidget(self.canvas)

    stop = pyqtSignal()

    def begin_script1 (self, component, target_dir):
        # Step 2: Create a QThread object
        self.thread1 = QThread()
        # Step 3: Create a worker object
        self.worker = script1.worker(component, target_dir, self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread1)
        # Step 5: Connect signals and slots
        self.thread1.started.connect(self.worker.run)

        self.worker.finished.connect(self.script1_finished)
        self.worker.finished.connect(self.thread1.quit)
        self.thread1.finished.connect(self.thread1.deleteLater)

        self.worker.progress.connect(self.script1_progress)
        # Step 6: Start the thread
        self.thread1.start()

    def script1_finished (self):
        print("result image load complete")
        self.resultimage_progressbar.setValue(100)
        self.begin_script2(self.component, self.input_dir)

    def script1_progress (self, prog):
        self.resultimage_progressbar.setValue(prog)

    def begin_script2 (self, component, input_dir):
        # Step 2: Create a QThread object
        self.thread2 = QThread()
        # Step 3: Create a worker object
        self.worker = script2.worker(component, input_dir, self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread2)
        # Step 5: Connect signals and slots
        self.thread2.started.connect(self.worker.run)

        self.worker.finished.connect(self.script2_finished)
        self.worker.finished.connect(self.thread2.quit)
        self.thread2.finished.connect(self.thread2.deleteLater)

        self.worker.progress.connect(self.script2_progress)
        # Step 6: Start the thread
        self.thread2.start()

    def script2_finished (self):
        self.script2_progressbar.setValue(100)
        self.begin_dieimages(self.component, self.input_dir)

    def script2_progress (self, prog):
        self.script2_progressbar.setValue(prog)

    def begin_dieimages (self, component, input_dir):
        # Step 2: Create a QThread object
        self.thread3 = QThread()
        # Step 3: Create a worker object
        self.worker = die_images.worker(component, input_dir, self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread3)
        # Step 5: Connect signals and slots
        self.thread3.started.connect(self.worker.run)

        self.worker.finished.connect(self.dieimages_finished)
        self.worker.finished.connect(self.thread3.quit)
        self.thread3.finished.connect(self.thread3.deleteLater)

        self.worker.progress.connect(self.dieimages_progress)
        # Step 6: Start the thread
        self.thread3.start()

    def dieimages_finished (self):
        self.dieimages_progressbar.setValue(100)
        self.status_label.setText("Training model...")
        self.begin_train_disp(self.component, self.process, self.material, "Displacement", self.epochs_num, self.batch_size)

    def dieimages_progress (self, prog):
        self.dieimages_progressbar.setValue(prog)

    def begin_train_disp (self, component, process, material, indicator, epochs_num, batch_size):
        # Step 2: Create a QThread object
        self.thread4 = QThread()
        # Step 3: Create a worker object
        self.worker = ReSEUNet_training.worker(component, process, material, indicator, epochs_num, batch_size, self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread4)
        # Step 5: Connect signals and slots
        self.thread4.started.connect(self.worker.run)

        self.worker.finished.connect(self.train_disp_finished)
        self.worker.finished.connect(self.thread4.quit)
        self.thread4.finished.connect(self.thread4.deleteLater)

        self.worker.progress.connect(self.train_disp_progress)
        # Step 6: Start the thread
        self.thread4.start()

    def train_disp_finished (self):
        self.disp_progressbar.setValue(100)
        self.begin_train_thinning(self.component, self.process, self.material, "Thinning", self.epochs_num, self.batch_size)

    def train_disp_progress (self, prog):
        self.disp_progressbar.setValue(prog)

    def begin_train_thinning (self, component, process, material, indicator, epochs_num, batch_size):
        # Step 2: Create a QThread object
        self.thread5 = QThread()
        # Step 3: Create a worker object
        self.worker = ReSEUNet_training.worker(component, process, material, indicator, epochs_num, batch_size, self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread5)
        # Step 5: Connect signals and slots
        self.thread5.started.connect(self.worker.run)

        self.worker.finished.connect(self.train_thinning_finished)
        self.worker.finished.connect(self.thread5.quit)
        self.thread5.finished.connect(self.thread5.deleteLater)

        self.worker.progress.connect(self.train_thinning_progress)
        # Step 6: Start the thread
        self.thread5.start()

    def train_thinning_finished (self):
        self.thinning_progressbar.setValue(100)
        self.status_label.setText("Training complete!")
        self.done_button.setEnabled(True)

    def train_thinning_progress (self, prog):
        self.thinning_progressbar.setValue(prog)

    def cancel (self):
        self.stop.emit()
        self.status_label.setText("Training cancelled!")

class TrainNewWindow (QMainWindow):
    def __init__(self):
        super(TrainNewWindow, self).__init__()
        uic.loadUi("ui/trainnew_setup.ui", self)

        self.input_dir = self.target_dir = ""

        self.input_button.pressed.connect(self.load_input)
        self.target_button.pressed.connect(self.load_output)

        self.input_selected = False
        self.target_selected = False

        self.epochs_slider.valueChanged.connect(self.epochs_changed)
        self.batchsize_slider.valueChanged.connect(self.batchsize_changed)

        self.component_dropdown.addItems(["Bulkhead", "U-bending"])

        self.cancel_button.pressed.connect(self.close)

        self.begin_button.pressed.connect(self.begin_training)
        
    def load_input (self):
        self.input_dir = QFileDialog().getExistingDirectory(self, "Choose input directory")
        self.input_dir_label.setText(self.input_dir.split("/")[-1] + " selected")
        print(self.input_dir)
        if self.target_selected:
            self.begin_button.setEnabled(True)
        self.input_selected = True

    def load_output (self):
        self.target_dir = QFileDialog().getExistingDirectory(self, "Choose output directory")
        self.target_dir_label.setText(self.target_dir.split("/")[-1] + " selected")
        print(self.target_dir)
        if self.input_selected:
            self.begin_button.setEnabled(True)
        self.target_selected = True

    def begin_training (self):
        component = self.component_dropdown.currentText().lower()
        process = self.process_edit.text()
        material = self.process_edit.text()
        indicator = self.indicator_edit.text()
        input_dir = self.input_dir
        target_dir = self.target_dir
        epochs_num = self.epochs_slider.value()
        batch_size = self.batchsize_slider.value()

        self.next_window = TrainProgressWindow(component, process, material, indicator, input_dir, target_dir, epochs_num, batch_size)
        self.close()
        self.next_window.show()

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

        for component in [c for c in os.listdir("components") if c[0] != "."]:
            # process_item = TreeItem(process)
            # root_node.appendRow(process_item)
            for material in [m for m in os.listdir(os.path.join("components", component)) if m[0] != "."]:
                # material_item = TreeItem(material)
                # process_item.appendRow(material_item)
                for target in [t for t in os.listdir(os.path.join("components", component, material)) if t[0] != "."]:
                    # material_item.appendRow(TreeItem(target))
                    trained_models.append([component, material, target])

        # self.model_tree.setModel(self.trained_models_model)
        # self.model_tree.expandAll()
        # self.model_tree.doubleClicked.connect(self.rename_treeitem)
        # self.model_tree.currentItemChanged.connect(self.rename_treeitem)

        self.load_project_structure ("components", self.model_tree)

        self.trained_models_model = TableModel(trained_models)
        self.model_table.setModel(self.trained_models_model)
        self.model_table.setShowGrid(False)

        self.delete_button.pressed.connect(lambda: self.delete_treeitem(self.model_tree.selectedIndexes()))

        self.train_new_button.pressed.connect(self.open_trainnew_window)

        self.action_prediction.triggered.connect(self.open_prediction_window)
        self.action_newoptimisation.triggered.connect(self.new_optimisaiton_window)
        self.action_optimisation.triggered.connect(self.open_optimisaiton_window)
        self.action_sensitivity.triggered.connect(self.open_sensitivity_window)

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

    def open_prediction_window (self):
        self.prediction_window = prediction.Window()
        self.close()
        self.prediction_window.show()   

    def new_optimisaiton_window (self):
        self.optimisation_window = optimisation_setup.Window()
        self.close()
        self.optimisation_window.show()   

    def open_optimisaiton_window (self):
        self.optimisation_window = optimisation_results.Window()
        self.close()
        self.optimisation_window.show()

    def open_sensitivity_window (self):
        self.sensitivity_window = sensitivity.Window()
        self.close()
        self.sensitivity_window.show()
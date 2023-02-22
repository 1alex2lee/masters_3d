from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QProgressDialog

from python import model_control

class TrainNewWindow (QMainWindow):
    def __init__(self):
        super(TrainNewWindow, self).__init__()
        uic.loadUi("trainnew.ui", self)

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
        if name == "":
            error = QMessageBox()
            error.setText("No name entered")
            error.exec()
        elif material == "":
            error = QMessageBox()
            error.setText("No material entered")
            error.exec()
        elif target == "":
            error = QMessageBox()
            error.setText("No target entered")
            error.exec()
        elif self.input_dir == "":
            error = QMessageBox()
            error.setText("No input directory selected")
            error.exec()
        elif self.output_dir == "":
            error = QMessageBox()
            error.setText("No output directory entered")
            error.exec()
        else:
            QProgressDialog.show(self)
            model_control.begin_training(self, name, material, target, epochs, batch_size, self.input_dir, self.output_dir)

    def epochs_changed (self, val):
        self.epochs_number.setText(str(val))
    def batchsize_changed (self, val):
        self.batchsize_number.setText(str(val))
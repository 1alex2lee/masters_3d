
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel

# from PySide6.QtCore import Qt

import os

from python import optimisation

class OptimisationWindow(QMainWindow):
  def __init__ (self, *args, **kwargs):
    super(OptimisationWindow, self).__init__(*args, **kwargs)
    uic.loadUi('ui/optimisation.ui', self)

    self.load_mesh_button.pressed.connect(lambda: self.load_mesh())

  def load_mesh (self):
    file = QFileDialog.getOpenFileName(self, "Import Mesh", filter="STL file (*.stl);; STEP file (*.step)")
    optimisation.load_mesh(file[0], self)


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHeaderView,
    QLabel, QLayout, QMainWindow, QMenu,
    QMenuBar, QPushButton, QRadioButton, QSizePolicy,
    QSlider, QStatusBar, QTreeView, QWidget)

from pyqtgraph.opengl import GLViewWidget

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(1174, 895)
        self.actionImport_New_Mesh = QAction(main_window)
        self.actionImport_New_Mesh.setObjectName(u"actionImport_New_Mesh")
        self.actionOptimisation_Mode = QAction(main_window)
        self.actionOptimisation_Mode.setObjectName(u"actionOptimisation_Mode")
        self.actionSensitiity_Mode = QAction(main_window)
        self.actionSensitiity_Mode.setObjectName(u"actionSensitiity_Mode")
        self.action_developer = QAction(main_window)
        self.action_developer.setObjectName(u"action_developer")
        self.actionExit = QAction(main_window)
        self.actionExit.setObjectName(u"actionExit")
        self.central_widget = QWidget(main_window)
        self.central_widget.setObjectName(u"central_widget")
        self.central_widget.setEnabled(True)
        self.central_widget.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_2 = QGridLayout(self.central_widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.springback_button = QRadioButton(self.central_widget)
        self.springback_button.setObjectName(u"springback_button")
        self.springback_button.setEnabled(True)

        self.grid_layout.addWidget(self.springback_button, 0, 3, 1, 1)

        self.strain_button = QRadioButton(self.central_widget)
        self.strain_button.setObjectName(u"strain_button")
        self.strain_button.setEnabled(True)

        self.grid_layout.addWidget(self.strain_button, 0, 4, 1, 1)

        self.material_label = QLabel(self.central_widget)
        self.material_label.setObjectName(u"material_label")

        self.grid_layout.addWidget(self.material_label, 1, 0, 1, 1)

        self.selected_label = QLabel(self.central_widget)
        self.selected_label.setObjectName(u"selected_label")
        self.selected_label.setMaximumSize(QSize(16777215, 24))
        self.selected_label.setAlignment(Qt.AlignCenter)

        self.grid_layout.addWidget(self.selected_label, 1, 2, 1, 3)

        self.process_label = QLabel(self.central_widget)
        self.process_label.setObjectName(u"process_label")

        self.grid_layout.addWidget(self.process_label, 0, 0, 1, 1)

        self.load_mesh_button = QPushButton(self.central_widget)
        self.load_mesh_button.setObjectName(u"load_mesh_button")

        self.grid_layout.addWidget(self.load_mesh_button, 3, 0, 1, 2)

        self.dir_tree = QTreeView(self.central_widget)
        self.dir_tree.setObjectName(u"dir_tree")
        self.dir_tree.setEnabled(True)
        self.dir_tree.setMaximumSize(QSize(16777215, 16777215))

        self.grid_layout.addWidget(self.dir_tree, 4, 0, 1, 2)

        self.material_dropdown = QComboBox(self.central_widget)
        self.material_dropdown.setObjectName(u"material_dropdown")

        self.grid_layout.addWidget(self.material_dropdown, 1, 1, 1, 1)

        self.thinning_button = QRadioButton(self.central_widget)
        self.thinning_button.setObjectName(u"thinning_button")
        self.thinning_button.setEnabled(True)

        self.grid_layout.addWidget(self.thinning_button, 0, 2, 1, 1)

        self.process_dropdown = QComboBox(self.central_widget)
        self.process_dropdown.setObjectName(u"process_dropdown")

        self.grid_layout.addWidget(self.process_dropdown, 0, 1, 1, 1)

        self.slider_label = QLabel(self.central_widget)
        self.slider_label.setObjectName(u"slider_label")

        self.grid_layout.addWidget(self.slider_label, 2, 0, 1, 1)

        self.main_view = GLViewWidget(self.central_widget)
        self.main_view.setObjectName(u"main_view")
        self.main_view.setEnabled(True)
        self.main_view.setMinimumSize(QSize(900, 0))

        self.grid_layout.addWidget(self.main_view, 2, 2, 3, 3)

        self.slider = QSlider(self.central_widget)
        self.slider.setObjectName(u"slider")
        self.slider.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.slider, 2, 1, 1, 1)


        self.gridLayout_2.addLayout(self.grid_layout, 0, 1, 1, 1)

        main_window.setCentralWidget(self.central_widget)
        self.statusbar = QStatusBar(main_window)
        self.statusbar.setObjectName(u"statusbar")
        main_window.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(main_window)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 16777214, 24))
        self.menubar.setMinimumSize(QSize(16777214, 24))
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(False)
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menuMode")
        self.menuExit = QMenu(self.menubar)
        self.menuExit.setObjectName(u"menuExit")
        main_window.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuExit.menuAction())
        self.menuFile.addAction(self.actionImport_New_Mesh)
        self.menuMode.addAction(self.actionOptimisation_Mode)
        self.menuMode.addAction(self.actionSensitiity_Mode)
        self.menuMode.addAction(self.action_developer)
        self.menuExit.addAction(self.actionExit)

        self.retranslateUi(main_window)

        QMetaObject.connectSlotsByName(main_window)
    # setupUi

    def retranslateUi(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("main_window", u"User-Centric Software to Assist Design for Forming", None))
        self.actionImport_New_Mesh.setText(QCoreApplication.translate("main_window", u"Import New Mesh", None))
        self.actionOptimisation_Mode.setText(QCoreApplication.translate("main_window", u"Optimisation", None))
        self.actionSensitiity_Mode.setText(QCoreApplication.translate("main_window", u"Sensitiity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("main_window", u"Developer", None))
        self.actionExit.setText(QCoreApplication.translate("main_window", u"Exit", None))
        self.springback_button.setText(QCoreApplication.translate("main_window", u"Springback", None))
        self.strain_button.setText(QCoreApplication.translate("main_window", u"Major Strain", None))
        self.material_label.setText(QCoreApplication.translate("main_window", u"Material", None))
        self.selected_label.setText(QCoreApplication.translate("main_window", u"No model selected", None))
        self.process_label.setText(QCoreApplication.translate("main_window", u"Process", None))
        self.load_mesh_button.setText(QCoreApplication.translate("main_window", u"Load New Mesh", None))
        self.thinning_button.setText(QCoreApplication.translate("main_window", u"Thinning", None))
        self.slider_label.setText(QCoreApplication.translate("main_window", u"Slider Placeholder", None))
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


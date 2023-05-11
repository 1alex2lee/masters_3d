# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'trainnew.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QLineEdit,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QSlider, QStatusBar, QWidget)

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(373, 454)
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
        self.gridLayout = QGridLayout(self.central_widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.material_label = QLabel(self.central_widget)
        self.material_label.setObjectName(u"material_label")

        self.gridLayout.addWidget(self.material_label, 2, 0, 1, 1)

        self.batchsize_label = QLabel(self.central_widget)
        self.batchsize_label.setObjectName(u"batchsize_label")

        self.gridLayout.addWidget(self.batchsize_label, 15, 0, 1, 1)

        self.output_dir_label = QLabel(self.central_widget)
        self.output_dir_label.setObjectName(u"output_dir_label")

        self.gridLayout.addWidget(self.output_dir_label, 12, 0, 1, 2)

        self.name_label = QLabel(self.central_widget)
        self.name_label.setObjectName(u"name_label")

        self.gridLayout.addWidget(self.name_label, 0, 0, 1, 1)

        self.input_dir_label = QLabel(self.central_widget)
        self.input_dir_label.setObjectName(u"input_dir_label")

        self.gridLayout.addWidget(self.input_dir_label, 10, 0, 1, 2)

        self.cancel_button = QPushButton(self.central_widget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 17, 0, 1, 1)

        self.input_label = QLabel(self.central_widget)
        self.input_label.setObjectName(u"input_label")

        self.gridLayout.addWidget(self.input_label, 7, 0, 1, 1)

        self.target_label = QLabel(self.central_widget)
        self.target_label.setObjectName(u"target_label")

        self.gridLayout.addWidget(self.target_label, 5, 0, 1, 1)

        self.epochs_slider = QSlider(self.central_widget)
        self.epochs_slider.setObjectName(u"epochs_slider")
        self.epochs_slider.setMinimumSize(QSize(0, 0))
        self.epochs_slider.setMinimum(1)
        self.epochs_slider.setMaximum(20000)
        self.epochs_slider.setSingleStep(10)
        self.epochs_slider.setValue(10000)
        self.epochs_slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.epochs_slider, 13, 1, 1, 1)

        self.output_label = QLabel(self.central_widget)
        self.output_label.setObjectName(u"output_label")

        self.gridLayout.addWidget(self.output_label, 11, 0, 1, 1)

        self.batchsize_slider = QSlider(self.central_widget)
        self.batchsize_slider.setObjectName(u"batchsize_slider")
        self.batchsize_slider.setMinimum(1)
        self.batchsize_slider.setMaximum(120)
        self.batchsize_slider.setSingleStep(1)
        self.batchsize_slider.setValue(60)
        self.batchsize_slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.batchsize_slider, 15, 1, 1, 1)

        self.epochs_label = QLabel(self.central_widget)
        self.epochs_label.setObjectName(u"epochs_label")

        self.gridLayout.addWidget(self.epochs_label, 13, 0, 1, 1)

        self.eta_label = QLabel(self.central_widget)
        self.eta_label.setObjectName(u"eta_label")

        self.gridLayout.addWidget(self.eta_label, 16, 0, 1, 2)

        self.epochs_number = QLabel(self.central_widget)
        self.epochs_number.setObjectName(u"epochs_number")
        self.epochs_number.setMinimumSize(QSize(40, 0))

        self.gridLayout.addWidget(self.epochs_number, 13, 2, 1, 1)

        self.batchsize_number = QLabel(self.central_widget)
        self.batchsize_number.setObjectName(u"batchsize_number")

        self.gridLayout.addWidget(self.batchsize_number, 15, 2, 1, 1)

        self.target_edit = QLineEdit(self.central_widget)
        self.target_edit.setObjectName(u"target_edit")

        self.gridLayout.addWidget(self.target_edit, 5, 1, 1, 2)

        self.material_edit = QLineEdit(self.central_widget)
        self.material_edit.setObjectName(u"material_edit")

        self.gridLayout.addWidget(self.material_edit, 2, 1, 1, 2)

        self.name_edit = QLineEdit(self.central_widget)
        self.name_edit.setObjectName(u"name_edit")

        self.gridLayout.addWidget(self.name_edit, 0, 1, 1, 2)

        self.output_button = QPushButton(self.central_widget)
        self.output_button.setObjectName(u"output_button")

        self.gridLayout.addWidget(self.output_button, 11, 1, 1, 1)

        self.input_button = QPushButton(self.central_widget)
        self.input_button.setObjectName(u"input_button")

        self.gridLayout.addWidget(self.input_button, 7, 1, 1, 1)

        self.begin_button = QPushButton(self.central_widget)
        self.begin_button.setObjectName(u"begin_button")

        self.gridLayout.addWidget(self.begin_button, 17, 1, 1, 1)

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
        self.material_label.setText(QCoreApplication.translate("main_window", u"Material", None))
        self.batchsize_label.setText(QCoreApplication.translate("main_window", u"Batch Size", None))
        self.output_dir_label.setText(QCoreApplication.translate("main_window", u"No folder selected", None))
        self.name_label.setText(QCoreApplication.translate("main_window", u"Model Name", None))
        self.input_dir_label.setText(QCoreApplication.translate("main_window", u"No folder selected", None))
        self.cancel_button.setText(QCoreApplication.translate("main_window", u"Cancel", None))
        self.input_label.setText(QCoreApplication.translate("main_window", u"Input Data Directory", None))
        self.target_label.setText(QCoreApplication.translate("main_window", u"Target", None))
        self.output_label.setText(QCoreApplication.translate("main_window", u"Output Data Directory", None))
        self.epochs_label.setText(QCoreApplication.translate("main_window", u"Epochs", None))
        self.eta_label.setText(QCoreApplication.translate("main_window", u"Estimated training time:", None))
        self.epochs_number.setText(QCoreApplication.translate("main_window", u"10000", None))
        self.batchsize_number.setText(QCoreApplication.translate("main_window", u"60", None))
        self.output_button.setText(QCoreApplication.translate("main_window", u"Browse", None))
        self.input_button.setText(QCoreApplication.translate("main_window", u"Browse", None))
        self.begin_button.setText(QCoreApplication.translate("main_window", u"Begin Training", None))
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


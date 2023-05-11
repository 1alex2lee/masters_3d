# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'trainnew_setup.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSlider,
    QStatusBar, QWidget)

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(428, 487)
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
        self.cancel_button = QPushButton(self.central_widget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 20, 0, 1, 1)

        self.begin_button = QPushButton(self.central_widget)
        self.begin_button.setObjectName(u"begin_button")
        self.begin_button.setEnabled(False)

        self.gridLayout.addWidget(self.begin_button, 20, 1, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.name_label = QLabel(self.central_widget)
        self.name_label.setObjectName(u"name_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.name_label)

        self.material_label = QLabel(self.central_widget)
        self.material_label.setObjectName(u"material_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.material_label)

        self.process_edit = QLineEdit(self.central_widget)
        self.process_edit.setObjectName(u"process_edit")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.process_edit)

        self.target_label = QLabel(self.central_widget)
        self.target_label.setObjectName(u"target_label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.target_label)

        self.material_edit = QLineEdit(self.central_widget)
        self.material_edit.setObjectName(u"material_edit")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.material_edit)

        self.indicator_label = QLabel(self.central_widget)
        self.indicator_label.setObjectName(u"indicator_label")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.indicator_label)

        self.indicator_edit = QLineEdit(self.central_widget)
        self.indicator_edit.setObjectName(u"indicator_edit")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.indicator_edit)

        self.input_label = QLabel(self.central_widget)
        self.input_label.setObjectName(u"input_label")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.input_label)

        self.input_button = QPushButton(self.central_widget)
        self.input_button.setObjectName(u"input_button")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.input_button)

        self.input_dir_label = QLabel(self.central_widget)
        self.input_dir_label.setObjectName(u"input_dir_label")
        self.input_dir_label.setAlignment(Qt.AlignCenter)

        self.formLayout.setWidget(6, QFormLayout.SpanningRole, self.input_dir_label)

        self.output_label = QLabel(self.central_widget)
        self.output_label.setObjectName(u"output_label")

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.output_label)

        self.output_button = QPushButton(self.central_widget)
        self.output_button.setObjectName(u"output_button")

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.output_button)

        self.output_dir_label = QLabel(self.central_widget)
        self.output_dir_label.setObjectName(u"output_dir_label")
        self.output_dir_label.setAlignment(Qt.AlignCenter)

        self.formLayout.setWidget(8, QFormLayout.SpanningRole, self.output_dir_label)

        self.epochs_label = QLabel(self.central_widget)
        self.epochs_label.setObjectName(u"epochs_label")

        self.formLayout.setWidget(10, QFormLayout.LabelRole, self.epochs_label)

        self.epochs_number = QLabel(self.central_widget)
        self.epochs_number.setObjectName(u"epochs_number")
        self.epochs_number.setMinimumSize(QSize(40, 0))

        self.formLayout.setWidget(10, QFormLayout.FieldRole, self.epochs_number)

        self.epochs_slider = QSlider(self.central_widget)
        self.epochs_slider.setObjectName(u"epochs_slider")
        self.epochs_slider.setMinimumSize(QSize(0, 0))
        self.epochs_slider.setMinimum(1)
        self.epochs_slider.setMaximum(20000)
        self.epochs_slider.setSingleStep(10)
        self.epochs_slider.setValue(10000)
        self.epochs_slider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(11, QFormLayout.SpanningRole, self.epochs_slider)

        self.batchsize_label = QLabel(self.central_widget)
        self.batchsize_label.setObjectName(u"batchsize_label")

        self.formLayout.setWidget(12, QFormLayout.LabelRole, self.batchsize_label)

        self.batchsize_number = QLabel(self.central_widget)
        self.batchsize_number.setObjectName(u"batchsize_number")

        self.formLayout.setWidget(12, QFormLayout.FieldRole, self.batchsize_number)

        self.batchsize_slider = QSlider(self.central_widget)
        self.batchsize_slider.setObjectName(u"batchsize_slider")
        self.batchsize_slider.setMinimumSize(QSize(0, 0))
        self.batchsize_slider.setMinimum(1)
        self.batchsize_slider.setMaximum(120)
        self.batchsize_slider.setSingleStep(1)
        self.batchsize_slider.setValue(60)
        self.batchsize_slider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(13, QFormLayout.SpanningRole, self.batchsize_slider)

        self.component_dropdown = QComboBox(self.central_widget)
        self.component_dropdown.setObjectName(u"component_dropdown")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.component_dropdown)


        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 2)

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
        self.cancel_button.setText(QCoreApplication.translate("main_window", u"Cancel", None))
        self.begin_button.setText(QCoreApplication.translate("main_window", u"Begin Training", None))
        self.name_label.setText(QCoreApplication.translate("main_window", u"Component", None))
        self.material_label.setText(QCoreApplication.translate("main_window", u"Process Name", None))
        self.target_label.setText(QCoreApplication.translate("main_window", u"Material Name", None))
        self.indicator_label.setText(QCoreApplication.translate("main_window", u"Performance Indicator Name", None))
        self.input_label.setText(QCoreApplication.translate("main_window", u"Input Data Directory", None))
        self.input_button.setText(QCoreApplication.translate("main_window", u"Browse", None))
        self.input_dir_label.setText(QCoreApplication.translate("main_window", u"No folder selected", None))
        self.output_label.setText(QCoreApplication.translate("main_window", u"Output Data Directory", None))
        self.output_button.setText(QCoreApplication.translate("main_window", u"Browse", None))
        self.output_dir_label.setText(QCoreApplication.translate("main_window", u"No folder selected", None))
        self.epochs_label.setText(QCoreApplication.translate("main_window", u"Epochs", None))
        self.epochs_number.setText(QCoreApplication.translate("main_window", u"10000", None))
        self.batchsize_label.setText(QCoreApplication.translate("main_window", u"Batch Size", None))
        self.batchsize_number.setText(QCoreApplication.translate("main_window", u"60", None))
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


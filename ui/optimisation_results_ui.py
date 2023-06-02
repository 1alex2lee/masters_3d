# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'optimisation_results.ui'
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
    QLabel, QLayout, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSlider,
    QStatusBar, QWidget)

from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph.opengl import GLViewWidget

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(1246, 895)
        self.actionImport_New_Mesh = QAction(main_window)
        self.actionImport_New_Mesh.setObjectName(u"actionImport_New_Mesh")
        self.action_prediction = QAction(main_window)
        self.action_prediction.setObjectName(u"action_prediction")
        self.action_sensitivity = QAction(main_window)
        self.action_sensitivity.setObjectName(u"action_sensitivity")
        self.action_developer = QAction(main_window)
        self.action_developer.setObjectName(u"action_developer")
        self.actionExit = QAction(main_window)
        self.actionExit.setObjectName(u"actionExit")
        self.action_newoptimisation = QAction(main_window)
        self.action_newoptimisation.setObjectName(u"action_newoptimisation")
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
        self.num_design_slider = QSlider(self.central_widget)
        self.num_design_slider.setObjectName(u"num_design_slider")
        self.num_design_slider.setEnabled(False)
        self.num_design_slider.setMaximumSize(QSize(300, 16777215))
        self.num_design_slider.setMinimum(1)
        self.num_design_slider.setMaximum(100)
        self.num_design_slider.setValue(50)
        self.num_design_slider.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.num_design_slider, 3, 0, 1, 1)

        self.load_results_button = QPushButton(self.central_widget)
        self.load_results_button.setObjectName(u"load_results_button")
        self.load_results_button.setEnabled(True)

        self.grid_layout.addWidget(self.load_results_button, 1, 0, 1, 1)

        self.GraphicsLayoutWidget = GraphicsLayoutWidget(self.central_widget)
        self.GraphicsLayoutWidget.setObjectName(u"GraphicsLayoutWidget")
        self.GraphicsLayoutWidget.setMaximumSize(QSize(120, 16777215))

        self.grid_layout.addWidget(self.GraphicsLayoutWidget, 0, 3, 7, 1)

        self.main_view = GLViewWidget(self.central_widget)
        self.main_view.setObjectName(u"main_view")
        self.main_view.setEnabled(True)
        self.main_view.setMinimumSize(QSize(300, 0))

        self.grid_layout.addWidget(self.main_view, 0, 2, 7, 1)

        self.component_label = QLabel(self.central_widget)
        self.component_label.setObjectName(u"component_label")
        self.component_label.setMaximumSize(QSize(16777215, 20))

        self.grid_layout.addWidget(self.component_label, 0, 0, 1, 1)

        self.num_design_label = QLabel(self.central_widget)
        self.num_design_label.setObjectName(u"num_design_label")
        self.num_design_label.setEnabled(False)
        self.num_design_label.setMaximumSize(QSize(16777215, 20))

        self.grid_layout.addWidget(self.num_design_label, 2, 0, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.indicator_label = QLabel(self.central_widget)
        self.indicator_label.setObjectName(u"indicator_label")
        self.indicator_label.setEnabled(False)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.indicator_label)

        self.indicator_dropdown = QComboBox(self.central_widget)
        self.indicator_dropdown.setObjectName(u"indicator_dropdown")
        self.indicator_dropdown.setEnabled(False)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.indicator_dropdown)

        self.direction_label = QLabel(self.central_widget)
        self.direction_label.setObjectName(u"direction_label")
        self.direction_label.setEnabled(False)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.direction_label)

        self.direction_dropdown = QComboBox(self.central_widget)
        self.direction_dropdown.setObjectName(u"direction_dropdown")
        self.direction_dropdown.setEnabled(False)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.direction_dropdown)

        self.refresh_button = QPushButton(self.central_widget)
        self.refresh_button.setObjectName(u"refresh_button")
        self.refresh_button.setEnabled(False)

        self.formLayout.setWidget(2, QFormLayout.SpanningRole, self.refresh_button)

        self.variables_label = QLabel(self.central_widget)
        self.variables_label.setObjectName(u"variables_label")
        self.variables_label.setEnabled(False)
        self.variables_label.setAlignment(Qt.AlignCenter)

        self.formLayout.setWidget(3, QFormLayout.SpanningRole, self.variables_label)

        self.label = QLabel(self.central_widget)
        self.label.setObjectName(u"label")
        self.label.setEnabled(False)

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label)

        self.label_3 = QLabel(self.central_widget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setEnabled(False)

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_3)

        self.label_4 = QLabel(self.central_widget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setEnabled(False)

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.label_4)

        self.label_2 = QLabel(self.central_widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setEnabled(False)

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.label_2)

        self.bhf_value = QLabel(self.central_widget)
        self.bhf_value.setObjectName(u"bhf_value")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.bhf_value)

        self.friction_value = QLabel(self.central_widget)
        self.friction_value.setObjectName(u"friction_value")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.friction_value)

        self.clearance_value = QLabel(self.central_widget)
        self.clearance_value.setObjectName(u"clearance_value")

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.clearance_value)

        self.thickness_value = QLabel(self.central_widget)
        self.thickness_value.setObjectName(u"thickness_value")

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.thickness_value)


        self.grid_layout.addLayout(self.formLayout, 4, 0, 1, 1)


        self.gridLayout_2.addLayout(self.grid_layout, 0, 0, 1, 1)

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
        self.menuMode.addAction(self.action_prediction)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_newoptimisation)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_sensitivity)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_developer)
        self.menuExit.addAction(self.actionExit)

        self.retranslateUi(main_window)

        QMetaObject.connectSlotsByName(main_window)
    # setupUi

    def retranslateUi(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("main_window", u"Optimisation Results", None))
        self.actionImport_New_Mesh.setText(QCoreApplication.translate("main_window", u"Import New Mesh", None))
        self.action_prediction.setText(QCoreApplication.translate("main_window", u"Prediction", None))
        self.action_sensitivity.setText(QCoreApplication.translate("main_window", u"Sensitiity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("main_window", u"Developer", None))
        self.actionExit.setText(QCoreApplication.translate("main_window", u"Exit", None))
        self.action_newoptimisation.setText(QCoreApplication.translate("main_window", u"New Optimisation", None))
        self.load_results_button.setText(QCoreApplication.translate("main_window", u"Load Optimisation Result", None))
        self.component_label.setText(QCoreApplication.translate("main_window", u"No Component loaded", None))
        self.num_design_label.setText(QCoreApplication.translate("main_window", u"No results loaded", None))
        self.indicator_label.setText(QCoreApplication.translate("main_window", u"Performance Indicator", None))
        self.direction_label.setText(QCoreApplication.translate("main_window", u"Displacement Direction", None))
        self.refresh_button.setText(QCoreApplication.translate("main_window", u"Refresh Mesh", None))
        self.variables_label.setText(QCoreApplication.translate("main_window", u"Optimised Variables", None))
        self.label.setText(QCoreApplication.translate("main_window", u"Blank Holding Force", None))
        self.label_3.setText(QCoreApplication.translate("main_window", u"Friction Coefficient", None))
        self.label_4.setText(QCoreApplication.translate("main_window", u"Thickness", None))
        self.label_2.setText(QCoreApplication.translate("main_window", u"Clearance", None))
        self.bhf_value.setText("")
        self.friction_value.setText("")
        self.clearance_value.setText("")
        self.thickness_value.setText("")
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


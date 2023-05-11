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
        self.action_optimisation = QAction(main_window)
        self.action_optimisation.setObjectName(u"action_optimisation")
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
        self.load_results_button = QPushButton(self.central_widget)
        self.load_results_button.setObjectName(u"load_results_button")
        self.load_results_button.setEnabled(True)

        self.grid_layout.addWidget(self.load_results_button, 1, 0, 1, 1)

        self.main_view = GLViewWidget(self.central_widget)
        self.main_view.setObjectName(u"main_view")
        self.main_view.setEnabled(True)
        self.main_view.setMinimumSize(QSize(300, 0))

        self.grid_layout.addWidget(self.main_view, 1, 2, 5, 1)

        self.GraphicsLayoutWidget = GraphicsLayoutWidget(self.central_widget)
        self.GraphicsLayoutWidget.setObjectName(u"GraphicsLayoutWidget")
        self.GraphicsLayoutWidget.setMaximumSize(QSize(120, 16777215))

        self.grid_layout.addWidget(self.GraphicsLayoutWidget, 1, 3, 5, 1)

        self.component_label = QLabel(self.central_widget)
        self.component_label.setObjectName(u"component_label")
        self.component_label.setMaximumSize(QSize(16777215, 20))

        self.grid_layout.addWidget(self.component_label, 0, 0, 1, 1)

        self.num_design_label = QLabel(self.central_widget)
        self.num_design_label.setObjectName(u"num_design_label")
        self.num_design_label.setMaximumSize(QSize(16777215, 20))

        self.grid_layout.addWidget(self.num_design_label, 2, 0, 1, 1)

        self.num_design_slider = QSlider(self.central_widget)
        self.num_design_slider.setObjectName(u"num_design_slider")
        self.num_design_slider.setMaximumSize(QSize(300, 16777215))
        self.num_design_slider.setMaximum(100)
        self.num_design_slider.setValue(100)
        self.num_design_slider.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.num_design_slider, 3, 0, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.indicator_label = QLabel(self.central_widget)
        self.indicator_label.setObjectName(u"indicator_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.indicator_label)

        self.indicator_dropdown = QComboBox(self.central_widget)
        self.indicator_dropdown.setObjectName(u"indicator_dropdown")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.indicator_dropdown)

        self.direction_label = QLabel(self.central_widget)
        self.direction_label.setObjectName(u"direction_label")
        self.direction_label.setEnabled(False)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.direction_label)

        self.direction_dropdown = QComboBox(self.central_widget)
        self.direction_dropdown.setObjectName(u"direction_dropdown")
        self.direction_dropdown.setEnabled(False)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.direction_dropdown)


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
        self.menuMode.addAction(self.action_optimisation)
        self.menuMode.addAction(self.actionSensitiity_Mode)
        self.menuMode.addAction(self.action_developer)
        self.menuExit.addAction(self.actionExit)

        self.retranslateUi(main_window)

        QMetaObject.connectSlotsByName(main_window)
    # setupUi

    def retranslateUi(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("main_window", u"User-Centric Software to Assist Design for Forming", None))
        self.actionImport_New_Mesh.setText(QCoreApplication.translate("main_window", u"Import New Mesh", None))
        self.action_optimisation.setText(QCoreApplication.translate("main_window", u"Optimisation", None))
        self.actionSensitiity_Mode.setText(QCoreApplication.translate("main_window", u"Sensitiity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("main_window", u"Developer", None))
        self.actionExit.setText(QCoreApplication.translate("main_window", u"Exit", None))
        self.load_results_button.setText(QCoreApplication.translate("main_window", u"Load Optimisation Result", None))
        self.component_label.setText(QCoreApplication.translate("main_window", u"No Component loaded", None))
        self.num_design_label.setText(QCoreApplication.translate("main_window", u"No results loaded", None))
        self.indicator_label.setText(QCoreApplication.translate("main_window", u"Performance Indicator", None))
        self.direction_label.setText(QCoreApplication.translate("main_window", u"Displacement Direction", None))
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


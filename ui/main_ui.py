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
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout,
    QLabel, QLayout, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSlider,
    QStatusBar, QWidget)

from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph.opengl import GLViewWidget

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(1269, 895)
        self.actionImport_New_Mesh = QAction(main_window)
        self.actionImport_New_Mesh.setObjectName(u"actionImport_New_Mesh")
        self.action_newoptimisation = QAction(main_window)
        self.action_newoptimisation.setObjectName(u"action_newoptimisation")
        self.action_sensitivity = QAction(main_window)
        self.action_sensitivity.setObjectName(u"action_sensitivity")
        self.action_developer = QAction(main_window)
        self.action_developer.setObjectName(u"action_developer")
        self.actionExit = QAction(main_window)
        self.actionExit.setObjectName(u"actionExit")
        self.action_optimisation = QAction(main_window)
        self.action_optimisation.setObjectName(u"action_optimisation")
        self.central_widget = QWidget(main_window)
        self.central_widget.setObjectName(u"central_widget")
        self.central_widget.setEnabled(True)
        self.central_widget.setMaximumSize(QSize(16777215, 16777215))
        self.horizontalLayout = QHBoxLayout(self.central_widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.component_dropdown = QComboBox(self.central_widget)
        self.component_dropdown.setObjectName(u"component_dropdown")
        self.component_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.component_dropdown, 0, 1, 1, 1)

        self.process_label = QLabel(self.central_widget)
        self.process_label.setObjectName(u"process_label")

        self.grid_layout.addWidget(self.process_label, 1, 0, 1, 1)

        self.label4 = QLabel(self.central_widget)
        self.label4.setObjectName(u"label4")

        self.grid_layout.addWidget(self.label4, 10, 0, 1, 1)

        self.slider3 = QSlider(self.central_widget)
        self.slider3.setObjectName(u"slider3")
        self.slider3.setMaximumSize(QSize(200, 16777215))
        self.slider3.setMinimum(110)
        self.slider3.setMaximum(149)
        self.slider3.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.slider3, 9, 1, 1, 1)

        self.material_label = QLabel(self.central_widget)
        self.material_label.setObjectName(u"material_label")

        self.grid_layout.addWidget(self.material_label, 2, 0, 1, 1)

        self.indicator_dropdown = QComboBox(self.central_widget)
        self.indicator_dropdown.setObjectName(u"indicator_dropdown")
        self.indicator_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.indicator_dropdown, 3, 1, 1, 1)

        self.slider4 = QSlider(self.central_widget)
        self.slider4.setObjectName(u"slider4")
        self.slider4.setMaximumSize(QSize(200, 16777215))
        self.slider4.setMinimum(51)
        self.slider4.setMaximum(299)
        self.slider4.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.slider4, 10, 1, 1, 1)

        self.process_dropdown = QComboBox(self.central_widget)
        self.process_dropdown.setObjectName(u"process_dropdown")
        self.process_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.process_dropdown, 1, 1, 1, 1)

        self.direction_dropdown = QComboBox(self.central_widget)
        self.direction_dropdown.setObjectName(u"direction_dropdown")
        self.direction_dropdown.setEnabled(False)
        self.direction_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.direction_dropdown, 4, 1, 1, 1)

        self.load_mesh_button = QPushButton(self.central_widget)
        self.load_mesh_button.setObjectName(u"load_mesh_button")
        self.load_mesh_button.setEnabled(True)
        self.load_mesh_button.setMaximumSize(QSize(400, 16777215))

        self.grid_layout.addWidget(self.load_mesh_button, 5, 0, 1, 2)

        self.slider2 = QSlider(self.central_widget)
        self.slider2.setObjectName(u"slider2")
        self.slider2.setMaximumSize(QSize(200, 16777215))
        self.slider2.setMinimum(10)
        self.slider2.setMaximum(20)
        self.slider2.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.slider2, 8, 1, 1, 1)

        self.label_description = QLabel(self.central_widget)
        self.label_description.setObjectName(u"label_description")
        self.label_description.setMaximumSize(QSize(200, 16777215))
        self.label_description.setWordWrap(True)

        self.grid_layout.addWidget(self.label_description, 6, 1, 1, 1)

        self.direction_label = QLabel(self.central_widget)
        self.direction_label.setObjectName(u"direction_label")
        self.direction_label.setEnabled(False)
        self.direction_label.setMaximumSize(QSize(160, 16777215))

        self.grid_layout.addWidget(self.direction_label, 4, 0, 1, 1)

        self.indicator_label = QLabel(self.central_widget)
        self.indicator_label.setObjectName(u"indicator_label")
        self.indicator_label.setMaximumSize(QSize(16777215, 16777215))

        self.grid_layout.addWidget(self.indicator_label, 3, 0, 1, 1)

        self.label_heading = QLabel(self.central_widget)
        self.label_heading.setObjectName(u"label_heading")
        self.label_heading.setMaximumSize(QSize(150, 200))
        self.label_heading.setPixmap(QPixmap(u"../../../../../../.designer/backup/cardoorpanel_blank_explanation.png"))
        self.label_heading.setScaledContents(True)
        self.label_heading.setAlignment(Qt.AlignCenter)
        self.label_heading.setWordWrap(True)

        self.grid_layout.addWidget(self.label_heading, 6, 0, 1, 1)

        self.material_dropdown = QComboBox(self.central_widget)
        self.material_dropdown.setObjectName(u"material_dropdown")
        self.material_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.material_dropdown, 2, 1, 1, 1)

        self.component_label = QLabel(self.central_widget)
        self.component_label.setObjectName(u"component_label")

        self.grid_layout.addWidget(self.component_label, 0, 0, 1, 1)

        self.label3 = QLabel(self.central_widget)
        self.label3.setObjectName(u"label3")

        self.grid_layout.addWidget(self.label3, 9, 0, 1, 1)

        self.label2 = QLabel(self.central_widget)
        self.label2.setObjectName(u"label2")

        self.grid_layout.addWidget(self.label2, 8, 0, 1, 1)

        self.label1 = QLabel(self.central_widget)
        self.label1.setObjectName(u"label1")
        self.label1.setMinimumSize(QSize(120, 0))

        self.grid_layout.addWidget(self.label1, 7, 0, 1, 1)

        self.slider1 = QSlider(self.central_widget)
        self.slider1.setObjectName(u"slider1")
        self.slider1.setMaximumSize(QSize(200, 16777215))
        self.slider1.setMinimum(520)
        self.slider1.setMaximum(5900)
        self.slider1.setPageStep(10)
        self.slider1.setValue(2500)
        self.slider1.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.slider1, 7, 1, 1, 1)

        self.main_view = GLViewWidget(self.central_widget)
        self.main_view.setObjectName(u"main_view")
        self.main_view.setEnabled(True)
        self.main_view.setMinimumSize(QSize(300, 0))

        self.grid_layout.addWidget(self.main_view, 0, 2, 11, 1)

        self.GraphicsLayoutWidget = GraphicsLayoutWidget(self.central_widget)
        self.GraphicsLayoutWidget.setObjectName(u"GraphicsLayoutWidget")
        self.GraphicsLayoutWidget.setMaximumSize(QSize(120, 16777215))

        self.grid_layout.addWidget(self.GraphicsLayoutWidget, 0, 3, 11, 1)


        self.horizontalLayout.addLayout(self.grid_layout)

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
        self.menuMode.addAction(self.action_newoptimisation)
        self.menuMode.addAction(self.action_optimisation)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_sensitivity)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_developer)
        self.menuExit.addAction(self.actionExit)

        self.retranslateUi(main_window)

        QMetaObject.connectSlotsByName(main_window)
    # setupUi

    def retranslateUi(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("main_window", u"User-Centric Software to Assist Design for Forming", None))
        self.actionImport_New_Mesh.setText(QCoreApplication.translate("main_window", u"Import New Mesh", None))
        self.action_newoptimisation.setText(QCoreApplication.translate("main_window", u"New Optimisation", None))
        self.action_sensitivity.setText(QCoreApplication.translate("main_window", u"Sensitiity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("main_window", u"Developer", None))
        self.actionExit.setText(QCoreApplication.translate("main_window", u"Exit", None))
        self.action_optimisation.setText(QCoreApplication.translate("main_window", u"Optimisation Workspace", None))
        self.process_label.setText(QCoreApplication.translate("main_window", u"Process", None))
        self.label4.setText(QCoreApplication.translate("main_window", u"TextLabel", None))
        self.material_label.setText(QCoreApplication.translate("main_window", u"Material", None))
        self.load_mesh_button.setText(QCoreApplication.translate("main_window", u"Load New Mesh", None))
        self.label_description.setText(QCoreApplication.translate("main_window", u"TextLabel", None))
        self.direction_label.setText(QCoreApplication.translate("main_window", u"Displacement Direction", None))
        self.indicator_label.setText(QCoreApplication.translate("main_window", u"Performance Indicator", None))
        self.label_heading.setText("")
        self.component_label.setText(QCoreApplication.translate("main_window", u"Component Family", None))
        self.label3.setText(QCoreApplication.translate("main_window", u"TextLabel", None))
        self.label2.setText(QCoreApplication.translate("main_window", u"TextLabel", None))
        self.label1.setText(QCoreApplication.translate("main_window", u"TextLabel", None))
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


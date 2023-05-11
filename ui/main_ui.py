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
    QHeaderView, QLabel, QLayout, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QTreeView, QWidget)

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
        self.horizontalLayout = QHBoxLayout(self.central_widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.indicator_dropdown = QComboBox(self.central_widget)
        self.indicator_dropdown.setObjectName(u"indicator_dropdown")

        self.grid_layout.addWidget(self.indicator_dropdown, 3, 1, 1, 1)

        self.material_label = QLabel(self.central_widget)
        self.material_label.setObjectName(u"material_label")

        self.grid_layout.addWidget(self.material_label, 2, 0, 1, 1)

        self.material_dropdown = QComboBox(self.central_widget)
        self.material_dropdown.setObjectName(u"material_dropdown")
        self.material_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.material_dropdown, 2, 1, 1, 1)

        self.direction_dropdown = QComboBox(self.central_widget)
        self.direction_dropdown.setObjectName(u"direction_dropdown")
        self.direction_dropdown.setEnabled(False)

        self.grid_layout.addWidget(self.direction_dropdown, 4, 1, 1, 1)

        self.process_label = QLabel(self.central_widget)
        self.process_label.setObjectName(u"process_label")

        self.grid_layout.addWidget(self.process_label, 1, 0, 1, 1)

        self.direction_label = QLabel(self.central_widget)
        self.direction_label.setObjectName(u"direction_label")
        self.direction_label.setEnabled(False)
        self.direction_label.setMaximumSize(QSize(160, 16777215))

        self.grid_layout.addWidget(self.direction_label, 4, 0, 1, 1)

        self.load_mesh_button = QPushButton(self.central_widget)
        self.load_mesh_button.setObjectName(u"load_mesh_button")
        self.load_mesh_button.setEnabled(True)

        self.grid_layout.addWidget(self.load_mesh_button, 5, 0, 1, 2)

        self.indicator_label = QLabel(self.central_widget)
        self.indicator_label.setObjectName(u"indicator_label")
        self.indicator_label.setMaximumSize(QSize(16777215, 16777215))

        self.grid_layout.addWidget(self.indicator_label, 3, 0, 1, 1)

        self.process_dropdown = QComboBox(self.central_widget)
        self.process_dropdown.setObjectName(u"process_dropdown")
        self.process_dropdown.setMaximumSize(QSize(200, 16777215))

        self.grid_layout.addWidget(self.process_dropdown, 1, 1, 1, 1)

        self.dir_tree = QTreeView(self.central_widget)
        self.dir_tree.setObjectName(u"dir_tree")
        self.dir_tree.setEnabled(True)
        self.dir_tree.setMaximumSize(QSize(16777215, 16777215))

        self.grid_layout.addWidget(self.dir_tree, 6, 0, 1, 2)

        self.component_label = QLabel(self.central_widget)
        self.component_label.setObjectName(u"component_label")

        self.grid_layout.addWidget(self.component_label, 0, 0, 1, 1)

        self.component_dropdown = QComboBox(self.central_widget)
        self.component_dropdown.setObjectName(u"component_dropdown")

        self.grid_layout.addWidget(self.component_dropdown, 0, 1, 1, 1)

        self.main_view = GLViewWidget(self.central_widget)
        self.main_view.setObjectName(u"main_view")
        self.main_view.setEnabled(True)
        self.main_view.setMinimumSize(QSize(300, 0))

        self.grid_layout.addWidget(self.main_view, 0, 2, 7, 1)

        self.GraphicsLayoutWidget = GraphicsLayoutWidget(self.central_widget)
        self.GraphicsLayoutWidget.setObjectName(u"GraphicsLayoutWidget")
        self.GraphicsLayoutWidget.setMaximumSize(QSize(120, 16777215))

        self.grid_layout.addWidget(self.GraphicsLayoutWidget, 0, 3, 7, 1)


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
        self.material_label.setText(QCoreApplication.translate("main_window", u"Material", None))
        self.process_label.setText(QCoreApplication.translate("main_window", u"Process", None))
        self.direction_label.setText(QCoreApplication.translate("main_window", u"Displacement Direction", None))
        self.load_mesh_button.setText(QCoreApplication.translate("main_window", u"Load New Mesh", None))
        self.indicator_label.setText(QCoreApplication.translate("main_window", u"Performance Indicator", None))
        self.component_label.setText(QCoreApplication.translate("main_window", u"Component Family", None))
        self.menuFile.setTitle(QCoreApplication.translate("main_window", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


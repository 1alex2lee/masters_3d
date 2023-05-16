# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'developer.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHeaderView, QLabel,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QTableView, QTreeWidget,
    QTreeWidgetItem, QWidget)

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(777, 532)
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
        self.action_prediction = QAction(main_window)
        self.action_prediction.setObjectName(u"action_prediction")
        self.central_widget = QWidget(main_window)
        self.central_widget.setObjectName(u"central_widget")
        self.central_widget.setEnabled(True)
        self.central_widget.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_2 = QGridLayout(self.central_widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(10, 10, 10, 10)
        self.label = QLabel(self.central_widget)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.delete_button = QPushButton(self.central_widget)
        self.delete_button.setObjectName(u"delete_button")
        self.delete_button.setEnabled(False)

        self.gridLayout_2.addWidget(self.delete_button, 2, 1, 1, 1)

        self.export_button = QPushButton(self.central_widget)
        self.export_button.setObjectName(u"export_button")
        self.export_button.setEnabled(False)

        self.gridLayout_2.addWidget(self.export_button, 2, 2, 1, 1)

        self.import_button = QPushButton(self.central_widget)
        self.import_button.setObjectName(u"import_button")
        self.import_button.setEnabled(False)

        self.gridLayout_2.addWidget(self.import_button, 2, 3, 1, 1)

        self.train_new_button = QPushButton(self.central_widget)
        self.train_new_button.setObjectName(u"train_new_button")

        self.gridLayout_2.addWidget(self.train_new_button, 2, 0, 1, 1)

        self.model_table = QTableView(self.central_widget)
        self.model_table.setObjectName(u"model_table")

        self.gridLayout_2.addWidget(self.model_table, 1, 0, 1, 2)

        self.model_tree = QTreeWidget(self.central_widget)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.model_tree.setHeaderItem(__qtreewidgetitem)
        self.model_tree.setObjectName(u"model_tree")

        self.gridLayout_2.addWidget(self.model_tree, 1, 2, 1, 2)

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
        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menuMode")
        self.menuExit = QMenu(self.menubar)
        self.menuExit.setObjectName(u"menuExit")
        main_window.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuExit.menuAction())
        self.menuMode.addAction(self.action_prediction)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_newoptimisation)
        self.menuMode.addAction(self.action_optimisation)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_sensitivity)
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
        self.action_prediction.setText(QCoreApplication.translate("main_window", u"Prediction", None))
        self.label.setText(QCoreApplication.translate("main_window", u"Current Models", None))
        self.delete_button.setText(QCoreApplication.translate("main_window", u"Delete", None))
        self.export_button.setText(QCoreApplication.translate("main_window", u"Export", None))
        self.import_button.setText(QCoreApplication.translate("main_window", u"Import", None))
        self.train_new_button.setText(QCoreApplication.translate("main_window", u"Train New", None))
        self.menuMode.setTitle(QCoreApplication.translate("main_window", u"Mode", None))
        self.menuExit.setTitle(QCoreApplication.translate("main_window", u"Exit", None))
    # retranslateUi


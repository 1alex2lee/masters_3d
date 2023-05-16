# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'sensitivity.ui'
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
    QLabel, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1031, 621)
        self.action_prediction = QAction(MainWindow)
        self.action_prediction.setObjectName(u"action_prediction")
        self.action_optimisation = QAction(MainWindow)
        self.action_optimisation.setObjectName(u"action_optimisation")
        self.action_optimisation.setEnabled(True)
        self.action_sensitivity = QAction(MainWindow)
        self.action_sensitivity.setObjectName(u"action_sensitivity")
        self.action_developer = QAction(MainWindow)
        self.action_developer.setObjectName(u"action_developer")
        self.action_newoptimisation = QAction(MainWindow)
        self.action_newoptimisation.setObjectName(u"action_newoptimisation")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.grid = QGridLayout()
        self.grid.setObjectName(u"grid")

        self.gridLayout.addLayout(self.grid, 0, 1, 1, 2)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.component_dropdown = QComboBox(self.centralwidget)
        self.component_dropdown.setObjectName(u"component_dropdown")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.component_dropdown)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_2)

        self.var1_dropdown = QComboBox(self.centralwidget)
        self.var1_dropdown.setObjectName(u"var1_dropdown")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.var1_dropdown)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_3)

        self.var2_dropdown = QComboBox(self.centralwidget)
        self.var2_dropdown.setObjectName(u"var2_dropdown")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.var2_dropdown)

        self.load_button = QPushButton(self.centralwidget)
        self.load_button.setObjectName(u"load_button")

        self.formLayout.setWidget(3, QFormLayout.SpanningRole, self.load_button)


        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 16777214, 24))
        self.menubar.setMinimumSize(QSize(16777214, 24))
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(False)
        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menuMode")
        self.menuFile_2 = QMenu(self.menubar)
        self.menuFile_2.setObjectName(u"menuFile_2")
        self.menuExit = QMenu(self.menubar)
        self.menuExit.setObjectName(u"menuExit")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuFile_2.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuExit.menuAction())
        self.menuMode.addAction(self.action_prediction)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_newoptimisation)
        self.menuMode.addAction(self.action_optimisation)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.action_developer)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Sensitivity Analysis", None))
        self.action_prediction.setText(QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.action_optimisation.setText(QCoreApplication.translate("MainWindow", u"Optimisation Workspace", None))
        self.action_sensitivity.setText(QCoreApplication.translate("MainWindow", u"Sensitivity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("MainWindow", u"Developer", None))
        self.action_newoptimisation.setText(QCoreApplication.translate("MainWindow", u"New Optimisation", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Component", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Variable 1", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Variable 2", None))
        self.load_button.setText(QCoreApplication.translate("MainWindow", u"Calculate Sensitivity", None))
        self.menuMode.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.menuFile_2.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuExit.setTitle(QCoreApplication.translate("MainWindow", u"Exit", None))
    # retranslateUi


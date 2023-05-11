# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'optimisation_setup2 copy.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QProgressBar, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(868, 574)
        self.action_prediction = QAction(MainWindow)
        self.action_prediction.setObjectName(u"action_prediction")
        self.action_optimisation = QAction(MainWindow)
        self.action_optimisation.setObjectName(u"action_optimisation")
        self.action_optimisation.setEnabled(False)
        self.action_sensitivity = QAction(MainWindow)
        self.action_sensitivity.setObjectName(u"action_sensitivity")
        self.action_developer = QAction(MainWindow)
        self.action_developer.setObjectName(u"action_developer")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.cancel_button = QPushButton(self.centralwidget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 3, 0, 1, 1)

        self.grid = QGridLayout()
        self.grid.setObjectName(u"grid")

        self.gridLayout.addLayout(self.grid, 2, 0, 1, 2)

        self.next_button = QPushButton(self.centralwidget)
        self.next_button.setObjectName(u"next_button")
        self.next_button.setEnabled(False)

        self.gridLayout.addWidget(self.next_button, 3, 1, 1, 1)

        self.runsno_label = QLabel(self.centralwidget)
        self.runsno_label.setObjectName(u"runsno_label")

        self.gridLayout.addWidget(self.runsno_label, 0, 0, 1, 2)

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)

        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 16777214, 24))
        self.menubar.setMinimumSize(QSize(16777214, 24))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.action_prediction)
        self.menuFile.addAction(self.action_optimisation)
        self.menuFile.addAction(self.action_sensitivity)
        self.menuFile.addAction(self.action_developer)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Optimisation in Progress", None))
        self.action_prediction.setText(QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.action_optimisation.setText(QCoreApplication.translate("MainWindow", u"Optimisation", None))
        self.action_sensitivity.setText(QCoreApplication.translate("MainWindow", u"Sensitivity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("MainWindow", u"Developer", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.next_button.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.runsno_label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
    # retranslateUi


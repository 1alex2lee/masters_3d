# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'trainnew_progress.ui'
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
from PySide6.QtWidgets import (QApplication, QFormLayout, QGridLayout, QLabel,
    QMainWindow, QMenu, QMenuBar, QProgressBar,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(390, 563)
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
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.resultimages_label = QLabel(self.centralwidget)
        self.resultimages_label.setObjectName(u"resultimages_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.resultimages_label)

        self.resultimage_progressbar = QProgressBar(self.centralwidget)
        self.resultimage_progressbar.setObjectName(u"resultimage_progressbar")
        self.resultimage_progressbar.setMaximum(100)
        self.resultimage_progressbar.setValue(0)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.resultimage_progressbar)

        self.script2_label = QLabel(self.centralwidget)
        self.script2_label.setObjectName(u"script2_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.script2_label)

        self.script2_progressbar = QProgressBar(self.centralwidget)
        self.script2_progressbar.setObjectName(u"script2_progressbar")
        self.script2_progressbar.setValue(0)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.script2_progressbar)

        self.dieimages_label = QLabel(self.centralwidget)
        self.dieimages_label.setObjectName(u"dieimages_label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.dieimages_label)

        self.dieimages_progressbar = QProgressBar(self.centralwidget)
        self.dieimages_progressbar.setObjectName(u"dieimages_progressbar")
        self.dieimages_progressbar.setValue(0)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.dieimages_progressbar)

        self.disp_label = QLabel(self.centralwidget)
        self.disp_label.setObjectName(u"disp_label")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.disp_label)

        self.thinning_label = QLabel(self.centralwidget)
        self.thinning_label.setObjectName(u"thinning_label")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.thinning_label)

        self.disp_progressbar = QProgressBar(self.centralwidget)
        self.disp_progressbar.setObjectName(u"disp_progressbar")
        self.disp_progressbar.setValue(0)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.disp_progressbar)

        self.thinning_progressbar = QProgressBar(self.centralwidget)
        self.thinning_progressbar.setObjectName(u"thinning_progressbar")
        self.thinning_progressbar.setValue(0)

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.thinning_progressbar)


        self.gridLayout.addLayout(self.formLayout, 1, 0, 1, 2)

        self.cancel_button = QPushButton(self.centralwidget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 5, 0, 1, 1)

        self.grid = QGridLayout()
        self.grid.setObjectName(u"grid")

        self.gridLayout.addLayout(self.grid, 4, 0, 1, 2)

        self.done_button = QPushButton(self.centralwidget)
        self.done_button.setObjectName(u"done_button")
        self.done_button.setEnabled(False)

        self.gridLayout.addWidget(self.done_button, 5, 1, 1, 1)

        self.status_label = QLabel(self.centralwidget)
        self.status_label.setObjectName(u"status_label")
        self.status_label.setMaximumSize(QSize(16777215, 24))
        self.status_label.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.status_label, 0, 0, 1, 2)

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
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Model Training in Progress", None))
        self.action_prediction.setText(QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.action_optimisation.setText(QCoreApplication.translate("MainWindow", u"Optimisation", None))
        self.action_sensitivity.setText(QCoreApplication.translate("MainWindow", u"Sensitivity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("MainWindow", u"Developer", None))
        self.resultimages_label.setText(QCoreApplication.translate("MainWindow", u"Load result files", None))
        self.script2_label.setText(QCoreApplication.translate("MainWindow", u"Obtain target images", None))
        self.dieimages_label.setText(QCoreApplication.translate("MainWindow", u"Obtain input die images", None))
        self.disp_label.setText(QCoreApplication.translate("MainWindow", u"Displacement Model", None))
        self.thinning_label.setText(QCoreApplication.translate("MainWindow", u"Thinning Model", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.done_button.setText(QCoreApplication.translate("MainWindow", u"Done", None))
        self.status_label.setText(QCoreApplication.translate("MainWindow", u"Preparing data...", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'optimisation_setup copy.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGridLayout, QLabel, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSpinBox,
    QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(373, 401)
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
        self.next_button = QPushButton(self.centralwidget)
        self.next_button.setObjectName(u"next_button")
        self.next_button.setEnabled(False)

        self.gridLayout.addWidget(self.next_button, 10, 3, 1, 1)

        self.searchmethod_label = QLabel(self.centralwidget)
        self.searchmethod_label.setObjectName(u"searchmethod_label")

        self.gridLayout.addWidget(self.searchmethod_label, 5, 0, 1, 1)

        self.goal_label = QLabel(self.centralwidget)
        self.goal_label.setObjectName(u"goal_label")

        self.gridLayout.addWidget(self.goal_label, 3, 0, 1, 1)

        self.maximise_checkbox = QCheckBox(self.centralwidget)
        self.maximise_checkbox.setObjectName(u"maximise_checkbox")

        self.gridLayout.addWidget(self.maximise_checkbox, 4, 0, 1, 1)

        self.model_dropdown = QComboBox(self.centralwidget)
        self.model_dropdown.setObjectName(u"model_dropdown")

        self.gridLayout.addWidget(self.model_dropdown, 2, 1, 1, 3)

        self.setto_value = QDoubleSpinBox(self.centralwidget)
        self.setto_value.setObjectName(u"setto_value")

        self.gridLayout.addWidget(self.setto_value, 4, 3, 1, 1)

        self.runsno_value = QSpinBox(self.centralwidget)
        self.runsno_value.setObjectName(u"runsno_value")
        self.runsno_value.setMaximum(100)
        self.runsno_value.setValue(20)

        self.gridLayout.addWidget(self.runsno_value, 6, 1, 1, 3)

        self.process_label = QLabel(self.centralwidget)
        self.process_label.setObjectName(u"process_label")

        self.gridLayout.addWidget(self.process_label, 0, 0, 1, 1)

        self.model_label = QLabel(self.centralwidget)
        self.model_label.setObjectName(u"model_label")

        self.gridLayout.addWidget(self.model_label, 2, 0, 1, 1)

        self.minimise_checkbox = QCheckBox(self.centralwidget)
        self.minimise_checkbox.setObjectName(u"minimise_checkbox")

        self.gridLayout.addWidget(self.minimise_checkbox, 4, 1, 1, 1)

        self.progress_label = QLabel(self.centralwidget)
        self.progress_label.setObjectName(u"progress_label")

        self.gridLayout.addWidget(self.progress_label, 8, 0, 1, 4, Qt.AlignHCenter)

        self.setto_checkbox = QCheckBox(self.centralwidget)
        self.setto_checkbox.setObjectName(u"setto_checkbox")

        self.gridLayout.addWidget(self.setto_checkbox, 4, 2, 1, 1, Qt.AlignRight)

        self.material_label = QLabel(self.centralwidget)
        self.material_label.setObjectName(u"material_label")

        self.gridLayout.addWidget(self.material_label, 1, 0, 1, 1)

        self.searchmethod_dropdown = QComboBox(self.centralwidget)
        self.searchmethod_dropdown.setObjectName(u"searchmethod_dropdown")

        self.gridLayout.addWidget(self.searchmethod_dropdown, 5, 1, 1, 3)

        self.geometry_label = QLabel(self.centralwidget)
        self.geometry_label.setObjectName(u"geometry_label")

        self.gridLayout.addWidget(self.geometry_label, 7, 0, 1, 1)

        self.goal_dropdown = QComboBox(self.centralwidget)
        self.goal_dropdown.setObjectName(u"goal_dropdown")

        self.gridLayout.addWidget(self.goal_dropdown, 3, 1, 1, 3)

        self.runsno_label = QLabel(self.centralwidget)
        self.runsno_label.setObjectName(u"runsno_label")

        self.gridLayout.addWidget(self.runsno_label, 6, 0, 1, 1)

        self.process_dropdown = QComboBox(self.centralwidget)
        self.process_dropdown.setObjectName(u"process_dropdown")

        self.gridLayout.addWidget(self.process_dropdown, 0, 1, 1, 3)

        self.cancel_button = QPushButton(self.centralwidget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 10, 0, 1, 1)

        self.load_mesh_button = QPushButton(self.centralwidget)
        self.load_mesh_button.setObjectName(u"load_mesh_button")

        self.gridLayout.addWidget(self.load_mesh_button, 7, 1, 1, 3)

        self.material_dropdown = QComboBox(self.centralwidget)
        self.material_dropdown.setObjectName(u"material_dropdown")

        self.gridLayout.addWidget(self.material_dropdown, 1, 1, 1, 3)

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
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.action_prediction.setText(QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.action_optimisation.setText(QCoreApplication.translate("MainWindow", u"Optimisation", None))
        self.action_sensitivity.setText(QCoreApplication.translate("MainWindow", u"Sensitivity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("MainWindow", u"Developer", None))
        self.next_button.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.searchmethod_label.setText(QCoreApplication.translate("MainWindow", u"Search Method", None))
        self.goal_label.setText(QCoreApplication.translate("MainWindow", u"Goal", None))
        self.maximise_checkbox.setText(QCoreApplication.translate("MainWindow", u"Maximise", None))
        self.process_label.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.model_label.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.minimise_checkbox.setText(QCoreApplication.translate("MainWindow", u"Minimise", None))
        self.progress_label.setText(QCoreApplication.translate("MainWindow", u"No geometry loaded", None))
        self.setto_checkbox.setText(QCoreApplication.translate("MainWindow", u"Set to", None))
        self.material_label.setText(QCoreApplication.translate("MainWindow", u"Material", None))
        self.geometry_label.setText(QCoreApplication.translate("MainWindow", u"Starting Geometry", None))
        self.runsno_label.setText(QCoreApplication.translate("MainWindow", u"Number of Runs", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.load_mesh_button.setText(QCoreApplication.translate("MainWindow", u"Load mesh", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
    # retranslateUi


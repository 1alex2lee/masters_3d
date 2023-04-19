# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'optimisation_setup.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGridLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QSpinBox, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(517, 465)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.process_label = QLabel(self.centralwidget)
        self.process_label.setObjectName(u"process_label")

        self.gridLayout.addWidget(self.process_label, 0, 2, 1, 1)

        self.minimise_checkbox = QCheckBox(self.centralwidget)
        self.minimise_checkbox.setObjectName(u"minimise_checkbox")

        self.gridLayout.addWidget(self.minimise_checkbox, 4, 3, 1, 1)

        self.variables_label = QLabel(self.centralwidget)
        self.variables_label.setObjectName(u"variables_label")
        self.variables_label.setMaximumSize(QSize(16777215, 12))

        self.gridLayout.addWidget(self.variables_label, 0, 0, 1, 1)

        self.material_label = QLabel(self.centralwidget)
        self.material_label.setObjectName(u"material_label")

        self.gridLayout.addWidget(self.material_label, 1, 2, 1, 1)

        self.model_label = QLabel(self.centralwidget)
        self.model_label.setObjectName(u"model_label")

        self.gridLayout.addWidget(self.model_label, 2, 2, 1, 1)

        self.maximise_checkbox = QCheckBox(self.centralwidget)
        self.maximise_checkbox.setObjectName(u"maximise_checkbox")

        self.gridLayout.addWidget(self.maximise_checkbox, 4, 2, 1, 1)

        self.runsno_label = QLabel(self.centralwidget)
        self.runsno_label.setObjectName(u"runsno_label")

        self.gridLayout.addWidget(self.runsno_label, 6, 2, 1, 1)

        self.searchmethod_label = QLabel(self.centralwidget)
        self.searchmethod_label.setObjectName(u"searchmethod_label")

        self.gridLayout.addWidget(self.searchmethod_label, 5, 2, 1, 1)

        self.goal_label = QLabel(self.centralwidget)
        self.goal_label.setObjectName(u"goal_label")

        self.gridLayout.addWidget(self.goal_label, 3, 2, 1, 1)

        self.next_button = QPushButton(self.centralwidget)
        self.next_button.setObjectName(u"next_button")

        self.gridLayout.addWidget(self.next_button, 7, 4, 1, 2)

        self.setto_checkbox = QCheckBox(self.centralwidget)
        self.setto_checkbox.setObjectName(u"setto_checkbox")

        self.gridLayout.addWidget(self.setto_checkbox, 4, 4, 1, 1)

        self.setto_value = QDoubleSpinBox(self.centralwidget)
        self.setto_value.setObjectName(u"setto_value")

        self.gridLayout.addWidget(self.setto_value, 4, 5, 1, 1)

        self.cancel_button = QPushButton(self.centralwidget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 7, 0, 1, 1)

        self.process_dropdown = QComboBox(self.centralwidget)
        self.process_dropdown.setObjectName(u"process_dropdown")

        self.gridLayout.addWidget(self.process_dropdown, 0, 3, 1, 3)

        self.material_dropdown = QComboBox(self.centralwidget)
        self.material_dropdown.setObjectName(u"material_dropdown")

        self.gridLayout.addWidget(self.material_dropdown, 1, 3, 1, 3)

        self.model_dropdown = QComboBox(self.centralwidget)
        self.model_dropdown.setObjectName(u"model_dropdown")

        self.gridLayout.addWidget(self.model_dropdown, 2, 3, 1, 3)

        self.goal_dropdown = QComboBox(self.centralwidget)
        self.goal_dropdown.setObjectName(u"goal_dropdown")

        self.gridLayout.addWidget(self.goal_dropdown, 3, 3, 1, 3)

        self.searchmethod_dropdown = QComboBox(self.centralwidget)
        self.searchmethod_dropdown.setObjectName(u"searchmethod_dropdown")

        self.gridLayout.addWidget(self.searchmethod_dropdown, 5, 3, 1, 3)

        self.runsno_value = QSpinBox(self.centralwidget)
        self.runsno_value.setObjectName(u"runsno_value")
        self.runsno_value.setMaximum(100)
        self.runsno_value.setValue(20)

        self.gridLayout.addWidget(self.runsno_value, 6, 3, 1, 3)

        self.variables_listwidget = QListWidget(self.centralwidget)
        self.variables_listwidget.setObjectName(u"variables_listwidget")

        self.gridLayout.addWidget(self.variables_listwidget, 1, 0, 6, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 16777214, 24))
        self.menubar.setMinimumSize(QSize(16777214, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.process_label.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.minimise_checkbox.setText(QCoreApplication.translate("MainWindow", u"Minimise", None))
        self.variables_label.setText(QCoreApplication.translate("MainWindow", u"Select Variables", None))
        self.material_label.setText(QCoreApplication.translate("MainWindow", u"Material", None))
        self.model_label.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.maximise_checkbox.setText(QCoreApplication.translate("MainWindow", u"Maximise", None))
        self.runsno_label.setText(QCoreApplication.translate("MainWindow", u"Number of Runs", None))
        self.searchmethod_label.setText(QCoreApplication.translate("MainWindow", u"Search Method", None))
        self.goal_label.setText(QCoreApplication.translate("MainWindow", u"Goal", None))
        self.next_button.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.setto_checkbox.setText(QCoreApplication.translate("MainWindow", u"Set to", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
    # retranslateUi


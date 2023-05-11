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
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QLabel, QLayout,
    QMainWindow, QMenu, QMenuBar, QProgressBar,
    QPushButton, QSizePolicy, QSpinBox, QStatusBar,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(367, 524)
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

        self.gridLayout.addWidget(self.next_button, 14, 3, 1, 1)

        self.cancel_button = QPushButton(self.centralwidget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.gridLayout.addWidget(self.cancel_button, 14, 0, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.formLayout.setLabelAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.formLayout.setFormAlignment(Qt.AlignRight|Qt.AlignTop|Qt.AlignTrailing)
        self.component_label = QLabel(self.centralwidget)
        self.component_label.setObjectName(u"component_label")
        self.component_label.setMinimumSize(QSize(0, 0))

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.component_label)

        self.component_dropdown = QComboBox(self.centralwidget)
        self.component_dropdown.setObjectName(u"component_dropdown")
        self.component_dropdown.setMinimumSize(QSize(0, 0))
        self.component_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.component_dropdown)

        self.process_label = QLabel(self.centralwidget)
        self.process_label.setObjectName(u"process_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.process_label)

        self.process_dropdown = QComboBox(self.centralwidget)
        self.process_dropdown.setObjectName(u"process_dropdown")
        self.process_dropdown.setEnabled(False)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.process_dropdown)

        self.material_label = QLabel(self.centralwidget)
        self.material_label.setObjectName(u"material_label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.material_label)

        self.material_dropdown = QComboBox(self.centralwidget)
        self.material_dropdown.setObjectName(u"material_dropdown")
        self.material_dropdown.setEnabled(False)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.material_dropdown)

        self.indicator_label = QLabel(self.centralwidget)
        self.indicator_label.setObjectName(u"indicator_label")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.indicator_label)

        self.indicator_dropdown = QComboBox(self.centralwidget)
        self.indicator_dropdown.setObjectName(u"indicator_dropdown")
        self.indicator_dropdown.setEnabled(False)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.indicator_dropdown)

        self.objfunc_label = QLabel(self.centralwidget)
        self.objfunc_label.setObjectName(u"objfunc_label")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.objfunc_label)

        self.objfunc_dropdown = QComboBox(self.centralwidget)
        self.objfunc_dropdown.setObjectName(u"objfunc_dropdown")
        self.objfunc_dropdown.setEnabled(False)

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.objfunc_dropdown)

        self.maximise_checkbox = QCheckBox(self.centralwidget)
        self.maximise_checkbox.setObjectName(u"maximise_checkbox")
        self.maximise_checkbox.setEnabled(False)

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.maximise_checkbox)

        self.minimise_checkbox = QCheckBox(self.centralwidget)
        self.minimise_checkbox.setObjectName(u"minimise_checkbox")
        self.minimise_checkbox.setEnabled(False)

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.minimise_checkbox)

        self.setto_checkbox = QCheckBox(self.centralwidget)
        self.setto_checkbox.setObjectName(u"setto_checkbox")
        self.setto_checkbox.setEnabled(False)

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.setto_checkbox)

        self.setto_value = QDoubleSpinBox(self.centralwidget)
        self.setto_value.setObjectName(u"setto_value")
        self.setto_value.setEnabled(False)

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.setto_value)

        self.optimiser_label = QLabel(self.centralwidget)
        self.optimiser_label.setObjectName(u"optimiser_label")
        self.optimiser_label.setEnabled(True)

        self.formLayout.setWidget(8, QFormLayout.LabelRole, self.optimiser_label)

        self.optimiser_dropdown = QComboBox(self.centralwidget)
        self.optimiser_dropdown.setObjectName(u"optimiser_dropdown")
        self.optimiser_dropdown.setEnabled(False)

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.optimiser_dropdown)

        self.runsno_label = QLabel(self.centralwidget)
        self.runsno_label.setObjectName(u"runsno_label")

        self.formLayout.setWidget(9, QFormLayout.LabelRole, self.runsno_label)

        self.runsno_value = QSpinBox(self.centralwidget)
        self.runsno_value.setObjectName(u"runsno_value")
        self.runsno_value.setMinimumSize(QSize(0, 0))
        self.runsno_value.setMaximum(1000)
        self.runsno_value.setValue(10)

        self.formLayout.setWidget(9, QFormLayout.FieldRole, self.runsno_value)

        self.geometry_label = QLabel(self.centralwidget)
        self.geometry_label.setObjectName(u"geometry_label")

        self.formLayout.setWidget(10, QFormLayout.LabelRole, self.geometry_label)

        self.load_mesh_button = QPushButton(self.centralwidget)
        self.load_mesh_button.setObjectName(u"load_mesh_button")

        self.formLayout.setWidget(10, QFormLayout.FieldRole, self.load_mesh_button)

        self.progress_label = QLabel(self.centralwidget)
        self.progress_label.setObjectName(u"progress_label")
        self.progress_label.setAlignment(Qt.AlignCenter)

        self.formLayout.setWidget(11, QFormLayout.SpanningRole, self.progress_label)

        self.progress_bar = QProgressBar(self.centralwidget)
        self.progress_bar.setObjectName(u"progress_bar")
        self.progress_bar.setValue(0)

        self.formLayout.setWidget(12, QFormLayout.SpanningRole, self.progress_bar)

        self.edit_objfunc_button = QPushButton(self.centralwidget)
        self.edit_objfunc_button.setObjectName(u"edit_objfunc_button")
        self.edit_objfunc_button.setEnabled(False)

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.edit_objfunc_button)


        self.gridLayout.addLayout(self.formLayout, 12, 0, 1, 4)

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
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Optimisation Mode", None))
        self.action_prediction.setText(QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.action_optimisation.setText(QCoreApplication.translate("MainWindow", u"Optimisation", None))
        self.action_sensitivity.setText(QCoreApplication.translate("MainWindow", u"Sensitivity Analysis", None))
        self.action_developer.setText(QCoreApplication.translate("MainWindow", u"Developer", None))
        self.next_button.setText(QCoreApplication.translate("MainWindow", u"Next", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.component_label.setText(QCoreApplication.translate("MainWindow", u"Component Family", None))
        self.process_label.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.material_label.setText(QCoreApplication.translate("MainWindow", u"Material", None))
        self.indicator_label.setText(QCoreApplication.translate("MainWindow", u"Performance Indicator", None))
        self.objfunc_label.setText(QCoreApplication.translate("MainWindow", u"Objective Function", None))
        self.maximise_checkbox.setText(QCoreApplication.translate("MainWindow", u"Maximise", None))
        self.minimise_checkbox.setText(QCoreApplication.translate("MainWindow", u"Minimise", None))
        self.setto_checkbox.setText(QCoreApplication.translate("MainWindow", u"Set to", None))
        self.optimiser_label.setText(QCoreApplication.translate("MainWindow", u"Optimiser", None))
        self.runsno_label.setText(QCoreApplication.translate("MainWindow", u"Number of Iterations", None))
        self.geometry_label.setText(QCoreApplication.translate("MainWindow", u"Starting Geometry", None))
        self.load_mesh_button.setText(QCoreApplication.translate("MainWindow", u"Load mesh", None))
        self.progress_label.setText(QCoreApplication.translate("MainWindow", u"No geometry loaded", None))
        self.edit_objfunc_button.setText(QCoreApplication.translate("MainWindow", u"Edit Objective Functions", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
    # retranslateUi


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'YM.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(584, 309)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setStyleSheet("color: grey;")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.cropHorizontalSlider = QtWidgets.QSlider(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.cropHorizontalSlider.setFont(font)
        self.cropHorizontalSlider.setMaximum(10)
        self.cropHorizontalSlider.setProperty("value", 0)
        self.cropHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.cropHorizontalSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.cropHorizontalSlider.setObjectName("cropHorizontalSlider")
        self.gridLayout.addWidget(self.cropHorizontalSlider, 0, 1, 1, 2)
        self.cropLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.cropLabel.setFont(font)
        self.cropLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.cropLabel.setObjectName("cropLabel")
        self.gridLayout.addWidget(self.cropLabel, 0, 3, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: grey;")
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: grey;")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.translationHorizontalSlider = QtWidgets.QSlider(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.translationHorizontalSlider.setFont(font)
        self.translationHorizontalSlider.setMaximum(10)
        self.translationHorizontalSlider.setProperty("value", 0)
        self.translationHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.translationHorizontalSlider.setObjectName("translationHorizontalSlider")
        self.gridLayout.addWidget(self.translationHorizontalSlider, 1, 1, 1, 2)
        self.translationLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.translationLabel.setFont(font)
        self.translationLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.translationLabel.setObjectName("translationLabel")
        self.gridLayout.addWidget(self.translationLabel, 1, 3, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("color: grey;")
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 1, 4, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: grey;")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.lightHorizontalSlider = QtWidgets.QSlider(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.lightHorizontalSlider.setFont(font)
        self.lightHorizontalSlider.setMaximum(10)
        self.lightHorizontalSlider.setProperty("value", 0)
        self.lightHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.lightHorizontalSlider.setObjectName("lightHorizontalSlider")
        self.gridLayout.addWidget(self.lightHorizontalSlider, 2, 1, 1, 2)
        self.lightLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.lightLabel.setFont(font)
        self.lightLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.lightLabel.setObjectName("lightLabel")
        self.gridLayout.addWidget(self.lightLabel, 2, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("color: grey;")
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 4, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("color: grey;")
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 2, 5, 1, 1)
        self.lightDOWNDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.lightDOWNDoubleSpinBox.setFont(font)
        self.lightDOWNDoubleSpinBox.setStyleSheet("border: 1px solid grey;")
        self.lightDOWNDoubleSpinBox.setDecimals(1)
        self.lightDOWNDoubleSpinBox.setMinimum(0.5)
        self.lightDOWNDoubleSpinBox.setMaximum(1.0)
        self.lightDOWNDoubleSpinBox.setSingleStep(0.1)
        self.lightDOWNDoubleSpinBox.setObjectName("lightDOWNDoubleSpinBox")
        self.gridLayout.addWidget(self.lightDOWNDoubleSpinBox, 2, 6, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("color: grey;")
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 2, 7, 1, 1)
        self.lightUPDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.lightUPDoubleSpinBox.setFont(font)
        self.lightUPDoubleSpinBox.setStyleSheet("border: 1px solid grey;")
        self.lightUPDoubleSpinBox.setDecimals(1)
        self.lightUPDoubleSpinBox.setMinimum(1.0)
        self.lightUPDoubleSpinBox.setMaximum(1.5)
        self.lightUPDoubleSpinBox.setSingleStep(0.1)
        self.lightUPDoubleSpinBox.setProperty("value", 1.5)
        self.lightUPDoubleSpinBox.setObjectName("lightUPDoubleSpinBox")
        self.gridLayout.addWidget(self.lightUPDoubleSpinBox, 2, 8, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: grey;")
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.noiseHorizontalSlider = QtWidgets.QSlider(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.noiseHorizontalSlider.setFont(font)
        self.noiseHorizontalSlider.setMaximum(10)
        self.noiseHorizontalSlider.setProperty("value", 0)
        self.noiseHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.noiseHorizontalSlider.setObjectName("noiseHorizontalSlider")
        self.gridLayout.addWidget(self.noiseHorizontalSlider, 3, 1, 1, 2)
        self.noiseLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.noiseLabel.setFont(font)
        self.noiseLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.noiseLabel.setObjectName("noiseLabel")
        self.gridLayout.addWidget(self.noiseLabel, 3, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("color: grey;")
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 3, 4, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: grey;")
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.rotateHorizontalSlider = QtWidgets.QSlider(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.rotateHorizontalSlider.setFont(font)
        self.rotateHorizontalSlider.setMaximum(10)
        self.rotateHorizontalSlider.setProperty("value", 0)
        self.rotateHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rotateHorizontalSlider.setObjectName("rotateHorizontalSlider")
        self.gridLayout.addWidget(self.rotateHorizontalSlider, 4, 1, 1, 2)
        self.rotateLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.rotateLabel.setFont(font)
        self.rotateLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.rotateLabel.setObjectName("rotateLabel")
        self.gridLayout.addWidget(self.rotateLabel, 4, 3, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("color: grey;")
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 4, 4, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("color: grey;")
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 4, 5, 1, 1)
        self.rotateDOWNdoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.rotateDOWNdoubleSpinBox.setFont(font)
        self.rotateDOWNdoubleSpinBox.setStyleSheet("border: 1px solid grey;")
        self.rotateDOWNdoubleSpinBox.setDecimals(0)
        self.rotateDOWNdoubleSpinBox.setMinimum(-360.0)
        self.rotateDOWNdoubleSpinBox.setMaximum(0.0)
        self.rotateDOWNdoubleSpinBox.setSingleStep(1.0)
        self.rotateDOWNdoubleSpinBox.setProperty("value", -45.0)
        self.rotateDOWNdoubleSpinBox.setObjectName("rotateDOWNdoubleSpinBox")
        self.gridLayout.addWidget(self.rotateDOWNdoubleSpinBox, 4, 6, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("color: grey;")
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 4, 7, 1, 1)
        self.rotateUPDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.rotateUPDoubleSpinBox.setFont(font)
        self.rotateUPDoubleSpinBox.setStyleSheet("border: 1px solid grey;")
        self.rotateUPDoubleSpinBox.setDecimals(0)
        self.rotateUPDoubleSpinBox.setMinimum(0.0)
        self.rotateUPDoubleSpinBox.setMaximum(360.0)
        self.rotateUPDoubleSpinBox.setSingleStep(1.0)
        self.rotateUPDoubleSpinBox.setProperty("value", 45.0)
        self.rotateUPDoubleSpinBox.setObjectName("rotateUPDoubleSpinBox")
        self.gridLayout.addWidget(self.rotateUPDoubleSpinBox, 4, 8, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("color: grey;")
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color: grey;")
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 5, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("color: grey;")
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 5, 4, 1, 2)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("color: grey;")
        self.label_18.setObjectName("label_18")
        self.gridLayout.addWidget(self.label_18, 6, 0, 1, 2)
        self.originalFolderLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setUnderline(True)
        self.originalFolderLabel.setFont(font)
        self.originalFolderLabel.setStyleSheet("color: rgb(85, 170, 255);")
        self.originalFolderLabel.setText("")
        self.originalFolderLabel.setObjectName("originalFolderLabel")
        self.gridLayout.addWidget(self.originalFolderLabel, 6, 2, 1, 5)
        self.oldToolButton = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.oldToolButton.setFont(font)
        self.oldToolButton.setObjectName("oldToolButton")
        self.gridLayout.addWidget(self.oldToolButton, 6, 9, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_20.setFont(font)
        self.label_20.setStyleSheet("color: grey;")
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 7, 0, 1, 2)
        self.changedFolderLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setUnderline(True)
        self.changedFolderLabel.setFont(font)
        self.changedFolderLabel.setStyleSheet("color: rgb(85, 170, 255);")
        self.changedFolderLabel.setText("")
        self.changedFolderLabel.setObjectName("changedFolderLabel")
        self.gridLayout.addWidget(self.changedFolderLabel, 7, 2, 1, 5)
        self.newToolButton = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.newToolButton.setFont(font)
        self.newToolButton.setObjectName("newToolButton")
        self.gridLayout.addWidget(self.newToolButton, 7, 9, 1, 1)
        self.resetPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.resetPushButton.setMinimumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.resetPushButton.setFont(font)
        self.resetPushButton.setStyleSheet("background-color: white;\n"
"color: grey;\n"
"border-radius:5px;\n"
"border:1px solid grey;")
        self.resetPushButton.setObjectName("resetPushButton")
        self.gridLayout.addWidget(self.resetPushButton, 8, 3, 1, 3)
        self.exitPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.exitPushButton.setMinimumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.exitPushButton.setFont(font)
        self.exitPushButton.setStyleSheet("background-color: white;\n"
"color: grey;\n"
"border-radius:5px;\n"
"border:1px solid grey;")
        self.exitPushButton.setObjectName("exitPushButton")
        self.gridLayout.addWidget(self.exitPushButton, 8, 7, 1, 3)
        self.startPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.startPushButton.setMinimumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.startPushButton.setFont(font)
        self.startPushButton.setStyleSheet("background-color: #A52A2A;\n"
"border-radius: 5px;\n"
"color: white;")
        self.startPushButton.setObjectName("startPushButton")
        self.gridLayout.addWidget(self.startPushButton, 8, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "裁切变换"))
        self.cropLabel.setText(_translate("MainWindow", "0"))
        self.label_8.setText(_translate("MainWindow", "次"))
        self.label_2.setText(_translate("MainWindow", "平移变换"))
        self.translationLabel.setText(_translate("MainWindow", "0"))
        self.label_9.setText(_translate("MainWindow", "次"))
        self.label_3.setText(_translate("MainWindow", "改变亮度"))
        self.lightLabel.setText(_translate("MainWindow", "0"))
        self.label_10.setText(_translate("MainWindow", "次"))
        self.label_11.setText(_translate("MainWindow", "亮度下界"))
        self.label_12.setText(_translate("MainWindow", "亮度上界"))
        self.label_4.setText(_translate("MainWindow", "高斯噪声"))
        self.noiseLabel.setText(_translate("MainWindow", "0"))
        self.label_13.setText(_translate("MainWindow", "次"))
        self.label_5.setText(_translate("MainWindow", "角度旋转"))
        self.rotateLabel.setText(_translate("MainWindow", "0"))
        self.label_14.setText(_translate("MainWindow", "次"))
        self.label_16.setText(_translate("MainWindow", "角度下界"))
        self.label_17.setText(_translate("MainWindow", "角度下界"))
        self.label_6.setText(_translate("MainWindow", "镜像"))
        self.label_7.setText(_translate("MainWindow", "垂直翻转1次"))
        self.label_15.setText(_translate("MainWindow", "水平垂直翻转1次"))
        self.label_18.setText(_translate("MainWindow", "原始图片标注地址"))
        self.oldToolButton.setText(_translate("MainWindow", "..."))
        self.label_20.setText(_translate("MainWindow", "增广图片存放地址"))
        self.newToolButton.setText(_translate("MainWindow", "..."))
        self.resetPushButton.setText(_translate("MainWindow", "重置"))
        self.exitPushButton.setText(_translate("MainWindow", "退出程序"))
        self.startPushButton.setText(_translate("MainWindow", "开始增广"))
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        """
        GUI description
        :return:
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(488, 228)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.predict_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_button.setGeometry(QtCore.QRect(150, 130, 201, 41))
        self.predict_button.setObjectName("predict_button")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(330, 20, 141, 71))
        self.groupBox.setObjectName("groupBox")
        self.model_box = QtWidgets.QComboBox(self.groupBox)
        self.model_box.setGeometry(QtCore.QRect(10, 30, 101, 21))
        self.model_box.setObjectName("model_box")
        self.model_box.addItem("")
        self.model_box.addItem("")
        self.model_box.addItem("")
        self.model_box.addItem("")
        self.model_box.addItem("")
        self.model_box.addItem("")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 20, 301, 71))
        self.groupBox_2.setObjectName("groupBox_2")
        self.path_img_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.path_img_lineedit.setGeometry(QtCore.QRect(10, 30, 241, 21))
        self.path_img_lineedit.setText("")
        self.path_img_lineedit.setObjectName("path_img_lineedit")
        self.file_path_button = QtWidgets.QToolButton(self.groupBox_2)
        self.file_path_button.setGeometry(QtCore.QRect(260, 30, 25, 19))
        self.file_path_button.setObjectName("file_path_button")
        self.path_model_lable = QtWidgets.QLabel(self.centralwidget)
        self.path_model_lable.setGeometry(QtCore.QRect(40, 90, 431, 41))
        self.path_model_lable.setObjectName("path_model_lable")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 488, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """
         GUI description
        :return:
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ВКР"))
        self.predict_button.setText(_translate("MainWindow", "Предсказать"))
        self.groupBox.setTitle(_translate("MainWindow", "Модель"))
        self.model_box.setItemText(0, _translate("MainWindow", "LeNet"))
        self.model_box.setItemText(1, _translate("MainWindow", "Alexnet"))
        self.model_box.setItemText(2, _translate("MainWindow", "VGG16"))
        self.model_box.setItemText(3, _translate("MainWindow", "VGG19"))
        self.model_box.setItemText(4, _translate("MainWindow", "Xception"))
        self.model_box.setItemText(5, _translate("MainWindow", "Выбрать из файла"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Путь к файлам"))
        self.file_path_button.setText(_translate("MainWindow", "..."))
        self.path_model_lable.setText(_translate("MainWindow", "TextLabel"))

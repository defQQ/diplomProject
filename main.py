from PyQt5 import QtWidgets
from window import Ui_MainWindow
from PyQt5.QtWidgets import QMessageBox
import sys
import config, predict


class WindowOperator(QtWidgets.QMainWindow):
    def __init__(self):
        super(WindowOperator, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.path_img_lineedit.setReadOnly(True)
        self.ui.predict_button.clicked.connect(self.predict_images)
        self.ui.file_path_button.clicked.connect(self.get_file_path)
        self.ui.model_box.currentIndexChanged.connect(self.get_model_path)
        self.ui.path_model_lable.setText('')
        self.ui.predict_button.setEnabled(False)

        self.images_path = ''
        self.model_path = config.lenet

    def get_file_path(self):
        """
        Get path to image or images
        :return:
        """
        self.images_path = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select imgs',
                                                                  'С:',
                                                                  "Images(*.png *.jpg *.jpeg)")[0]
        if not self.images_path:
            QMessageBox.critical(self, "Error path to image or images", "Please select path.")
            self.ui.predict_button.setEnabled(False)
            self.ui.path_model_lable.setText('')
        else:
            if self.model_path:
                self.ui.predict_button.setEnabled(True)

        temp_str = ''.join(self.images_path)
        self.ui.path_img_lineedit.setText(temp_str)

    def get_model_path(self):
        """
        Get model path, input path to label
        :return:
        """
        if self.ui.model_box.currentText() == 'LeNet':
            self.model_path = config.lenet
            self.ui.predict_button.setEnabled(True)
            self.ui.path_model_lable.setText('')
        if self.ui.model_box.currentText() == 'AlexNet':
            self.model_path = config.alexnet
            self.ui.predict_button.setEnabled(True)
            self.ui.path_model_lable.setText('')
        if self.ui.model_box.currentText() == 'VGG16':
            self.model_path = config.vgg16
            self.ui.predict_button.setEnabled(True)
            self.ui.path_model_lable.setText('')
        if self.ui.model_box.currentText() == 'VGG19':
            self.model_path = config.vgg19
            self.ui.predict_button.setEnabled(True)
            self.ui.path_model_lable.setText('')
        if self.ui.model_box.currentText() == 'Xception':
            self.model_path = config.xception
            self.ui.predict_button.setEnabled(True)
            self.ui.path_model_lable.setText('')

        if self.ui.model_box.currentIndex() == 5:
            self.model_path = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select model',
                                                                     'D:/MAGA_DIP/DiplomProject',
                                                                     "Model (*.hdf5)")[0]
            path_model_label_text = 'Select model: ' + ''.join(self.model_path)
            self.ui.path_model_lable.setText(path_model_label_text)
        if not self.model_path:
            QMessageBox.critical(self, "Error path to model", "Please select path to model.")
            self.ui.predict_button.setEnabled(False)
            self.ui.path_model_lable.setText('')
        else:
            if not self.images_path:
                self.ui.predict_button.setEnabled(False)

    def predict_images(self):
        """
        Predict classes for images
        :return:
        """
        predict.predict(self.model_path, self.images_path)


app = QtWidgets.QApplication([])
application = WindowOperator()
application.show()

sys.exit(app.exec())

import sys
import cv2
import numpy as np
import tensorflow as tf
import tflearn
import cv2
import pickle
from sklearn.cross_validation import train_test_split
import os
from skimage import color, io
from scipy.misc import imresize
from glob import glob
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QTextEdit,QLabel,QFileDialog,QCheckBox,QComboBox,QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.Qt import QImage,QPixmap
from PyQt5.QtCore import pyqtSlot
def selectBestBiger(arg):
    if arg[0]>arg[1] and arg[0]>arg[2]:
        return 'Fire'
    if arg[1]>arg[0] and arg[1]>arg[2]:
        return 'Smoke'
    if arg[2]>arg[0] and arg[2]>arg[1]:
        return 'Normal'
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title='Fire and Smoke detection'
        self.left=10
        self.top=10
        self.width=800
        self.heigh=800
        self.btn_text='Start'
        self.ready=False
        self.isOpened=False
        self.init_model()
        self.allX = np.zeros((1, 64, 64, 3), dtype='float64')
        self.mode='video'
        timer=QTimer(self)
        timer.timeout.connect(self.loop)
        timer.start(40)
        self.initUI()
    def init_model(self):
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)
        network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
        conv_1 = conv_2d(network, 16, 3, 1, 'same', activation='relu', name='conv_1')
        conv_2 = conv_2d(conv_1, 16, 3, 1, 'same', activation='relu', name='conv_2')
        network = max_pool_2d(conv_2, 2, 2, 'same')
        conv_3 = conv_2d(network, 16, 3, 1, 'same', activation='relu', name='conv_3')
        conv_4 = conv_2d(conv_3, 1, 3, 1, 'same', activation='relu', name='conv_4')
        network = max_pool_2d(conv_4, 2, 2, 'same')
        network1 = fully_connected(network, 100, activation='relu')
        network2 = fully_connected(network1, 100, activation='relu')
        network3 = fully_connected(network2, 3, activation='softmax')
        self.model = tflearn.DNN(network3)
        self.model.load('model_fire_smoke_6_final.tflearn', weights_only=False)
    def loop(self):
        if self.ready and self.isOpened:
            frame=self.cap.read()
            if frame[0]==True:
                frame_1=cv2.resize(frame[1],(600,480))
                frame_2=cv2.cvtColor(frame_1,cv2.COLOR_BGR2RGB)
                image=QImage(frame_2,frame_2.shape[1],frame_2.shape[0],frame_2.strides[0],QImage.Format_RGB888)
                self.label_display.setPixmap(QPixmap.fromImage(image))
                new_img = imresize(frame_1, (64, 64, 3))
                self.allX[0] = np.array(new_img)
                result=self.model.predict(self.allX)
                self.label_result.setText(selectBestBiger(result[0]))
            else:
                self.btn_start.setText('Start')
                self.btn_text = 'Start'
                self.ready = False
                self.cap = cv2.VideoCapture(self.fileName)
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.heigh)
        self.btn_browser=QPushButton('Browser',self)
        self.btn_browser.setToolTip('Open Video File')
        self.btn_browser.move(600,105)
        self.btn_browser.clicked.connect(self.on_browser_click)
        self.pathEdit=QTextEdit(self)
        self.pathEdit.setGeometry(100,100,450,28)
        self.btn_start=QPushButton('Start',self)
        self.btn_start.setToolTip('Start')
        self.btn_start.move(100,150)
        self.btn_start.clicked.connect(self.on_btn_start_click)
        self.btn_start_ip=QPushButton("Camera Connect",self)
        self.btn_start_ip.move(600,105)
        self.btn_start_ip.clicked.connect(self.on_start_ip_connected)
        self.label_display=QLabel(self)
        self.label_display.setGeometry(100,180,600,480)
        self.btn_start_ip.setVisible(False)
        self.label_result=QLabel(self)
        self.label_result.setGeometry(100,700,50,40)
        self.label_result.setText('Normal')
        self.ipAddress=QTextEdit(self)
        self.ipAddress.setGeometry(100,100,450,28)
        self.ipAddress.setVisible(False)
        self.btn_start.setVisible(True)
        self.comboMode=QComboBox(self)
        self.comboMode.addItem("VIDEO")
        self.comboMode.addItem("CAMERA")
        self.comboMode.setGeometry(100,50,100,30)
        self.comboMode.activated[str].connect(self.onChange)
        self.show()
    @pyqtSlot()
    def on_start_ip_connected(self):
        ipAdress=self.ipAddress.toPlainText()
        self.cap = cv2.VideoCapture(self.ipAddress.toPlainText())
        self.isOpened = self.cap.read()[0]
        if self.isOpened==False:
            msg=QMessageBox()
            msg.setText('Input error')
            msg.setWindowTitle('Warning')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
    def on_browser_click(self):
        self.openFileNameDialog()
        self.pathEdit.setText(self.fileName)
        self.cap=cv2.VideoCapture(self.fileName)
        self.isOpened=self.cap.read()[0]
        if self.isOpened==False:
            msg=QMessageBox()
            msg.setText('Input error')
            msg.setWindowTitle('Warning')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
    def onChange(self,text):
        if text=='CAMERA':
            self.btn_browser.setVisible(False)
            self.pathEdit.setVisible(False)
            self.ipAddress.setVisible(True)
            self.btn_start_ip.setVisible(True)
            self.mode='camera'
        if text=='VIDEO':
            self.mode='video'
            self.btn_browser.setVisible(True)
            self.pathEdit.setVisible(True)
            self.ipAddress.setVisible(False)
            self.btn_start_ip.setVisible(False)
    def on_btn_start_click(self):
        if self.isOpened==True:
            if self.btn_text=='Start':
                self.btn_start.setText('Stop')
                self.btn_text='Stop'
                self.ready=True
            else:
                self.btn_start.setText('Start')
                self.btn_text = 'Start'
                self.ready=False
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "","All Files (*);;Video Files (*.avi)", options=options)
        if self.fileName:
            print(self.fileName)
if __name__=='__main__':
    app=QApplication(sys.argv)
    ex=App()
    sys.exit(app.exec_())
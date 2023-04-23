import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import numpy as np

import cv2 as OpenCV

from ui_main_window import *

car_cascade = OpenCV.CascadeClassifier('Cascades/cars.xml')
stop = OpenCV.CascadeClassifier('Cascades/cascade_stop_3_15.xml')


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.pause = False

        self.timer = QTimer()

        self.timer.timeout.connect(self.viewCam)
        self.ui.control_bt.clicked.connect(self.togglePause)

        self.cap = OpenCV.VideoCapture("Videos/road.mp4")

    def viewCam(self):
        global xvoiture, yvoiture, wvoiture, hvoiture
        if self.pause:
            return
        ret, frame = self.cap.read()
        autoroute = False
        gray = OpenCV.cvtColor(frame, OpenCV.COLOR_BGR2GRAY)
        frame = OpenCV.cvtColor(frame, OpenCV.COLOR_BGR2RGB)
        gray = OpenCV.blur(gray,(8,8))
        edges = OpenCV.Canny(gray, 50, 200)
        cars = car_cascade.detectMultiScale(gray, 1.12, 7) #(image,Facteur d'aggrandissement, Voisins proches)
        stop_sign = stop.detectMultiScale(gray, 2.4, 6)
        try:
            for (x, y, w, h) in cars:
                OpenCV.putText(frame, 'Vehicule', (x, y), OpenCV.FONT_ITALIC, 1, (255, 255, 255), 2)
                OpenCV.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except:
            pass
        if autoroute == True:
            lines = OpenCV.HoughLinesP(edges, 2, np.pi/180, 15, minLineLength=5, maxLineGap=30)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    OpenCV.line(frame, (x1, y1), (x2, y2), (0, 0, 250), 3)
            if lines is None:
                pass
        else:
            try:
                for (x, y, w, h) in stop_sign:
                    OpenCV.putText(frame, 'STOP', (x, y), OpenCV.FONT_ITALIC, 1, (255, 255, 255), 2)
                    OpenCV.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except:
                pass
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        


    def togglePause(self):
        self.pause = not self.pause
        if self.pause:
            self.ui.control_bt.setText("Reprendre")
        else:
            self.ui.control_bt.setText("Mettre en pause")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
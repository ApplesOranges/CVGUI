import cv2
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
import CVGUI as gui
import sys
from PyQt5 import QtCore, QtGui, QtWidgets


def openFileNameDialog():
    try:
        fileName, _ = QFileDialog.getOpenFileName(None)
    except Exception:
        return
    if fileName:
        return fileName


def saveFileNameDialog():
    try:
        fileName, _ = QFileDialog.getSaveFileName(None)
    except Exception:
        return
    if fileName:
        return fileName


def threshHold():
    try:
        _, img2 = cv2.threshold(img, int(ui.param1), int(ui.param2), cv2.THRESH_BINARY)
    except Exception:
        return
    cv2.imwrite("display.jpg", img2)
    ui.label_2.setPixmap(QPixmap("display.jpg"))


def grayScale():
    try:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return
    cv2.imwrite("display.jpg", img2)
    ui.label_2.setPixmap(QPixmap("display.jpg"))


def smooth():
    try:
        img2 = cv2.bilateralFilter(img, 15, 100, 100)
    except Exception:
        return
    cv2.imwrite("display.jpg", img2)
    ui.label_2.setPixmap(QPixmap("display.jpg"))


def buttonPressed():
    if ui.comboBox.currentText() == "Umbralización":
        threshHold()
    if ui.comboBox.currentText() == "Escala de Grises":
        grayScale()
    if ui.comboBox.currentText() == "Suavizado":
        smooth()
    if ui.comboBox.currentText() == "LBP":
        LBP()
    if ui.comboBox.currentText() == "LTP":
        LTP()
    if ui.comboBox.currentText() == "Detección de iris":
        irisDetection()


def updateParams():
    ui.param1 = ui.lineEdit.text()
    ui.param2 = ui.lineEdit_2.text()


def openFile():
    filename = openFileNameDialog()
    print(filename)
    ui.label.setPixmap(QPixmap(filename))
    global img
    img = cv2.imread(filename)


def saveFile():
    img2 = cv2.imread("display.jpg")
    filename = saveFileNameDialog()
    print(filename)
    cv2.imwrite(filename, img2)


def setImage():
    img2 = cv2.imread("display.jpg")
    global img
    img = img2
    ui.label.setPixmap(QPixmap("display.jpg"))


def pixLBP(img, x, y):
    pix = img[x][y]
    binary = []
    binary.append(newPixVal(img, pix, x - 1, y - 1))
    binary.append(newPixVal(img, pix, x - 1, y))
    binary.append(newPixVal(img, pix, x - 1, y + 1))
    binary.append(newPixVal(img, pix, x, y + 1))
    binary.append(newPixVal(img, pix, x + 1, y + 1))
    binary.append(newPixVal(img, pix, x + 1, y))
    binary.append(newPixVal(img, pix, x + 1, y - 1))
    binary.append(newPixVal(img, pix, x, y - 1))
    exp = len(binary) - 1
    res = 0
    for i in binary:
        res += i * pow(2, exp)
        exp -= 1
    return res


def LBP():
    global img
    height, width, _ = img.shape
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            lbp[i, j] = pixLBP(img2, i, j)
    cv2.imwrite("display.jpg", lbp)
    ui.label_2.setPixmap(QPixmap("display.jpg"))


def pixLTP(img, x, y, t):
    pix = img[x][y]
    binary = []
    binary.append(newLTPPixVal(img, pix, x - 1, y - 1, t))
    binary.append(newLTPPixVal(img, pix, x - 1, y, t))
    binary.append(newLTPPixVal(img, pix, x - 1, y + 1, t))
    binary.append(newLTPPixVal(img, pix, x, y + 1, t))
    binary.append(newLTPPixVal(img, pix, x + 1, y + 1, t))
    binary.append(newLTPPixVal(img, pix, x + 1, y, t))
    binary.append(newLTPPixVal(img, pix, x + 1, y - 1, t))
    binary.append(newLTPPixVal(img, pix, x, y - 1, t))
    exp = len(binary) - 1
    upper = 0
    lower = 0
    for i in binary:
        if i > 0:
            upper += i * pow(2, exp)
            exp -= 1
        elif i < 0:
            lower += i * pow(2, exp)
            exp -= 1
    return upper, lower


def LTP():
    global img
    height, width, _ = img.shape
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ltpu = np.zeros((height, width), np.uint8)
    ltpl = np.zeros((height, width), np.uint8)
    ltp = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            (upper, lower) = pixLTP(img2, i, j, 3)#3 is the interval t
            ltpu[i, j] = upper
            ltpl[i, j] = lower
            ltp[i, j] = upper-lower/2
    cv2.imwrite("display.jpg", ltp)
    ui.label_2.setPixmap(QPixmap("display.jpg"))


def newPixVal(img, pix, x, y):
    val = 0
    try:
        if img[x][y] >= pix:
            val = 1
    except:
        pass
    return val


def newLTPPixVal(img, pix, x, y, t):
    val = 0
    try:
        if img[x][y] >= pix + t:
            val = 1
        elif img[x][y] <= pix - t:
            val = -1
    except:
        pass
    return val


def irisDetection():
    global img
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 100, 200)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1000000, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    img2 = img.copy()
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img2, (i[0], i[1]), i[2], (0, 125, 40), 2)
    cv2.imwrite("display.jpg", img2)
    ui.label_2.setPixmap(QPixmap("display.jpg"))


global img
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = gui.Ui_MainWindow()
ui.setupUi(MainWindow)
ui.pushButton_3.clicked.connect(openFile)
ui.pushButton_2.clicked.connect(setImage)
ui.pushButton.clicked.connect(buttonPressed)
ui.pushButton_4.clicked.connect(saveFile)
ui.lineEdit_2.textChanged.connect(updateParams)
ui.lineEdit.textChanged.connect(updateParams)
MainWindow.show()
sys.exit(app.exec_())

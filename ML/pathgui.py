from random import random, randint

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class AStarAnimator(QtWidgets.QWidget):
    def __init__(self, terrains):
        super().__init__()
        self.terrains = terrains
        self.paths = [[] for i in range(len(terrains))]
        self.current_terrain_index = 0
        self.painter = None

        # Create the QLabel that will display the terrain
        self.terrain_label = QtWidgets.QLabel(self)
        self.terrain_label.setAlignment(QtCore.Qt.AlignCenter)

        # Create the QSlider that will allow the user to switch between terrains
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, len(self.terrains) - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_terrain)

        # Create the QPushButton that will allow the user to save the path for the current terrain
        self.save_path_button = QtWidgets.QPushButton("Save Path", self)
        self.save_path_button.clicked.connect(self.save_path)

        # Create the QVBoxLayout that will contain the QLabel, QSlider, and QPushButton
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.terrain_label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.save_path_button)

        self.update_terrain()

    def update_terrain(self):
        self.current_terrain_index = self.slider.value()
        terrain = self.terrains[self.current_terrain_index]
        self.path = []
        self.paths[self.current_terrain_index] = []

        # Convert the terrain to a QImage and set it on the QLabel
        img = np.uint8(terrain * 255)
        qimage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.terrain_label.setPixmap(pixmap)
        self.terrain_label.setFixedSize(pixmap.size())

        if self.painter:
            del self.painter
        self.painter = QtGui.QPainter(pixmap)

        # Draw the path on the terrain
        path_pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 8)
        self.painter.setPen(path_pen)
        for i in range(len(self.paths[self.current_terrain_index]) - 1):
            x1, y1 = self.paths[self.current_terrain_index][i]
            x2, y2 = self.paths[self.current_terrain_index][i + 1]
            self.painter.drawLine(x1, y1, x2, y2)
        self.painter.end()
        self.terrain_label.setPixmap(pixmap)
        self.terrain_label.setFixedSize(pixmap.size())

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        self.path.append((x,y))
        self.update_terrain()

    def save_path(self):
        print("Saved")

if __name__ == '__main__':
    import sys

    # Create a random terrain
    terrain = np.zeros((100, 100))
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            terrain[i, j] = random()


    # Create the GUI
    app = QtWidgets.QApplication(sys.argv)
    gui = AStarAnimator([terrain])
    gui.show()
    sys.exit(app.exec_())
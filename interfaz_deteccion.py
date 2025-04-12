from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import sys
import os

class DetectionWorker(QThread):
    error_signal = pyqtSignal(str)

    def __init__(self, model_path, source):
        super().__init__()
        self.model_path = model_path
        self.source = source

    def run(self):
        try:
            model = YOLO(self.model_path)
            model.predict(source=self.source, show=True, conf=0.3)
        except Exception as e:
            self.error_signal.emit(str(e))

class YOLODetectorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Personas con Celular")
        self.setMinimumSize(400, 200)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "runs", "detect", "train4", "weights", "best.pt")

        self.initUI()

    def initUI(self):
        title = QLabel("🧠 Detector de personas con celular")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #4B8BBE")

        webcam_btn = QPushButton("Usar Webcam")
        webcam_btn.clicked.connect(self.run_webcam)
        webcam_btn.setStyleSheet(self.button_style())

        video_btn = QPushButton("Cargar Video y Detectar")
        video_btn.clicked.connect(self.load_video)
        video_btn.setStyleSheet(self.button_style())

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(webcam_btn)
        layout.addWidget(video_btn)

        self.setLayout(layout)

    def button_style(self):
        return (
            "QPushButton {"
            "  background-color: #306998;"
            "  color: white;"
            "  padding: 10px;"
            "  border-radius: 5px;"
            "  font-size: 14px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #204060;"
            "}"
        )

    def run_webcam(self):
        self.start_detection(source=0)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.start_detection(source=file_path)

    def start_detection(self, source):
        self.close()  
        model = YOLO(self.model_path)
        model.predict(source=source, show=True, conf=0.3)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", f"No se pudo ejecutar la detección:\n{message}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLODetectorUI()
    window.show()
    sys.exit(app.exec_())

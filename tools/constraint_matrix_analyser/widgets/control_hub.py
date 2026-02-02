from PyQt6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout)


class ControlHub(QGroupBox):

    def __init__(self, app):
        super().__init__("Controls")
        self.app = app
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # --- Preprocessing ---
        preprocess_layout = QHBoxLayout()
        preprocess_layout.addWidget(QLabel("Preprocessing:"))

        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItems(["None", "Normalize", "Blur", "Sharpen"])
        preprocess_layout.addWidget(self.preprocess_combo)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_preprocessing)
        preprocess_layout.addWidget(apply_btn)

        layout.addLayout(preprocess_layout)

        # --- Detection ---
        detect_layout = QHBoxLayout()
        detect_layout.addWidget(QLabel("Detection:"))

        self.detect_combo = QComboBox()
        self.detect_combo.addItems(
            ["Edge Detection", "Object Detection", "Motion Detection"])
        self.detect_combo.currentIndexChanged.connect(self.apply_detection)

        detect_layout.addWidget(self.detect_combo)
        layout.addLayout(detect_layout)

    def apply_preprocessing(self):
        technique = self.preprocess_combo.currentText()
        print("Apply preprocessing:", technique)
        self.app.apply_preprocessing(technique)

    def apply_detection(self):
        technique = self.detect_combo.currentText()
        print("Apply detection:", technique)
        self.app.apply_detection(technique)

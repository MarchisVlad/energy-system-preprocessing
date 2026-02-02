from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QScrollArea, QVBoxLayout,
                             QWidget)

from src.core.Presolving import PresolvingMethod

from ..widgets.history_tab import HistoryTab


class HistoryPanel(QGroupBox):
    """Preprocessing history and controls"""

    def __init__(self, app):
        super().__init__("Preprocessing History")
        self.app = app
        self.tabs: List[HistoryTab] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.container_layout = QHBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.placeholder = QLabel(
            "No model loaded. Select a model file to begin.")
        self.container_layout.addWidget(self.placeholder)

        scroll.setWidget(self.container)
        layout.addWidget(scroll)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Preprocessing Technique:"))

        self.preprocessing_techniques = QComboBox()
        for method in PresolvingMethod:
            self.preprocessing_techniques.addItem(method.name)

        controls.addWidget(self.preprocessing_techniques)

        self.apply_preprocessing_button = QPushButton("Apply")
        self.apply_preprocessing_button.clicked.connect(
            self.apply_preprocessing)
        controls.addWidget(self.apply_preprocessing_button)

        # self.normalize_btn = QPushButton("Normalize")
        # self.scale_btn = QPushButton("Scale")
        # self.remove_redundant_btn = QPushButton("Remove Redundant")

        # # TODO: connect preprocessing
        # controls.addWidget(self.normalize_btn)
        # controls.addWidget(self.scale_btn)
        # controls.addWidget(self.remove_redundant_btn)

        # controls.addWidget(QLabel("Tolerance:"))
        # self.tolerance = QDoubleSpinBox()
        # self.tolerance.setRange(0.0001, 1.0)
        # self.tolerance.setValue(0.001)
        # self.tolerance.setDecimals(4)
        # controls.addWidget(self.tolerance)

        controls.addStretch()
        layout.addLayout(controls)

    def apply_preprocessing(self):
        if not self.app.model_history:
            print("No model loaded!")
            return

        # Generate new matrix info (placeholder - simulate changes)
        current_state = self.app.model_history.get_current_state()
        if current_state:
            _, A = current_state

            new_matrix = A

            # Add new state
            self.app.model_history.add_state(
                PresolvingMethod[self.preprocessing_techniques.currentText()],
                new_matrix)
            self.update_history()
            self.app.update_matrix_display()

        self.update_history()

    def update_history(self):
        for tab in self.tabs:
            self.container_layout.removeWidget(tab)
            tab.deleteLater()
        self.tabs.clear()

        if self.placeholder:
            self.container_layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None

        mh = self.app.model_history
        if not mh:
            return

        for i, summary in enumerate(mh.get_history_summary()):
            state = mh.get_state_at_index(i)
            if not state:
                continue

            is_current = (i == mh.current_index)
            tab = HistoryTab(i, summary, is_current)
            tab.clicked.connect(self.app.on_history_tab_clicked)
            tab.right_clicked.connect(self.app.on_history_tab_right_clicked)

            self.tabs.append(tab)
            self.container_layout.addWidget(tab)

        self.container_layout.addStretch()

    def mark_current(self, index: int):
        for i, tab in enumerate(self.tabs):
            tab.set_current(i == index)

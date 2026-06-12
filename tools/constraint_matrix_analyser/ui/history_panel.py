from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.presolving import Presolver, PresolvingMethod
from src.utils.presolve_handler import PresolveHandler, _STATIC_ALGORITHM_MAP

from ..widgets.history_tab import HistoryTab

# PaPILO supports every PresolvingMethod; static only supports those in the map.
_PAPILO_METHODS = list(PresolvingMethod)
_STATIC_METHODS = list(_STATIC_ALGORITHM_MAP.keys())


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

        self.placeholder = QLabel("No model loaded. Select a model file to begin.")
        self.container_layout.addWidget(self.placeholder)

        scroll.setWidget(self.container)
        layout.addWidget(scroll)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Presolver:"))
        self.presolver_selector = QComboBox()
        for p in Presolver:
            self.presolver_selector.addItem(p.name)
        self.presolver_selector.currentIndexChanged.connect(self._on_presolver_changed)
        controls.addWidget(self.presolver_selector)

        controls.addWidget(QLabel("Technique:"))
        self.preprocessing_techniques = QComboBox()
        controls.addWidget(self.preprocessing_techniques)

        self.apply_preprocessing_button = QPushButton("Apply")
        self.apply_preprocessing_button.clicked.connect(self.apply_preprocessing)
        controls.addWidget(self.apply_preprocessing_button)

        controls.addStretch()
        layout.addLayout(controls)

        # Populate technique list for the default presolver selection.
        self._on_presolver_changed(0)

    def _on_presolver_changed(self, _index: int):
        presolver = Presolver[self.presolver_selector.currentText()]
        methods = _PAPILO_METHODS if presolver == Presolver.PaPILO else _STATIC_METHODS

        self.preprocessing_techniques.blockSignals(True)
        self.preprocessing_techniques.clear()
        for m in methods:
            self.preprocessing_techniques.addItem(m.name)
        self.preprocessing_techniques.blockSignals(False)

    def apply_preprocessing(self):
        if not self.app.model_history:
            print("No model loaded!")
            return

        current_state = self.app.model_history.get_current_state()
        if not current_state:
            return

        _, model = current_state

        method = PresolvingMethod[self.preprocessing_techniques.currentText()]
        presolver = Presolver[self.presolver_selector.currentText()]

        step = len(self.app.model_history.states)
        model = PresolveHandler.presolve(
            model=model,
            presolver=presolver,
            method=method,
            temp_dir=self.app.work_path,
            step=step,
        )

        self.app.model_history.add_state(method, model)
        self.update_history()
        self.app.update_matrix_display()

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

            is_current = i == mh.current_index
            tab = HistoryTab(i, summary, is_current)
            tab.clicked.connect(self.app.on_history_tab_clicked)
            tab.right_clicked.connect(self.app.on_history_tab_right_clicked)

            self.tabs.append(tab)
            self.container_layout.addWidget(tab)

        self.container_layout.addStretch()

    def mark_current(self, index: int):
        for i, tab in enumerate(self.tabs):
            tab.set_current(i == index)

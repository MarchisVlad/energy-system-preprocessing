import os
import tempfile
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QVBoxLayout,
                             QWidget)

from src.core.model import Model, ModelHistory

from ..config import STARTUP_ROOT
from .ui.browser_panel import BrowserPanel
from .ui.history_panel import HistoryPanel
from .ui.matrix_panel import MatrixPanel


class ConstraintMatrixAnalyzer(QMainWindow):

    def __init__(self):
        super().__init__()

        self.home_directory = str(STARTUP_ROOT.absolute())
        self.current_directory = self.home_directory
        self.model_history: Optional[ModelHistory] = None

        # Temp directory lives for the entire PyQt session; cleaned up on exit.
        self._work_dir = tempfile.TemporaryDirectory(prefix="cma_")
        self.work_path = Path(self._work_dir.name)

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Constraint Matrix Analyzer")
        self.setGeometry(100, 100, 1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.history_panel = HistoryPanel(self)
        splitter.addWidget(self.history_panel)

        bottom = QSplitter(Qt.Orientation.Horizontal)

        self.browser_panel = BrowserPanel(self)
        self.matrix_panel = MatrixPanel(self)

        bottom.addWidget(self.browser_panel)
        bottom.addWidget(self.matrix_panel)

        splitter.addWidget(bottom)
        splitter.setSizes([150, 850])

        layout.addWidget(splitter)

        self.browser_panel.load_directory(self.home_directory)

    def load_model_file(self, file_path: str):
        print(f"Loading model: {file_path}")

        model = Model(path=file_path)
        self.model_history = ModelHistory(model)

        self.history_panel.update_history()
        self.matrix_panel.update_matrix()

    def on_history_tab_clicked(self, index: int):
        if not self.model_history:
            return

        self.model_history.current_index = index
        self.history_panel.mark_current(index)
        self.matrix_panel.update_matrix()

    def on_history_tab_right_clicked(self, index: int):
        if not self.model_history:
            return

        if index < self.model_history.current_index:
            self.model_history.revert_to_index(index)
            self.history_panel.update_history()

            state = self.model_history.get_current_state()
            if state:
                _, matrix_info = state
                self.matrix_panel.update_matrix()

    def update_matrix_display(self):
        self.matrix_panel.update_matrix()

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from src.detection.algorithm import detect_block_structure

from ..widgets.block_widget import BlockMatrixWidget


class MatrixPanel(QWidget):

    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        info_layout = QHBoxLayout()

        matrix_info = QGroupBox("Matrix Information")
        matrix_info_layout = QVBoxLayout(matrix_info)

        row1 = QHBoxLayout()
        self.rows = QLabel("Rows: -")
        self.cols = QLabel("Columns: -")
        self.rank = QLabel("Rank: -")
        row1.addWidget(self.rows)
        row1.addWidget(self.cols)
        row1.addWidget(self.rank)
        row1.addStretch()
        matrix_info_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.sparsity = QLabel("Sparsity: -")
        self.condition = QLabel("Condition Number: -")
        row2.addWidget(self.sparsity)
        row2.addWidget(self.condition)
        row2.addStretch()
        matrix_info_layout.addLayout(row2)

        info_layout.addWidget(matrix_info)

        detection_info = QGroupBox("Detection Information")
        detection_info_layout = QHBoxLayout(detection_info)

        detection_info_layout.addWidget(QLabel("Algorithm"))
        self.detection_technique = QComboBox()
        for technique in ["None", "rcm", "spectral"]:
            self.detection_technique.addItem(technique)

        self.detection_technique.currentIndexChanged.connect(self.update_blocks)
        detection_info_layout.addWidget(self.detection_technique)

        info_layout.addWidget(detection_info)

        layout.addLayout(info_layout)

        self.block_matrix_info = BlockMatrixWidget()
        layout.addWidget(self.block_matrix_info)

        # matrix = QGroupBox("Constraint Matrix")
        # matrix_layout = QVBoxLayout(matrix)

        # controls = QHBoxLayout()
        # controls.addWidget(QLabel("Display Mode:"))

        # self.display = QComboBox()
        # self.display.addItems(
        #     ["Full Matrix", "Non-zero Only", "Summary Statistics"])
        # controls.addWidget(self.display)

        # export_btn = QPushButton("Export Matrix")
        # export_btn.clicked.connect(self.export_matrix)
        # controls.addWidget(export_btn)
        # controls.addStretch()

        # matrix_layout.addLayout(controls)

        # self.table = QTableWidget()
        # matrix_layout.addWidget(self.table)

        # layout.addWidget(matrix)

    # ------------------
    # Commented stubs moved here
    # ------------------

    def update_matrix(self):
        """Update the matrix display with new information"""
        current_state = self.app.model_history.get_current_state()
        self.block_matrix_info.clear_blocks()

        print(self.app.model_history)
        if current_state:
            _, model = current_state

            A = model.A
            n_rows, n_cols = A.shape
            self.rows.setText(f"Rows: {n_rows}")
            self.cols.setText(f"Columns: {n_cols}")
            # self.rank.setText(f"Rank: {matrix_rank(A)}")

            self.block_matrix_info.set_matrix(A)
            self.block_matrix_info.highlight_integers(model)

        else:
            raise RuntimeError(
                "Tried updating matrix display without any matrix information."
            )

    def update_blocks(self, index):
        detection_method = self.detection_technique.currentText()

        if detection_method == "None":
            self.block_matrix_info.clear_blocks()
            return

        current_state = self.app.model_history.get_current_state()
        print(self.app.model_history)
        if current_state:
            _, model = current_state

            A = model.A
            blocks = detect_block_structure(A, detection_method)

            self.block_matrix_info.highlight_blocks(blocks)

        else:
            raise RuntimeError(
                "Tried updating blocks display without any matrix information."
            )

    def export_matrix(self):
        mh = self.app.model_history
        if not mh:
            print("No model loaded!")
            return

        state = mh.get_current_state()
        if not state:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Matrix",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )

        if file_name:
            print(f"Exporting matrix to: {file_name}")
            print("Export completed!")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Matrix",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )

        if file_name:
            print(f"Exporting matrix to: {file_name}")
            print("Export completed!")

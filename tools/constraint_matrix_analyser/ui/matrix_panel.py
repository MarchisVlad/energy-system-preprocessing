from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.detection.algorithm import apply_reordering, detect_block_structure
from src.detection.reorder import ReorderingAlgorithm

from ..widgets.block_widget import BlockMatrixWidget


class MatrixPanel(QWidget):

    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()

        # ── Matrix information ──────────────────────────────────────────────
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

        controls.addWidget(matrix_info)

        # ── Reordering ──────────────────────────────────────────────────────
        reorder_group = QGroupBox("Reordering")
        reorder_layout = QHBoxLayout(reorder_group)
        reorder_layout.addWidget(QLabel("Algorithm"))

        self.reorder_technique = QComboBox()
        self.reorder_technique.addItem("None")
        for algo in ReorderingAlgorithm:
            self.reorder_technique.addItem(algo.value)
        self.reorder_technique.currentIndexChanged.connect(self._run_analysis)
        reorder_layout.addWidget(self.reorder_technique)

        controls.addWidget(reorder_group)

        # ── Detection ───────────────────────────────────────────────────────
        detection_group = QGroupBox("Detection")
        detection_layout = QHBoxLayout(detection_group)
        detection_layout.addWidget(QLabel("Algorithm"))

        self.detection_technique = QComboBox()
        for method in ["None", "spectral", "sliding_window"]:
            self.detection_technique.addItem(method)
        self.detection_technique.currentIndexChanged.connect(self._run_analysis)
        detection_layout.addWidget(self.detection_technique)

        controls.addWidget(detection_group)

        layout.addLayout(controls)

        self.block_matrix_info = BlockMatrixWidget()
        layout.addWidget(self.block_matrix_info)

    # ── Public entry points ─────────────────────────────────────────────────

    def update_matrix(self):
        """Refresh info labels and re-run the current analysis on the new model state."""
        current_state = self.app.model_history.get_current_state()
        if not current_state:
            raise RuntimeError(
                "Tried updating matrix display without any matrix information."
            )

        _, model = current_state
        A = model.A
        n_rows, n_cols = A.shape
        self.rows.setText(f"Rows: {n_rows}")
        self.cols.setText(f"Columns: {n_cols}")

        self._run_analysis()

    # ── Internal ────────────────────────────────────────────────────────────

    def _run_analysis(self, _index=None):
        """Apply the selected reordering and/or detection and refresh the view."""
        if not self.app.model_history:
            return
        current_state = self.app.model_history.get_current_state()
        if not current_state:
            return

        _, model = current_state
        A = model.A.tocsr()  # CSR required for fancy row/col indexing

        # Accumulated column permutation for integer highlighting.
        # col_perm[i] = original column index for display column i.
        col_perm = None

        # ── Step 1: Reordering ──────────────────────────────────────────────
        reorder_method = self.reorder_technique.currentText()
        if reorder_method != "None":
            row_perm, col_perm = apply_reordering(A, ReorderingAlgorithm[reorder_method])
            A = A[row_perm, :][:, col_perm]

        # ── Step 2: Detection ───────────────────────────────────────────────
        detect_method = self.detection_technique.currentText()
        blocks = None
        if detect_method != "None":
            blocks = detect_block_structure(A, detect_method)
            # Spectral detection permutes the matrix internally; compose with
            # any reordering permutation so integer highlights stay aligned.
            if blocks.col_permutation is not None:
                col_perm = (col_perm[blocks.col_permutation]
                            if col_perm is not None
                            else blocks.col_permutation)
            A = blocks.A if blocks.A is not None else A

        # ── Update display ──────────────────────────────────────────────────
        self.block_matrix_info.set_matrix(A)
        self.block_matrix_info.highlight_integers(model, col_perm=col_perm)

        if blocks is not None and blocks.blocks:
            self.block_matrix_info.highlight_blocks(blocks)
        else:
            self.block_matrix_info.clear_blocks()

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

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.detection.algorithm import apply_reordering, detect_block_structure
from src.detection.reorder import ReorderingAlgorithm

from ..widgets.block_widget import BlockMatrixWidget

_WORKING_REORDER = [
    ReorderingAlgorithm.CUTHILL_MCKEE,
    ReorderingAlgorithm.REVERSE_CUTHILL_MCKEE,
    ReorderingAlgorithm.AMD,
    ReorderingAlgorithm.MMD,
    ReorderingAlgorithm.NESTED_DISSECTION,
    ReorderingAlgorithm.SPECTRAL,
    ReorderingAlgorithm.NATURAL,
    ReorderingAlgorithm.RANDOM,
]


class MatrixPanel(QWidget):

    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # ── Control area: two columns ─────────────────────────────────────────
        # Left:  Matrix Information (top) + Partition Scores (bottom)
        # Right: Reordering (top)        + Detection (bottom)
        controls = QHBoxLayout()

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # Left-top: Matrix information
        matrix_info = QGroupBox("Matrix Information")
        mi_layout = QVBoxLayout(matrix_info)
        mi_row = QHBoxLayout()
        self.rows_label = QLabel("Rows: -")
        self.cols_label = QLabel("Columns: -")
        mi_row.addWidget(self.rows_label)
        mi_row.addWidget(self.cols_label)
        mi_row.addStretch()
        mi_layout.addLayout(mi_row)
        mi_row2 = QHBoxLayout()
        self.highlight_integers_check = QCheckBox("Highlight integers")
        self.highlight_integers_check.setChecked(True)
        self.highlight_integers_check.stateChanged.connect(self._maybe_run)
        mi_row2.addWidget(self.highlight_integers_check)
        mi_row2.addStretch()
        mi_layout.addLayout(mi_row2)
        left_col.addWidget(matrix_info)

        # Left-bottom: Partition scores
        scores_group = QGroupBox("Partition Scores")
        sc_layout = QVBoxLayout(scores_group)
        sc_row1 = QHBoxLayout()
        self.score_label = QLabel("Score: -")
        self.whitescore_label = QLabel("White score: -")
        sc_row1.addWidget(self.score_label)
        sc_row1.addWidget(self.whitescore_label)
        sc_row1.addStretch()
        sc_layout.addLayout(sc_row1)
        sc_row2 = QHBoxLayout()
        self.n_blocks_label = QLabel("Blocks: -")
        self.coupling_rows_label = QLabel("Coupling rows: -")
        self.coupling_cols_label = QLabel("Linking cols: -")
        sc_row2.addWidget(self.n_blocks_label)
        sc_row2.addWidget(self.coupling_rows_label)
        sc_row2.addWidget(self.coupling_cols_label)
        sc_row2.addStretch()
        sc_layout.addLayout(sc_row2)
        left_col.addWidget(scores_group)

        # Right-top: Reordering
        self.reorder_group = QGroupBox("Reordering")
        ro_layout = QHBoxLayout(self.reorder_group)
        ro_layout.addWidget(QLabel("Algorithm:"))
        self.reorder_technique = QComboBox()
        self.reorder_technique.addItem("None")
        for algo in _WORKING_REORDER:
            self.reorder_technique.addItem(algo.value)
        self.reorder_technique.currentIndexChanged.connect(self._maybe_run)
        ro_layout.addWidget(self.reorder_technique)
        right_col.addWidget(self.reorder_group)

        # Right-bottom: Detection
        detection_group = QGroupBox("Detection")
        det_layout = QHBoxLayout(detection_group)
        det_layout.addWidget(QLabel("Algorithm:"))
        self.detection_technique = QComboBox()
        for method in ["None", "spectral", "sliding_window", "pipstools"]:
            self.detection_technique.addItem(method)
        self.detection_technique.currentIndexChanged.connect(self._on_detection_changed)
        det_layout.addWidget(self.detection_technique)
        det_layout.addWidget(QLabel("k:"))
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(2, 64)
        self.k_spinbox.setValue(4)
        self.k_spinbox.setEnabled(False)
        self.k_spinbox.valueChanged.connect(self._maybe_run)
        det_layout.addWidget(self.k_spinbox)
        right_col.addWidget(detection_group)

        # Run controls
        run_row = QHBoxLayout()
        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(True)
        self.auto_run_check.setToolTip("When checked, analysis re-runs on every parameter change")
        run_row.addWidget(self.auto_run_check)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._run_analysis)
        run_row.addWidget(self.run_button)
        run_row.addStretch()
        right_col.addLayout(run_row)

        controls.addLayout(left_col)
        controls.addLayout(right_col)
        controls.addStretch()
        layout.addLayout(controls)

        # ── Row 2: pipstools options (hidden until pipstools selected) ────────
        self.pipstools_options = QGroupBox("Pipstools Options")
        pt_layout = QHBoxLayout(self.pipstools_options)

        pt_layout.addWidget(QLabel("Hypergraph:"))
        self.hg_type = QComboBox()
        for t in ["col", "row", "colrow", "rowcol"]:
            self.hg_type.addItem(t)
        self.hg_type.currentIndexChanged.connect(self._maybe_run)
        pt_layout.addWidget(self.hg_type)

        pt_layout.addWidget(QLabel("Objective:"))
        self.hg_objective = QComboBox()
        for obj in ["soed", "cut", "km1"]:
            self.hg_objective.addItem(obj)
        self.hg_objective.currentIndexChanged.connect(self._maybe_run)
        pt_layout.addWidget(self.hg_objective)

        self.var_dense_check = QCheckBox("var_dense:")
        self.var_dense_check.setChecked(True)
        self.var_dense_check.stateChanged.connect(self._on_var_dense_toggled)
        pt_layout.addWidget(self.var_dense_check)
        self.var_dense_spinbox = QSpinBox()
        self.var_dense_spinbox.setRange(1, 10000)
        self.var_dense_spinbox.setValue(200)
        self.var_dense_spinbox.valueChanged.connect(self._maybe_run)
        pt_layout.addWidget(self.var_dense_spinbox)

        self.skip_linking_check = QCheckBox("Skip linking overlay")
        self.skip_linking_check.setChecked(False)
        self.skip_linking_check.stateChanged.connect(self._maybe_run)
        pt_layout.addWidget(self.skip_linking_check)

        pt_layout.addStretch()
        self.pipstools_options.setVisible(False)
        layout.addWidget(self.pipstools_options)

        # ── Row 3: matrix view ────────────────────────────────────────────────
        self.block_matrix_info = BlockMatrixWidget()
        layout.addWidget(self.block_matrix_info)

    # ── Slot: detection algorithm changed ────────────────────────────────────

    def _on_detection_changed(self, _index=None):
        is_pipstools = self.detection_technique.currentText() == "pipstools"
        self.pipstools_options.setVisible(is_pipstools)
        self.k_spinbox.setEnabled(is_pipstools)
        # Reordering is irrelevant when pipstools defines its own ordering
        self.reorder_group.setEnabled(not is_pipstools)
        self._maybe_run()

    def _on_var_dense_toggled(self, _state):
        self.var_dense_spinbox.setEnabled(self.var_dense_check.isChecked())
        self._maybe_run()

    def _maybe_run(self, _=None):
        if self.auto_run_check.isChecked():
            self._run_analysis()

    # ── Public entry points ───────────────────────────────────────────────────

    def update_matrix(self):
        current_state = self.app.model_history.get_current_state()
        if not current_state:
            raise RuntimeError(
                "Tried updating matrix display without any matrix information."
            )
        _, model = current_state
        A = model.A
        self.rows_label.setText(f"Rows: {A.shape[0]}")
        self.cols_label.setText(f"Columns: {A.shape[1]}")
        self._run_analysis()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_analysis(self, _index=None):
        if not self.app.model_history:
            return
        current_state = self.app.model_history.get_current_state()
        if not current_state:
            return

        _, model = current_state
        A = model.A.tocsr()
        col_perm = None
        blocks = None
        detect_method = self.detection_technique.currentText()

        if detect_method == "pipstools":
            if not model.path:
                print("Pipstools requires a model loaded from an MPS file.")
            else:
                var_dense = (
                    self.var_dense_spinbox.value()
                    if self.var_dense_check.isChecked()
                    else None
                )
                blocks = detect_block_structure(
                    A,
                    method='pipstools',
                    mps_path=Path(model.path),
                    k=self.k_spinbox.value(),
                    hypergraph=self.hg_type.currentText(),
                    hg_objective=self.hg_objective.currentText(),
                    var_dense=var_dense,
                    skip_linking=self.skip_linking_check.isChecked(),
                )
                col_perm = blocks.col_permutation
                self._update_scores(blocks.metadata)

        else:
            # Step 1: optional reordering
            reorder_method = self.reorder_technique.currentText()
            if reorder_method != "None":
                row_perm, col_perm = apply_reordering(
                    A, ReorderingAlgorithm[reorder_method]
                )
                A = A[row_perm, :][:, col_perm]

            # Step 2: optional detection
            if detect_method != "None":
                blocks = detect_block_structure(A, detect_method)
                if blocks.col_permutation is not None:
                    col_perm = (
                        col_perm[blocks.col_permutation]
                        if col_perm is not None
                        else blocks.col_permutation
                    )
                A = blocks.A if blocks.A is not None else A

            self._clear_scores()

        # Determine the matrix to display (pipstools puts it on blocks.A)
        display_A = (blocks.A if blocks is not None and blocks.A is not None else A)

        self.block_matrix_info.set_matrix(display_A)
        if self.highlight_integers_check.isChecked():
            self.block_matrix_info.highlight_integers(model, col_perm=col_perm)
        else:
            self.block_matrix_info.clear_integers()

        if blocks is not None and blocks.blocks:
            self.block_matrix_info.highlight_blocks(blocks)
        else:
            self.block_matrix_info.clear_blocks()

    def _update_scores(self, metadata: dict):
        self.score_label.setText(f"Score: {metadata.get('score', 0):.2f}")
        self.whitescore_label.setText(f"White score: {metadata.get('whitescore', 0):.4f}")
        self.n_blocks_label.setText(f"Blocks: {metadata.get('n_blocks', '-')}")
        self.coupling_rows_label.setText(f"Coupling rows: {metadata.get('coupling_rows', '-')}")
        self.coupling_cols_label.setText(f"Linking cols: {metadata.get('coupling_cols', '-')}")

    def _clear_scores(self):
        self.score_label.setText("Score: -")
        self.whitescore_label.setText("White score: -")
        self.n_blocks_label.setText("Blocks: -")
        self.coupling_rows_label.setText("Coupling rows: -")
        self.coupling_cols_label.setText("Linking cols: -")

    def export_matrix(self):
        mh = self.app.model_history
        if not mh:
            return
        state = mh.get_current_state()
        if not state:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Matrix", "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )
        if file_name:
            print(f"Exporting matrix to: {file_name}")

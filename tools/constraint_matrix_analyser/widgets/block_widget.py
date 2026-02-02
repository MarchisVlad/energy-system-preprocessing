import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (QGraphicsRectItem, QHBoxLayout, QLabel, QScrollBar,
                             QVBoxLayout, QWidget)
from pyqtgraph.Qt import QtCore, QtGui

from src.core.BlockStructure import BlockStructure


class BlockMatrixWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set config options BEFORE creating widgets
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Info label at the top
        self.info_label = QLabel("Matrix: N/A | View: N/A")
        self.info_label.setStyleSheet(
            "padding: 5px; background-color: #f0f0f0;")
        main_layout.addWidget(self.info_label)

        # Container for graphics view and scrollbars
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # Horizontal layout for view + vertical scrollbar
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Graphics view
        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground('w')
        h_layout.addWidget(self.view)

        # Vertical scrollbar
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.v_scrollbar.valueChanged.connect(self._on_v_scroll)
        h_layout.addWidget(self.v_scrollbar)

        view_layout.addLayout(h_layout)

        # Horizontal scrollbar
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.h_scrollbar.valueChanged.connect(self._on_h_scroll)
        view_layout.addWidget(self.h_scrollbar)

        main_layout.addWidget(view_container)

        # Plot area
        self.plot_item = self.view.addViewBox()
        self.plot_item.invertY(True)  # Standard matrix orientation

        # Lock aspect ratio to 1:1 (no distortion)
        self.plot_item.setAspectLocked(True, ratio=1.0)

        # Enable mouse interaction
        self.plot_item.setMouseEnabled(x=True, y=True)

        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Store block overlays
        self.block_overlays = []

        # Store matrix dimensions
        self.matrix_shape = (0, 0)

        # Connect view change signal
        self.plot_item.sigRangeChanged.connect(self._on_view_changed)

        # Flag to prevent feedback loops
        self._updating_from_scrollbar = False

    def set_matrix(self, A):
        """Set the matrix and display heatmap"""
        import numpy as np
        import scipy.sparse as sp

        if sp.issparse(A):
            A = A.toarray()

        # Store dimensions
        self.matrix_shape = A.shape
        rows, cols = A.shape

        # Invert the colormap for black-on-white appearance
        inverted_A = np.max(A) - A

        self.image_item.setImage(inverted_A.T, autoLevels=True)

        # Get available viewport size
        viewport_rect = self.view.viewport().rect()
        viewport_width = viewport_rect.width()
        viewport_height = viewport_rect.height()

        # Calculate aspect ratios
        matrix_aspect = cols / rows if rows > 0 else 1
        viewport_aspect = viewport_width / viewport_height if viewport_height > 0 else 1

        # Determine view dimensions to fit one dimension and allow scrolling on the other
        if matrix_aspect > viewport_aspect:
            # Matrix is wider than viewport - fit height, scroll horizontally
            view_height = rows
            view_width = rows * viewport_aspect
        else:
            # Matrix is taller than viewport - fit width, scroll vertically
            view_width = cols
            view_height = cols / viewport_aspect

        # Set initial view range (top-left corner)
        self.plot_item.setRange(xRange=(0, view_width),
                                yRange=(0, view_height),
                                padding=0)

        # Set limits to matrix bounds
        self.plot_item.setLimits(xMin=0, xMax=cols, yMin=0, yMax=rows)

        # Update scrollbars
        self._update_scrollbar_ranges()
        self.update_view_info()

    def _update_scrollbar_ranges(self):
        """Update scrollbar ranges based on current view and matrix size"""
        rows, cols = self.matrix_shape

        # Get current view dimensions
        view_range = self.plot_item.viewRange()
        view_width = view_range[0][1] - view_range[0][0]
        view_height = view_range[1][1] - view_range[1][0]

        # Horizontal scrollbar
        h_max = max(0, int(cols - view_width))
        self.h_scrollbar.setRange(0, h_max)
        self.h_scrollbar.setPageStep(int(view_width))
        self.h_scrollbar.setSingleStep(max(1, int(view_width) // 10))

        # Vertical scrollbar
        v_max = max(0, int(rows - view_height))
        self.v_scrollbar.setRange(0, v_max)
        self.v_scrollbar.setPageStep(int(view_height))
        self.v_scrollbar.setSingleStep(max(1, int(view_height) // 10))

    def _on_h_scroll(self, value):
        """Handle horizontal scrollbar changes"""
        if self._updating_from_scrollbar:
            return

        self._updating_from_scrollbar = True

        view_range = self.plot_item.viewRange()
        view_width = view_range[0][1] - view_range[0][0]
        view_height = view_range[1][1] - view_range[1][0]

        # Calculate center Y to maintain it
        center_y = (view_range[1][0] + view_range[1][1]) / 2

        # Set new X range
        new_x_min = value
        new_x_max = value + view_width

        # Due to aspect lock, setting X might affect Y, so we recalculate
        self.plot_item.setRange(xRange=(new_x_min, new_x_max),
                                yRange=(center_y - view_height / 2,
                                        center_y + view_height / 2),
                                padding=0,
                                update=True)

        self._updating_from_scrollbar = False

    def _on_v_scroll(self, value):
        """Handle vertical scrollbar changes"""
        if self._updating_from_scrollbar:
            return

        self._updating_from_scrollbar = True

        view_range = self.plot_item.viewRange()
        view_width = view_range[0][1] - view_range[0][0]
        view_height = view_range[1][1] - view_range[1][0]

        # Calculate center X to maintain it
        center_x = (view_range[0][0] + view_range[0][1]) / 2

        # Set new Y range
        new_y_min = value
        new_y_max = value + view_height

        self.plot_item.setRange(xRange=(center_x - view_width / 2,
                                        center_x + view_width / 2),
                                yRange=(new_y_min, new_y_max),
                                padding=0,
                                update=True)

        self._updating_from_scrollbar = False

    def _on_view_changed(self):
        """Update scrollbars when view changes (from mouse pan/zoom)"""
        if self._updating_from_scrollbar:
            return

        # Small delay to let aspect lock settle
        QtCore.QTimer.singleShot(0, self._update_scrollbars_delayed)

    def _update_scrollbars_delayed(self):
        """Delayed scrollbar update to avoid aspect lock conflicts"""
        if self._updating_from_scrollbar:
            return

        # Update scrollbar ranges (they change with zoom)
        self._update_scrollbar_ranges()

        # Update scrollbar positions
        view_range = self.plot_item.viewRange()
        x_min = int(view_range[0][0])
        y_min = int(view_range[1][0])

        # Block signals to prevent feedback
        self.h_scrollbar.blockSignals(True)
        self.v_scrollbar.blockSignals(True)

        self.h_scrollbar.setValue(max(0, min(x_min,
                                             self.h_scrollbar.maximum())))
        self.v_scrollbar.setValue(max(0, min(y_min,
                                             self.v_scrollbar.maximum())))

        self.h_scrollbar.blockSignals(False)
        self.v_scrollbar.blockSignals(False)

        self.update_view_info()

    def update_view_info(self):
        """Update the info label with current view range"""
        if self.matrix_shape[0] == 0:
            return

        # Get current view range
        view_range = self.plot_item.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]

        # Clamp to matrix bounds
        x_min = max(0, int(x_range[0]))
        x_max = min(self.matrix_shape[1], int(x_range[1]))
        y_min = max(0, int(y_range[0]))
        y_max = min(self.matrix_shape[0], int(y_range[1]))

        # Update label
        info_text = (f"Matrix: {self.matrix_shape[0]}×{self.matrix_shape[1]} | "
                     f"View: Rows [{y_min}–{y_max}], Cols [{x_min}–{x_max}]")
        self.info_label.setText(info_text)

    def highlight_blocks(self, blocks: BlockStructure):
        """Highlight blocks with interactive hover info"""
        # Remove previous overlays
        for rect in self.block_overlays:
            self.plot_item.removeItem(rect)
        self.block_overlays.clear()

        for b in blocks.blocks:
            rect = QGraphicsRectItem(b.col_range[0], b.row_range[0],
                                     b.col_range[1] - b.col_range[0],
                                     b.row_range[1] - b.row_range[0])
            rect.setPen(pg.mkPen(color=(255, 0, 0), width=2))
            rect.setBrush(QBrush(QColor(255, 0, 0, 50)))  # semi-transparent

            # Enable hover events
            rect.setAcceptHoverEvents(True)
            rect.info_text = b.info

            # Connect hover events
            rect.hoverEnterEvent = self._make_hover_enter(rect)
            rect.hoverLeaveEvent = self._make_hover_leave(rect)

            self.plot_item.addItem(rect)
            self.block_overlays.append(rect)

    def _make_hover_enter(self, rect):

        def hoverEnter(event):
            QtGui.QToolTip.showText(event.screenPos().toPoint(), rect.info_text)

        return hoverEnter

    def _make_hover_leave(self, rect):

        def hoverLeave(event):
            QtGui.QToolTip.hideText()

        return hoverLeave

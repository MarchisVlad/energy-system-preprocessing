from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget


class HistoryTab(QWidget):
    """Custom widget for displaying a preprocessing history tab"""

    clicked = pyqtSignal(int)
    right_clicked = pyqtSignal(int)

    def __init__(self, index: int, text: str, is_current: bool = False):
        super().__init__()
        self.index = index
        self.is_current = is_current
        self.setMinimumWidth(180)
        self.setMaximumHeight(70)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self.title_label = QLabel(text)
        font = QFont()
        font.setPointSize(10)
        if is_current:
            font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        # self.time_label = QLabel(timestamp)
        # time_font = QFont()
        # time_font.setPointSize(8)
        # self.time_label.setFont(time_font)
        # layout.addWidget(self.time_label)

        # Style
        self.update_style()

    def update_style(self):
        """Update the visual style based on current state"""
        if self.is_current:
            self.setStyleSheet("""
                HistoryTab {
                    background-color: #4a90e2;
                    color: white;
                    border-radius: 5px;
                    border: 2px solid #357abd;
                }
                QLabel {
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                HistoryTab {
                    background-color: #e8e8e8;
                    border-radius: 5px;
                    border: 1px solid #cccccc;
                }
                QLabel {
                    color: black;
                }
            """)
            self.title_label.setStyleSheet("color: black;")
            # self.time_label.setStyleSheet("color: #666666;")

    def set_current(self, is_current: bool):
        """Set whether this tab is the current one"""
        self.is_current = is_current
        font = self.title_label.font()
        font.setBold(is_current)
        self.title_label.setFont(font)
        self.update_style()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self.index)

    def contextMenuEvent(self, event):
        """Show context menu on right-click"""
        menu = QMenu(self)
        revert_action = QAction("Revert to this state", self)
        revert_action.triggered.connect(
            lambda: self.right_clicked.emit(self.index))
        menu.addAction(revert_action)
        menu.exec(event.globalPos())

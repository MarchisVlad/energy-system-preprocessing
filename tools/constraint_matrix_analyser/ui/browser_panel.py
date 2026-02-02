import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout)


class BrowserPanel(QGroupBox):

    def __init__(self, app):
        super().__init__("File Browser")
        self.app = app
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        home = QHBoxLayout()
        home.addWidget(QLabel("Home Directory:"))

        self.home_dir = QLineEdit(self.app.home_directory)
        self.home_dir.setReadOnly(True)
        home.addWidget(self.home_dir)

        set_home = QPushButton("Set Home")
        set_home.clicked.connect(self.set_home_directory)
        home.addWidget(set_home)

        layout.addLayout(home)

        nav = QHBoxLayout()
        self.current_dir = QLineEdit(self.app.current_directory)
        self.current_dir.setReadOnly(True)
        nav.addWidget(self.current_dir)

        up = QPushButton("↑ Up")
        up.clicked.connect(self.go_up)
        nav.addWidget(up)

        layout.addLayout(nav)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.itemDoubleClicked.connect(self.on_double_click)
        layout.addWidget(self.tree)

    def load_directory(self, directory: str):
        self.tree.clear()

        try:
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                kind = "Folder" if os.path.isdir(path) else "File"
                item = QTreeWidgetItem([name, kind])
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                self.tree.addTopLevelItem(item)
        except Exception as e:
            print(f"Error loading directory: {e}")

    def set_home_directory(self):
        directory = QFileDialog.getExistingDirectory(self,
                                                     "Select Home Directory",
                                                     self.app.home_directory)
        if directory:
            self.app.home_directory = directory
            self.app.current_directory = directory
            self.home_dir.setText(directory)
            self.current_dir.setText(directory)
            self.load_directory(directory)

    def go_up(self):
        parent = os.path.dirname(self.app.current_directory)
        if parent and parent != self.app.current_directory:
            self.app.current_directory = parent
            self.current_dir.setText(parent)
            self.load_directory(parent)

    def on_double_click(self, item, _):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if os.path.isdir(path):
            self.app.current_directory = path
            self.current_dir.setText(path)
            self.load_directory(path)
        else:
            self.app.load_model_file(path)

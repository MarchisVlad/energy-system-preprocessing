import sys

from PyQt6.QtWidgets import QApplication

from tools.constraint_matrix_analyser.app import ConstraintMatrixAnalyzer


def main():
    app = QApplication(sys.argv)
    window = ConstraintMatrixAnalyzer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

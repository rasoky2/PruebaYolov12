"""
Widget para controles de modelo y performance - PyQt6 Version
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSpinBox, QGroupBox
)
from PyQt6.QtCore import Qt


class ControlsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Selección de Fruta
        fruit_group = QGroupBox("Evaluación")
        fruit_layout = QVBoxLayout()
        
        lbl_fruit = QLabel("Fruta a evaluar:")
        self.combo_fruit = QComboBox()
        self.combo_fruit.addItem("Plátano", "platano")
        self.combo_fruit.addItem("Manzana", "manzana")
        self.combo_fruit.addItem("Naranja", "naranja")
        
        fruit_layout.addWidget(lbl_fruit)
        fruit_layout.addWidget(self.combo_fruit)
        fruit_group.setLayout(fruit_layout)
        layout.addWidget(fruit_group)

        layout.addStretch()
        
        layout.addStretch()

    def get_fruit(self):
        """Obtener fruta seleccionada (folder name)"""
        return self.combo_fruit.currentData()

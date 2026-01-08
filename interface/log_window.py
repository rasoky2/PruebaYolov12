"""
Ventana separada para mostrar logs del sistema - PyQt6 Version
"""

import threading
import time
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTextEdit, QCheckBox, QLabel, QFileDialog, QStyle
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtGui import QTextCursor, QColor


class LogWindowSignals(QObject):
    log_received = pyqtSignal(str, str)


class LogWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Logs del Sistema")
        self.resize(800, 500)
        # Style inherited from main app via qApp, or can force if needed
        # self.setStyleSheet("background-color: #FFFFFF; color: #09090b;") # Handled by global

        self.signals = LogWindowSignals()
        self.signals.log_received.connect(self.append_log)

        self.log_buffer = deque(maxlen=1000)
        self.log_lock = threading.Lock()
        
        self.setup_ui()
        self.is_visible = False

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls Frame
        controls_layout = QHBoxLayout()
        
        # Buttons
        self.btn_clear = QPushButton(" Limpiar")
        self.btn_clear.setObjectName("Secondary")
        self.btn_clear.clicked.connect(self.clear_logs)
        
        self.btn_save = QPushButton(" Guardar")
        self.btn_save.setObjectName("Secondary")
        self.btn_save.clicked.connect(self.save_logs)
        
        controls_layout.addWidget(self.btn_clear)
        controls_layout.addWidget(self.btn_save)
        
        # Auto-scroll
        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        controls_layout.addWidget(self.chk_autoscroll)
        
        controls_layout.addStretch()
        
        # Line count
        self.lbl_lines = QLabel("Líneas: 0")
        controls_layout.addWidget(self.lbl_lines)
        
        layout.addLayout(controls_layout)
        
        # Text Area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        # Stylesheet is handled globally, but we might want specifics for colors
        # The global one sets generic text edit style.
        layout.addWidget(self.text_edit)

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
            self.is_visible = False
        else:
            self.show()
            self.raise_()
            self.activateWindow()
            self.is_visible = True

    def log(self, message: str, level: str = "INFO"):
        """Thread-safe logging method"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        with self.log_lock:
            self.log_buffer.append((log_entry, level))
            
        self.signals.log_received.emit(log_entry, level)

    def append_log(self, text, level):
        """Append log to text edit (runs in main thread via signal)"""
        color = self.get_log_color(level)
        formatted_text = f'<span style="color:{color}">{text}</span>'
        
        self.text_edit.append(formatted_text)
        
        if self.chk_autoscroll.isChecked():
            self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
            
        # Update counter
        # Rough estimate of lines based on blocks
        lines = self.text_edit.document().blockCount()
        self.lbl_lines.setText(f"Líneas: {lines}")
        
        # Limit lines (simple check)
        if lines > 2000:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            for _ in range(1000):
                cursor.movePosition(QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()

    def get_log_color(self, level):
        level_upper = level.upper()
        if "ERROR" in level_upper or "❌" in level:
            return "#ef4444"  # Red 500
        if "WARNING" in level_upper or "⚠️" in level:
            return "#f97316"  # Orange 500
        if "YOLO" in level_upper:
            return "#eab308"  # Yellow 500 range (formatted for light bg might need darker) -> #ca8a04
        if "CAMERA" in level_upper:
            return "#0ea5e9"  # Sky Blue
        if "DEBUG" in level_upper:
            return "#6366f1"  # Indigo
        if "INFO" in level_upper:
            return "#10b981"  # Emerald
        return "#18181b"      # Zinc 900

    def load_existing_logs(self):
        self.text_edit.clear()
        with self.log_lock:
            for log_entry, level in self.log_buffer:
                self.append_log(log_entry, level)

    def clear_logs(self):
        self.text_edit.clear()
        with self.log_lock:
            self.log_buffer.clear()
        self.lbl_lines.setText("Líneas: 0")

    def save_logs(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar logs", "", "Archivos de texto (*.txt);;Todos los archivos (*)"
        )
        
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    with self.log_lock:
                        for log_entry, _ in self.log_buffer:
                            f.write(log_entry + "\n")
                self.log(f"Logs guardados en: {filename}", "INFO")
            except Exception as e:
                self.log(f"Error guardando logs: {e}", "ERROR")

    def set_root_reference(self, root):
        pass # Not needed in PyQt6 logic as we use signals


# Instancia global para capturar prints
_log_window_instance = None


def setup_global_logging(log_window):
    """Configurar logging global para capturar prints"""
    global _log_window_instance
    _log_window_instance = log_window

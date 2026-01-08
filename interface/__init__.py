"""
Módulo de interfaz gráfica para CastañaSerial
Migrado a PyQt6
"""

from .log_window import LogWindow, setup_global_logging
from .main_window import CastañaSerialInterface


def main():
    """Función principal para iniciar la interfaz"""
    import sys
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    
    # Crear ventana principal
    main_window = CastañaSerialInterface()
    main_window.show()

    # Configurar cierre
    app.aboutToQuit.connect(main_window.stop_detection)
    
    sys.exit(app.exec())


__all__ = ["CastañaSerialInterface", "LogWindow", "main"]

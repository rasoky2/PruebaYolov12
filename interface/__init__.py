"""
Módulo de interfaz gráfica para CastañaSerial
Separado en widgets especializados para mejor mantenibilidad
"""

from .main_window import CastañaSerialInterface
from .log_window import LogWindow

def main():
    """Función principal para iniciar la interfaz"""
    import tkinter as tk
    
    root = tk.Tk()
    app = CastañaSerialInterface(root)
    
    # Configurar cierre de ventana
    def on_closing():
        app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

__all__ = ['CastañaSerialInterface', 'main']

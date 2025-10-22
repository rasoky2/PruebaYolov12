#!/usr/bin/env python3
"""
Interfaz gráfica para CastañaSerial
Punto de entrada principal - ahora usa módulos separados
"""

import tkinter as tk

from interface import CastañaSerialInterface


def main():
    """Función principal"""
    root = tk.Tk()
    app = CastañaSerialInterface(root)

    # Configurar cierre de ventana
    def on_closing():
        app.stop_detection()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

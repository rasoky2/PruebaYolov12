"""
Widget para manejo de Arduino y comunicación serial
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
from arduino.arduino_manager import ArduinoManager


class ArduinoPanel:
    def __init__(self, parent):
        self.parent = parent
        self.arduino_manager = ArduinoManager()
        
        # Variables UI
        self.arduino_ports = []
        self.arduino_port_var = tk.StringVar()
        
        # UI Elements
        self.arduino_combo = None
        self.arduino_connect_btn = None
        self.arduino_disconnect_btn = None
        self.arduino_status_label = None
        self.arduino_last_signal_label = None
        
        self.create_ui()
        self.refresh_arduino_ports()
    
    def create_ui(self):
        """Crear interfaz del panel de Arduino"""
        arduino_frame = ttk.LabelFrame(self.parent, text="Arduino", padding=5)
        arduino_frame.pack(fill=tk.X, pady=5)
        
        # Fila selector de puerto
        row1 = ttk.Frame(arduino_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Puerto:").pack(side=tk.LEFT)
        self.arduino_combo = ttk.Combobox(row1, textvariable=self.arduino_port_var, state="readonly")
        self.arduino_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(row1, text="Refrescar", command=self.refresh_arduino_ports).pack(side=tk.RIGHT)

        # Fila conectar/desconectar
        row2 = ttk.Frame(arduino_frame)
        row2.pack(fill=tk.X, pady=2)
        self.arduino_connect_btn = ttk.Button(row2, text="Conectar", command=self.connect_arduino)
        self.arduino_connect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.arduino_disconnect_btn = ttk.Button(row2, text="Desconectar", command=self.disconnect_arduino, state=tk.DISABLED)
        self.arduino_disconnect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Fila pruebas
        row3 = ttk.Frame(arduino_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Button(row3, text="Prueba SANO", command=lambda: self.test_arduino_signal('SANO')).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(row3, text="Prueba CONTAMINADO", command=lambda: self.test_arduino_signal('CONTAMINADO')).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Estado
        self.arduino_status_label = ttk.Label(arduino_frame, text="Desconectado", foreground='red')
        self.arduino_status_label.pack(fill=tk.X, pady=(4, 0))

        # Última señal enviada
        self.arduino_last_signal_label = ttk.Label(arduino_frame, text="Última señal: -")
        self.arduino_last_signal_label.pack(fill=tk.X, pady=(2, 0))

    def refresh_arduino_ports(self):
        """Listar puertos seriales disponibles y cargar en el combo"""
        try:
            ports = self.arduino_manager.list_available_ports() if self.arduino_manager.is_serial_available() else []
        except Exception:
            ports = []
        self.arduino_ports = ports

        # Reordenar para que el puerto autodetectado (Arduino) quede primero
        auto_port = None
        try:
            auto_port = self.arduino_manager.auto_detect_port() if self.arduino_manager.is_serial_available() else None
        except Exception:
            auto_port = None

        values = [p.device for p in ports]
        if auto_port and auto_port in values:
            values = [auto_port] + [v for v in values if v != auto_port]

        self.arduino_combo['values'] = values
        if values:
            # Seleccionar por defecto el autodetectado si existe, si no el primero
            self.arduino_combo.set(values[0])
        else:
            self.arduino_combo.set('')

    def connect_arduino(self):
        """Conectar Arduino al puerto seleccionado"""
        if not self.arduino_manager.is_serial_available():
            messagebox.showerror("Arduino", "PySerial no está instalado. Instala con: pip install pyserial")
            return
        port = self.arduino_port_var.get().strip() or None
        if not port:
            # Intentar autodetección si no se eligió
            port = self.arduino_manager.auto_detect_port()
            if not port:
                messagebox.showwarning("Arduino", "No hay puerto seleccionado ni autodetectado")
                return
        ok = self.arduino_manager.connect(port)
        if ok:
            self.arduino_status_label.config(text=f"Conectado: {port}", foreground='green')
            self.arduino_connect_btn.config(state=tk.DISABLED)
            self.arduino_disconnect_btn.config(state=tk.NORMAL)
        else:
            self.arduino_status_label.config(text="Desconectado", foreground='red')

    def disconnect_arduino(self):
        """Desconectar Arduino"""
        self.arduino_manager.disconnect()
        self.arduino_status_label.config(text="Desconectado", foreground='red')
        self.arduino_connect_btn.config(state=tk.NORMAL)
        self.arduino_disconnect_btn.config(state=tk.DISABLED)

    def test_arduino_signal(self, signal: str):
        """Enviar señal de prueba al Arduino"""
        if not self.arduino_manager.enabled:
            messagebox.showwarning("Arduino", "Conecta el Arduino antes de probar")
            return
        sent = self.arduino_manager.send_signal(signal, delay_prevention=0.0)
        if sent:
            messagebox.showinfo("Arduino", f"Señal enviada: {signal}")
            try:
                ts = time.strftime('%H:%M:%S')
                self.arduino_last_signal_label.config(text=f"Última señal: {signal} a las {ts}")
            except Exception:
                pass
        else:
            messagebox.showwarning("Arduino", "No se pudo enviar la señal (posible limitación de frecuencia)")
    
    def send_signal(self, signal: str, delay_prevention: float = 0.3):
        """Enviar señal al Arduino"""
        if not self.arduino_manager.enabled:
            return False
        
        sent = self.arduino_manager.send_signal(signal, delay_prevention)
        if sent:
            try:
                ts = time.strftime('%H:%M:%S')
                self.arduino_last_signal_label.config(text=f"Última señal: {signal} a las {ts}")
            except Exception:
                pass
        return sent
    
    def is_connected(self):
        """Verificar si Arduino está conectado"""
        return self.arduino_manager.enabled
    
    def get_manager(self):
        """Obtener el manager de Arduino"""
        return self.arduino_manager

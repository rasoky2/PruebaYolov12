"""
Widget para mostrar estadísticas en tiempo real
"""

import tkinter as tk
from tkinter import ttk


class StatsPanel:
    def __init__(self, parent):
        self.parent = parent

        # Variables
        self.stats = {
            'total_detections': 0,
            'sanas': 0,
            'contaminadas': 0,
            'frames_processed': 0,
            'classes_ignored': 0
        }

        # UI Elements
        self.stats_labels = {}

        self.create_ui()

    def create_ui(self):
        """Crear interfaz del panel de estadísticas"""
        stats_frame = ttk.LabelFrame(self.parent, text="Estadísticas", padding=5)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        stats_data = [
            ('Cámara Actual:', 'current_camera'),
            ('Frames Procesados:', 'frames_processed'),
            ('FPS Real:', 'real_fps'),
            ('RAM (MB):', 'memory_usage'),
            ('CPU (%):', 'cpu_usage'),
            ('Total Detecciones:', 'total_detections'),
            ('Objetos Sanos:', 'sanas'),
            ('Objetos Contaminados:', 'contaminadas'),
            ('Arduino:', 'arduino_status'),
            ('Aceleración:', 'accel_status'),
            ('Modelo:', 'model_status')
        ]

        for label_text, key in stats_data:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            self.stats_labels[key] = ttk.Label(frame, text="0", font=('Arial', 10, 'bold'))
            self.stats_labels[key].pack(side=tk.RIGHT)

        # Separador
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Botón para resetear estadísticas
        reset_btn = ttk.Button(stats_frame, text="Resetear Estadísticas",
                              command=self.reset_stats)
        reset_btn.pack(fill=tk.X, pady=5)

    def update_stats(self, **kwargs):
        """Actualizar estadísticas"""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value
        self.update_display()

    def update_display(self):
        """Actualizar visualización de estadísticas"""
        # Actualizar estadísticas normales
        for key, value in self.stats.items():
            if key in self.stats_labels:
                self.stats_labels[key].config(text=str(value))

    def update_camera_info(self, camera_id: int, camera_name: str):
        """Actualizar información de cámara"""
        if 'current_camera' in self.stats_labels:
            self.stats_labels['current_camera'].config(text=f"{camera_id}: {camera_name}")

    def update_arduino_status(self, connected: bool, port: str = None, last_signal: str = None):
        """Actualizar estado de Arduino"""
        if 'arduino_status' in self.stats_labels:
            if connected:
                port_text = port or 'N/D'
                last_text = last_signal or '-'
                self.stats_labels['arduino_status'].config(text=f"Conectado ({port_text}) / Última: {last_text}")
            else:
                self.stats_labels['arduino_status'].config(text="Desconectado")

    def update_acceleration_status(self, accel_text: str):
        """Actualizar estado de aceleración"""
        if 'accel_status' in self.stats_labels:
            self.stats_labels['accel_status'].config(text=accel_text)

    def update_fps(self, fps: float):
        """Actualizar FPS real"""
        if 'real_fps' in self.stats_labels:
            self.stats_labels['real_fps'].config(text=f"{fps:.1f}")

    def update_model_status(self, model_name: str):
        """Actualizar modelo actual"""
        if 'model_status' in self.stats_labels:
            self.stats_labels['model_status'].config(text=model_name)

    def update_performance_info(self, memory_mb: float, cpu_percent: float):
        """Actualizar información de rendimiento"""
        if 'memory_usage' in self.stats_labels:
            self.stats_labels['memory_usage'].config(text=f"{memory_mb:.1f}")
        if 'cpu_usage' in self.stats_labels:
            self.stats_labels['cpu_usage'].config(text=f"{cpu_percent:.1f}")

    def reset_stats(self):
        """Resetear estadísticas"""
        self.stats = {
            'total_detections': 0,
            'sanas': 0,
            'contaminadas': 0,
            'frames_processed': 0,
            'classes_ignored': 0
        }
        self.update_display()

    def get_stats(self):
        """Obtener estadísticas actuales"""
        return self.stats.copy()

"""
Widget para mostrar detecciones en tiempo real
"""

import tkinter as tk
from tkinter import ttk


class DetectionPanel:
    def __init__(self, parent):
        self.parent = parent

        # UI Elements
        self.detections_listbox = None

        self.create_ui()

    def create_ui(self):
        """Crear interfaz del panel de detecciones"""
        detections_frame = ttk.LabelFrame(self.parent, text="Detecciones en Tiempo Real", padding=5)
        detections_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Listbox para mostrar detecciones
        self.detections_listbox = tk.Listbox(detections_frame, height=8, font=('Consolas', 10))
        self.detections_listbox.pack(fill=tk.BOTH, expand=True)

        # Scrollbar para la lista
        scrollbar = ttk.Scrollbar(detections_frame, orient=tk.VERTICAL, command=self.detections_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detections_listbox.config(yscrollcommand=scrollbar.set)

    def add_detection(self, detection_text: str):
        """Agregar una detección a la lista"""
        self.detections_listbox.insert(0, detection_text)
        # Mantener solo las últimas 50 detecciones
        if self.detections_listbox.size() > 50:
            self.detections_listbox.delete(50, tk.END)

    def add_detections(self, detections_list):
        """Agregar múltiples detecciones"""
        for detection in detections_list:
            self.add_detection(detection)

    def clear_detections(self):
        """Limpiar todas las detecciones"""
        self.detections_listbox.delete(0, tk.END)

    def get_detection_count(self):
        """Obtener número de detecciones mostradas"""
        return self.detections_listbox.size()

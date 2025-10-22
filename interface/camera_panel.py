"""
Widget para manejo de cámaras y configuración
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

import cv2


class CameraPanel:
    def __init__(self, parent, camera_info_callback=None):
        self.parent = parent
        self.camera_info_callback = camera_info_callback

        # Variables
        self.camera_config = None
        self.available_cameras = []
        self.camera_info = {}
        self.camera_id = 0

        # UI Elements
        self.camera_var = None
        self.camera_combo = None
        self.camera_info_frame = None

        self.load_camera_config()
        self.detect_cameras()
        self.create_ui()

    def create_ui(self):
        """Crear interfaz del panel de cámara"""
        camera_frame = ttk.LabelFrame(self.parent, text="Cámara", padding=5)
        camera_frame.pack(fill=tk.X, pady=5)

        # Información de cámaras disponibles
        if not self.available_cameras:
            ttk.Label(camera_frame, text="No hay cámaras disponibles",
                     foreground='red').pack(pady=5)
            return

        # Selector de cámara
        camera_select_frame = ttk.Frame(camera_frame)
        camera_select_frame.pack(fill=tk.X, pady=2)

        ttk.Label(camera_select_frame, text="Cámara:").pack(side=tk.LEFT)

        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_select_frame, textvariable=self.camera_var,
                                        state="readonly", width=15)
        self.camera_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Llenar combo con cámaras disponibles
        self.update_camera_options()

        # Información de la cámara seleccionada
        self.camera_info_frame = ttk.Frame(camera_frame)
        self.camera_info_frame.pack(fill=tk.X, pady=5)

        self.update_camera_info_display()

        # Botón para aplicar cambios de cámara
        apply_camera_btn = ttk.Button(camera_frame, text="Cambiar Cámara",
                                     command=self.change_camera)
        apply_camera_btn.pack(fill=tk.X, pady=2)

    def load_camera_config(self):
        """Cargar configuración de cámaras desde JSON"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera_config.json")
            with open(config_path, encoding='utf-8') as f:
                self.camera_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.camera_config = None

    def get_camera_name(self, cam_id: int) -> str:
        """Obtener nombre personalizado de la cámara"""
        if self.camera_config and "cameras" in self.camera_config and str(cam_id) in self.camera_config["cameras"]:
            return self.camera_config["cameras"][str(cam_id)]["name"]
        return f"Dispositivo {cam_id}"

    def get_camera_description(self, cam_id: int) -> str:
        """Obtener descripción de la cámara"""
        if self.camera_config and "cameras" in self.camera_config and str(cam_id) in self.camera_config["cameras"]:
            return self.camera_config["cameras"][str(cam_id)].get("description", "")
        return ""

    def is_favorite_camera(self, cam_id: int) -> bool:
        """Verificar si la cámara es favorita"""
        if self.camera_config and "cameras" in self.camera_config and str(cam_id) in self.camera_config["cameras"]:
            return self.camera_config["cameras"][str(cam_id)].get("is_favorite", False)
        return False

    def detect_cameras(self):
        """Detectar cámaras disponibles"""
        self.available_cameras = []
        self.camera_info = {}

        for i in range(5):  # Probar cámaras 0-4
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Obtener información de la cámara
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Obtener nombre personalizado desde configuración
                device_name = self.get_camera_name(i)
                device_description = self.get_camera_description(i)
                is_favorite = self.is_favorite_camera(i)

                self.camera_info[i] = {
                    'name': device_name,
                    'description': device_description,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'is_favorite': is_favorite
                }

                self.available_cameras.append(i)
                cap.release()

        # Establecer cámara por defecto (favorita o primera disponible)
        if self.available_cameras:
            favorite_cam = None
            for cam_id in self.available_cameras:
                if self.camera_info[cam_id]['is_favorite']:
                    favorite_cam = cam_id
                    break

            self.camera_id = favorite_cam if favorite_cam else self.available_cameras[0]

    def update_camera_options(self):
        """Actualizar opciones del combobox de cámaras"""
        camera_options = []
        for cam_id in self.available_cameras:
            cam_info = self.camera_info[cam_id]
            favorite_marker = " ⭐" if cam_info['is_favorite'] else ""
            camera_options.append(f"{cam_id}: {cam_info['name']}{favorite_marker}")

        self.camera_combo['values'] = camera_options

        # Establecer cámara actual
        current_cam_info = self.camera_info[self.camera_id]
        favorite_marker = " ⭐" if current_cam_info['is_favorite'] else ""
        current_option = f"{self.camera_id}: {current_cam_info['name']}{favorite_marker}"
        self.camera_combo.set(current_option)

    def update_camera_info_display(self):
        """Actualizar visualización de información de cámara"""
        # Limpiar frame anterior
        for widget in self.camera_info_frame.winfo_children():
            widget.destroy()

        if self.camera_id in self.camera_info:
            cam_info = self.camera_info[self.camera_id]

            # Nombre de la cámara
            name_frame = ttk.Frame(self.camera_info_frame)
            name_frame.pack(fill=tk.X, pady=1)
            ttk.Label(name_frame, text="Nombre:", font=('Arial', 8)).pack(side=tk.LEFT)
            ttk.Label(name_frame, text=cam_info['name'], font=('Arial', 8, 'bold')).pack(side=tk.RIGHT)

            # Resolución
            res_frame = ttk.Frame(self.camera_info_frame)
            res_frame.pack(fill=tk.X, pady=1)
            ttk.Label(res_frame, text="Resolución:", font=('Arial', 8)).pack(side=tk.LEFT)
            ttk.Label(res_frame, text=cam_info['resolution'], font=('Arial', 8)).pack(side=tk.RIGHT)

            # FPS
            fps_frame = ttk.Frame(self.camera_info_frame)
            fps_frame.pack(fill=tk.X, pady=1)
            ttk.Label(fps_frame, text="FPS:", font=('Arial', 8)).pack(side=tk.LEFT)
            ttk.Label(fps_frame, text=f"{cam_info['fps']:.1f}", font=('Arial', 8)).pack(side=tk.RIGHT)

            # Estado favorito
            if cam_info['is_favorite']:
                ttk.Label(self.camera_info_frame, text="⭐ FAVORITA", foreground='gold',
                         font=('Arial', 8, 'bold')).pack(pady=2)

    def change_camera(self):
        """Cambiar la cámara seleccionada"""
        selected_option = self.camera_combo.get()
        if not selected_option:
            return

        # Extraer ID de cámara del texto seleccionado
        try:
            new_camera_id = int(selected_option.split(':')[0])
            if new_camera_id in self.available_cameras:
                self.camera_id = new_camera_id
                self.update_camera_info_display()
                messagebox.showinfo("Cámara cambiada",
                                  f"Cámara cambiada a: {self.camera_info[self.camera_id]['name']}")

                # Notificar cambio al callback
                if self.camera_info_callback:
                    self.camera_info_callback(self.camera_id, self.camera_info[self.camera_id])
            else:
                messagebox.showerror("Error", "Cámara no disponible")
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Error al procesar selección de cámara")

    def get_camera_id(self):
        """Obtener ID de cámara actual"""
        return self.camera_id

    def get_camera_info(self):
        """Obtener información de cámara actual"""
        return self.camera_info.get(self.camera_id, {})

    def get_all_camera_info(self):
        """Obtener información de todas las cámaras"""
        return self.camera_info

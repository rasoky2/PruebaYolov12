#!/usr/bin/env python3
"""
Interfaz gráfica para CastañaSerial
Visualización de detecciones y configuración de parámetros
"""

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import json
import os
from ultralytics import YOLO
from functions.analysys import analyze_chestnut_quality_dual, get_contamination_details
from typing import Dict, Any, Optional


class CastañaSerialInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("CastañaSerial - Interfaz de Control")
        self.root.geometry("1200x800")
        
        # Variables del sistema
        self.model = None
        self.camera = None
        self.camera_id = 0
        self.available_cameras = []
        self.camera_info = {}
        self.running = False
        self.detection_thread = None
        
        # Configuración de colores (configurable)
        self.colors = {
            'sana': '#00FF00',      # Verde
            'contaminada': '#FF0000',  # Rojo
            'detectada': '#FFFF00',    # Amarillo
            'fondo': '#2C3E50'        # Azul oscuro
        }
        
        # Configuración de clases
        self.chestnut_classes = ['sports ball', 'apple', 'orange', 'donut', 'bowl', 'carrot', 'banana']
        self.detected_objects = {}
        
        # Configuración de parámetros RGB
        self.rgb_params = {
            'brightness_low': 70,
            'brightness_medium': 100,
            'variation_threshold': 35.0,
            'edge_density_threshold': 0.15,
            'contamination_threshold': 1
        }
        
        # Estadísticas
        self.stats = {
            'total_detections': 0,
            'sanas': 0,
            'contaminadas': 0,
            'frames_processed': 0
        }
        
        self.load_camera_config()
        self.detect_cameras()
        self.setup_ui()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superior - Video y controles
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Video
        video_frame = ttk.LabelFrame(top_frame, text="Video en Tiempo Real", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(video_frame, text="Cámara no iniciada", 
                                   font=('Arial', 12), foreground='white', 
                                   background=self.colors['fondo'])
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel derecho - Controles
        controls_frame = ttk.LabelFrame(top_frame, text="Controles", padding=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Configuración de cámara
        camera_frame = ttk.LabelFrame(controls_frame, text="Cámara", padding=5)
        camera_frame.pack(fill=tk.X, pady=5)
        
        self.create_camera_controls(camera_frame)
        
        # Separador
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Botones de control
        self.start_btn = ttk.Button(controls_frame, text="Iniciar Detección", 
                                  command=self.toggle_detection)
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = ttk.Button(controls_frame, text="Detener", 
                                 command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Separador
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Configuración de colores
        color_frame = ttk.LabelFrame(controls_frame, text="Configurar Colores", padding=5)
        color_frame.pack(fill=tk.X, pady=5)
        
        self.create_color_controls(color_frame)
        
        # Separador
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Configuración de parámetros RGB
        params_frame = ttk.LabelFrame(controls_frame, text="Parámetros RGB", padding=5)
        params_frame.pack(fill=tk.X, pady=5)
        
        self.create_rgb_controls(params_frame)
        
        # Frame inferior - Detecciones y estadísticas
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Panel de detecciones
        detections_frame = ttk.LabelFrame(bottom_frame, text="Detecciones en Tiempo Real", padding=5)
        detections_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.create_detections_panel(detections_frame)
        
        # Panel de estadísticas
        stats_frame = ttk.LabelFrame(bottom_frame, text="Estadísticas", padding=5)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        self.create_stats_panel(stats_frame)
    
    def load_camera_config(self):
        """Cargar configuración de cámaras desde JSON"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
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
    
    def create_camera_controls(self, parent):
        """Crear controles de selección de cámara"""
        
        # Información de cámaras disponibles
        if not self.available_cameras:
            ttk.Label(parent, text="No hay cámaras disponibles", 
                     foreground='red').pack(pady=5)
            return
        
        # Selector de cámara
        camera_select_frame = ttk.Frame(parent)
        camera_select_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(camera_select_frame, text="Cámara:").pack(side=tk.LEFT)
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_select_frame, textvariable=self.camera_var, 
                                        state="readonly", width=15)
        self.camera_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Llenar combo con cámaras disponibles
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
        
        # Información de la cámara seleccionada
        self.camera_info_frame = ttk.Frame(parent)
        self.camera_info_frame.pack(fill=tk.X, pady=5)
        
        self.update_camera_info_display()
        
        # Botón para aplicar cambios de cámara
        apply_camera_btn = ttk.Button(parent, text="Cambiar Cámara", 
                                     command=self.change_camera)
        apply_camera_btn.pack(fill=tk.X, pady=2)
    
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
        if self.running:
            messagebox.showwarning("Cámara en uso", 
                                 "Detén la detección antes de cambiar de cámara")
            return
        
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
            else:
                messagebox.showerror("Error", "Cámara no disponible")
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Error al procesar selección de cámara")
        
    def create_color_controls(self, parent):
        """Crear controles de configuración de colores"""
        
        # Color para castañas sanas
        sana_frame = ttk.Frame(parent)
        sana_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(sana_frame, text="Castañas Sanas:").pack(side=tk.LEFT)
        self.sana_color_btn = tk.Button(sana_frame, bg=self.colors['sana'], width=10,
                                       command=lambda: self.choose_color('sana'))
        self.sana_color_btn.pack(side=tk.RIGHT)
        
        # Color para castañas contaminadas
        cont_frame = ttk.Frame(parent)
        cont_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(cont_frame, text="Castañas Contaminadas:").pack(side=tk.LEFT)
        self.cont_color_btn = tk.Button(cont_frame, bg=self.colors['contaminada'], width=10,
                                       command=lambda: self.choose_color('contaminada'))
        self.cont_color_btn.pack(side=tk.RIGHT)
        
        # Color para detecciones no confirmadas
        det_frame = ttk.Frame(parent)
        det_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(det_frame, text="Detecciones No Confirmadas:").pack(side=tk.LEFT)
        self.det_color_btn = tk.Button(det_frame, bg=self.colors['detectada'], width=10,
                                      command=lambda: self.choose_color('detectada'))
        self.det_color_btn.pack(side=tk.RIGHT)
        
    def create_rgb_controls(self, parent):
        """Crear controles de parámetros RGB"""
        
        # Umbral de brillo bajo
        brightness_low_frame = ttk.Frame(parent)
        brightness_low_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(brightness_low_frame, text="Brillo Bajo:").pack(side=tk.LEFT)
        self.brightness_low_var = tk.IntVar(value=self.rgb_params['brightness_low'])
        self.brightness_low_scale = ttk.Scale(brightness_low_frame, from_=30, to=120, 
                                            variable=self.brightness_low_var, orient=tk.HORIZONTAL)
        self.brightness_low_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Umbral de brillo medio
        brightness_med_frame = ttk.Frame(parent)
        brightness_med_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(brightness_med_frame, text="Brillo Medio:").pack(side=tk.LEFT)
        self.brightness_med_var = tk.IntVar(value=self.rgb_params['brightness_medium'])
        self.brightness_med_scale = ttk.Scale(brightness_med_frame, from_=60, to=150, 
                                            variable=self.brightness_med_var, orient=tk.HORIZONTAL)
        self.brightness_med_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Umbral de variación
        variation_frame = ttk.Frame(parent)
        variation_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(variation_frame, text="Variación:").pack(side=tk.LEFT)
        self.variation_var = tk.DoubleVar(value=self.rgb_params['variation_threshold'])
        self.variation_scale = ttk.Scale(variation_frame, from_=20.0, to=60.0, 
                                       variable=self.variation_var, orient=tk.HORIZONTAL)
        self.variation_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Umbral de densidad de bordes
        edges_frame = ttk.Frame(parent)
        edges_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(edges_frame, text="Densidad Bordes:").pack(side=tk.LEFT)
        self.edges_var = tk.DoubleVar(value=self.rgb_params['edge_density_threshold'])
        self.edges_scale = ttk.Scale(edges_frame, from_=0.05, to=0.30, 
                                   variable=self.edges_var, orient=tk.HORIZONTAL)
        self.edges_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Botón para aplicar cambios
        apply_btn = ttk.Button(parent, text="Aplicar Cambios", 
                              command=self.apply_rgb_changes)
        apply_btn.pack(fill=tk.X, pady=5)
        
    def create_detections_panel(self, parent):
        """Crear panel de detecciones"""
        
        # Listbox para mostrar detecciones
        self.detections_listbox = tk.Listbox(parent, height=8, font=('Consolas', 10))
        self.detections_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para la lista
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.detections_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detections_listbox.config(yscrollcommand=scrollbar.set)
        
    def create_stats_panel(self, parent):
        """Crear panel de estadísticas"""
        
        # Estadísticas en tiempo real
        self.stats_labels = {}
        
        stats_data = [
            ('Cámara Actual:', 'current_camera'),
            ('Frames Procesados:', 'frames_processed'),
            ('Total Detecciones:', 'total_detections'),
            ('Castañas Sanas:', 'sanas'),
            ('Castañas Contaminadas:', 'contaminadas')
        ]
        
        for label_text, key in stats_data:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            self.stats_labels[key] = ttk.Label(frame, text="0", font=('Arial', 10, 'bold'))
            self.stats_labels[key].pack(side=tk.RIGHT)
        
        # Separador
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Botón para resetear estadísticas
        reset_btn = ttk.Button(parent, text="Resetear Estadísticas", 
                              command=self.reset_stats)
        reset_btn.pack(fill=tk.X, pady=5)
        
    def choose_color(self, color_type):
        """Permitir al usuario elegir un color"""
        color = colorchooser.askcolor(title=f"Elegir color para {color_type}")[1]
        if color:
            self.colors[color_type] = color
            # Actualizar botón de color
            if color_type == 'sana':
                self.sana_color_btn.config(bg=color)
            elif color_type == 'contaminada':
                self.cont_color_btn.config(bg=color)
            elif color_type == 'detectada':
                self.det_color_btn.config(bg=color)
    
    def apply_rgb_changes(self):
        """Aplicar cambios en los parámetros RGB"""
        self.rgb_params = {
            'brightness_low': self.brightness_low_var.get(),
            'brightness_medium': self.brightness_med_var.get(),
            'variation_threshold': self.variation_var.get(),
            'edge_density_threshold': self.edges_var.get(),
            'contamination_threshold': 1  # Fijo por ahora
        }
        messagebox.showinfo("Parámetros", "Parámetros RGB actualizados")
    
    def reset_stats(self):
        """Resetear estadísticas"""
        self.stats = {
            'total_detections': 0,
            'sanas': 0,
            'contaminadas': 0,
            'frames_processed': 0
        }
        self.update_stats_display()
        self.detections_listbox.delete(0, tk.END)
    
    def update_stats_display(self):
        """Actualizar visualización de estadísticas"""
        # Actualizar estadísticas normales
        for key, value in self.stats.items():
            if key in self.stats_labels:
                self.stats_labels[key].config(text=str(value))
        
        # Actualizar información de cámara
        if 'current_camera' in self.stats_labels:
            if self.camera_id in self.camera_info:
                cam_name = self.camera_info[self.camera_id]['name']
                self.stats_labels['current_camera'].config(text=f"{self.camera_id}: {cam_name}")
            else:
                self.stats_labels['current_camera'].config(text=f"Cámara {self.camera_id}")
    
    def toggle_detection(self):
        """Iniciar o detener la detección"""
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """Iniciar detección"""
        try:
            # Cargar modelo YOLO
            self.model = YOLO('core/yolo12n.pt')
            
            # Inicializar cámara con la seleccionada
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                messagebox.showerror("Error", f"No se pudo abrir la cámara {self.camera_id}")
                return
            
            # Configurar resolución si está disponible
            if self.camera_id in self.camera_info:
                cam_info = self.camera_info[self.camera_id]
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_info['width'])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_info['height'])
                self.camera.set(cv2.CAP_PROP_FPS, cam_info['fps'])
            
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Iniciar hilo de detección
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar detección: {str(e)}")
    
    def stop_detection(self):
        """Detener detección"""
        self.running = False
        if self.camera:
            self.camera.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(text="Cámara detenida")
    
    def detection_loop(self):
        """Bucle principal de detección"""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Realizar detección
                results = self.model.predict(frame, conf=0.5)
                
                # Procesar resultados
                detections_text = []
                frame_detections = {
                    'sanas': 0,
                    'contaminadas': 0,
                    'detectadas': 0
                }
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            confidence = box.conf.item()
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Recortar imagen para análisis RGB
                            if x2 > x1 and y2 > y1:
                                crop_img = frame[y1:y2, x1:x1+(x2-x1)]
                                
                                if class_name.lower() in self.chestnut_classes:
                                    # Análisis RGB
                                    quality = analyze_chestnut_quality_dual(crop_img)
                                    
                                    if quality == 'sana':
                                        frame_detections['sanas'] += 1
                                        color = self.colors['sana']
                                        label = f"SANA: {class_name} ({confidence:.2f})"
                                    else:
                                        frame_detections['contaminadas'] += 1
                                        color = self.colors['contaminada']
                                        label = f"CONTAMINADA: {class_name} ({confidence:.2f})"
                                    
                                    # Dibujar rectángulo
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                                self.hex_to_bgr(color), 3)
                                    cv2.putText(frame, label, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                              self.hex_to_bgr(color), 2)
                                    
                                    detections_text.append(f"{label} - {time.strftime('%H:%M:%S')}")
                                else:
                                    frame_detections['detectadas'] += 1
                                    color = self.colors['detectada']
                                    label = f"DETECTADA: {class_name} ({confidence:.2f})"
                                    
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                                self.hex_to_bgr(color), 2)
                                    cv2.putText(frame, label, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                              self.hex_to_bgr(color), 1)
                                    
                                    detections_text.append(f"{label} - {time.strftime('%H:%M:%S')}")
                
                # Actualizar estadísticas
                self.stats['frames_processed'] += 1
                self.stats['sanas'] += frame_detections['sanas']
                self.stats['contaminadas'] += frame_detections['contaminadas']
                self.stats['total_detections'] += sum(frame_detections.values())
                
                # Actualizar interfaz en el hilo principal
                self.root.after(0, self.update_ui, frame, detections_text)
                
            except Exception as e:
                print(f"Error en detección: {e}")
                time.sleep(0.1)
    
    def hex_to_bgr(self, hex_color):
        """Convertir color hexadecimal a BGR para OpenCV"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)  # BGR para OpenCV
    
    def update_ui(self, frame, detections_text):
        """Actualizar interfaz de usuario"""
        try:
            # Actualizar video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil.resize((400, 300)))
            
            self.video_label.config(image=frame_tk, text="")
            self.video_label.image = frame_tk  # Mantener referencia
            
            # Actualizar lista de detecciones
            for detection in detections_text:
                self.detections_listbox.insert(0, detection)
                # Mantener solo las últimas 50 detecciones
                if self.detections_listbox.size() > 50:
                    self.detections_listbox.delete(50, tk.END)
            
            # Actualizar estadísticas
            self.update_stats_display()
            
        except Exception as e:
            print(f"Error actualizando UI: {e}")


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

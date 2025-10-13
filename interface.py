#!/usr/bin/env python3
"""
Interfaz gráfica para CastañaSerial
Visualización de detecciones y configuración de parámetros
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import json
import os
from ultralytics import YOLO
from functions.analysys import analyze_object_quality_with_logging
from typing import Dict, Any, Optional


class CastañaSerialInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Calidad - Interfaz de Control")
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
        
        # Configuración de clases (se carga desde interface_config.json)
        self.chestnut_classes = []  # Se cargará desde JSON
        self.detected_objects = {}
        # Memoria de áreas contaminadas: lista de tuplas (cx, cy, timestamp)
        self.contaminated_memory = []
        
        # Parámetros RGB manejados automáticamente por functions/analysys.py
        
        # Estadísticas
        self.stats = {
            'total_detections': 0,
            'sanas': 0,
            'contaminadas': 0,
            'frames_processed': 0,
            'classes_ignored': 0
        }
        
        self.load_camera_config()
        self.load_interface_config()
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
        video_frame = ttk.LabelFrame(top_frame, text="Video en Tiempo Real", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(video_frame, text="Cámara no iniciada", 
                                   font=('Arial', 12), foreground='white', 
                                   background=self.colors['fondo'])
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
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
    
    def load_interface_config(self):
        """Cargar configuración de interfaz desde JSON"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                interface_config = json.load(f)
                
                # Cargar colores desde configuración
                if "colors" in interface_config:
                    self.colors.update(interface_config["colors"])
                
                # Cargar clases de detección desde configuración
                if "detection_classes" in interface_config:
                    self.chestnut_classes = interface_config["detection_classes"]
                    print(f"✅ Clases de detección cargadas desde JSON: {self.chestnut_classes}")
                    
                    # Verificar que no esté vacío
                    if not self.chestnut_classes:
                        print("⚠️  Advertencia: No hay clases de detección en interface_config.json")
                        self.chestnut_classes = ['apple', 'orange']  # Fallback
                        print(f"🔄 Usando clases por defecto: {self.chestnut_classes}")
                else:
                    print("❌ No se encontraron clases de detección en interface_config.json")
                    self.chestnut_classes = ['apple', 'orange']  # Fallback
                    print(f"🔄 Usando clases por defecto: {self.chestnut_classes}")
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error cargando interface_config.json: {e}")
            print("🔄 Usando configuración por defecto")
            # Fallback para clases de detección
            if not self.chestnut_classes:
                self.chestnut_classes = ['apple', 'orange']
                print(f"🔄 Clases por defecto: {self.chestnut_classes}")
    
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
            ('Objetos Sanos:', 'sanas'),
            ('Objetos Contaminados:', 'contaminadas')
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
            
            # Configurar clases específicas
            self.configure_yolo_classes()
            
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
            
            # Mostrar información de las clases cargadas
            print(f"🎯 Sistema iniciado con clases: {self.chestnut_classes}")
            print(f"📋 Clases YOLO disponibles: {list(self.model.names.values())}")
            print(f"🔍 Solo procesando: {self.chestnut_classes}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar detección: {str(e)}")
    
    def configure_yolo_classes(self):
        """Configurar YOLO para detectar solo las clases deseadas"""
        if not self.model:
            return
            
        try:
            # Obtener todas las clases disponibles en YOLO
            all_classes = list(self.model.names.values())
            print(f"📋 Clases YOLO disponibles: {all_classes}")
            
            # Encontrar los IDs de las clases que queremos detectar
            desired_class_ids = []
            for desired_class in self.chestnut_classes:
                for class_id, class_name in self.model.names.items():
                    if class_name.lower() == desired_class.lower():
                        desired_class_ids.append(class_id)
                        print(f"✅ Clase encontrada: {class_name} (ID: {class_id})")
                        break
                else:
                    print(f"⚠️  Clase no encontrada en YOLO: {desired_class}")
            
            if desired_class_ids:
                print(f"🎯 Configurando YOLO para detectar solo: {[self.model.names[i] for i in desired_class_ids]}")
                # Nota: YOLO no permite filtrar clases directamente en la carga del modelo
                # El filtrado se hace en el procesamiento de resultados
            else:
                print("⚠️  No se encontraron clases válidas en YOLO")
                
        except Exception as e:
            print(f"❌ Error configurando clases YOLO: {e}")
    
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
                
                # Realizar detección con filtrado optimizado
                results = self.model.predict(frame, conf=0.5)
                
                # Procesar resultados
                detections_text = []
                frame_detections = {
                    'sanas': 0,
                    'contaminadas': 0,
                    'detectadas': 0
                }
                
                # Lista para evitar duplicados en la misma área
                processed_areas = []
                
                # Purga de memoria de contaminados (expira a los 3s)
                now_ts = time.time()
                self.contaminated_memory = [m for m in self.contaminated_memory if now_ts - m[2] < 3.0]
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            confidence = box.conf.item()
                            
                            # FILTRAR: Solo procesar clases configuradas
                            if class_name.lower() not in [cls.lower() for cls in self.chestnut_classes]:
                                # Contar clases ignoradas
                                self.stats['classes_ignored'] += 1
                                
                                # Log de clases ignoradas (solo ocasionalmente para no spamear)
                                if confidence > 0.7 and self.stats['frames_processed'] % 30 == 0:  # Cada 30 frames
                                    print(f"🚫 Clase ignorada: {class_name} (confianza: {confidence:.2f}) - No está en configuración")
                                continue  # Saltar esta detección
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Verificar si ya procesamos un objeto en esta área (evitar duplicados)
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            area_size = (x2 - x1) * (y2 - y1)
                            
                            # Buscar si hay otra detección cerca (dentro de 50 píxeles)
                            is_duplicate = False
                            for prev_center_x, prev_center_y, prev_area_size in processed_areas:
                                distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
                                if distance < 50:  # Si está muy cerca
                                    # Mantener la detección con mayor confianza
                                    if confidence <= 0.7:  # Si la confianza es baja, saltar
                                        is_duplicate = True
                                        break
                            
                            if is_duplicate:
                                continue
                            
                            # Agregar esta área a las procesadas
                            processed_areas.append((center_x, center_y, area_size))
                            
                            # Recortar imagen para análisis RGB
                            if x2 > x1 and y2 > y1:
                                crop_img = frame[y1:y2, x1:x1+(x2-x1)]
                                
                                # Si área coincide con memoria de contaminados, forzar como contaminada
                                forced_contaminated = False
                                for mem_cx, mem_cy, mem_ts in self.contaminated_memory:
                                    dist_mem = ((center_x - mem_cx) ** 2 + (center_y - mem_cy) ** 2) ** 0.5
                                    if dist_mem < 60:  # radio de recuerdo
                                        forced_contaminated = True
                                        break
                                
                                if forced_contaminated:
                                    quality = 'contaminada'
                                else:
                                    # Análisis RGB: forzar reglas de manzana para frutas
                                    analysis_class = class_name.lower()
                                    if analysis_class in ['orange']:
                                        analysis_class = 'apple'
                                    quality = analyze_object_quality_with_logging(crop_img, analysis_class)
                                
                                if quality == 'sana':
                                    frame_detections['sanas'] += 1
                                    color = self.colors['sana']
                                    label = f"SANA ({confidence:.2f})"
                                elif quality == 'contaminada':
                                    frame_detections['contaminadas'] += 1
                                    color = self.colors['contaminada']
                                    label = f"CONTAMINADA ({confidence:.2f})"
                                    # Guardar/actualizar memoria de contaminados
                                    updated = False
                                    for i, (mem_cx, mem_cy, mem_ts) in enumerate(self.contaminated_memory):
                                        dist_mem = ((center_x - mem_cx) ** 2 + (center_y - mem_cy) ** 2) ** 0.5
                                        if dist_mem < 60:
                                            self.contaminated_memory[i] = (center_x, center_y, now_ts)
                                            updated = True
                                            break
                                    if not updated:
                                        self.contaminated_memory.append((center_x, center_y, now_ts))
                                else:
                                    frame_detections['detectadas'] += 1
                                    color = self.colors['detectada']
                                    label = f"DETECTADA ({confidence:.2f})"
                                
                                # Dibujar rectángulo y texto para todas las detecciones
                                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                            self.hex_to_bgr(color), 3)
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                          self.hex_to_bgr(color), 2)
                                
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
            # Obtener dimensiones del label de video
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            # Si el label aún no tiene dimensiones, usar valores por defecto
            if label_width <= 1 or label_height <= 1:
                label_width = 600  # Ancho por defecto
                label_height = 400  # Alto por defecto
            
            # Calcular dimensiones manteniendo la proporción del video
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            # Calcular nuevas dimensiones que se ajusten al espacio disponible
            if label_width / label_height > aspect_ratio:
                # El espacio es más ancho que el video, ajustar por altura
                new_height = label_height - 10  # Margen mínimo
                new_width = int(new_height * aspect_ratio)
            else:
                # El espacio es más alto que el video, ajustar por ancho
                new_width = label_width - 10  # Margen mínimo
                new_height = int(new_width / aspect_ratio)
            
            # Asegurar dimensiones mínimas
            new_width = max(new_width, 200)
            new_height = max(new_height, 150)
            
            # Actualizar video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil.resize((new_width, new_height)))
            
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

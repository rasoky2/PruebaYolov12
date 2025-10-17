"""
Ventana principal que coordina todos los widgets
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

# Importar widgets
from .camera_panel import CameraPanel
from .arduino_panel import ArduinoPanel
from .detection_panel import DetectionPanel
from .stats_panel import StatsPanel
from .controls_panel import ControlsPanel
from .log_window import LogWindow, setup_global_logging


class CastañaSerialInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Calidad - Interfaz de Control")
        self.root.geometry("1200x800")
        
        # Variables del sistema
        self.model = None
        self.camera = None
        self.running = False
        self.detection_thread = None
        self.frame_count = 0
        self.camera_restart_interval = 1000
        self.last_frame_time = time.time()
        self.camera_timeout_threshold = 2.0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_model_name = "YOLO12x"
        
        # Variables de estabilidad Arduino
        self.last_detection_contaminated = False
        self.state_stable_since = time.time()
        self.arduino_block_until = 0.0
        self.last_line_trigger_time = 0.0
        self.last_line_side = None
        
        # Configuración de colores
        self.colors = {
            'sana': '#00FF00',
            'contaminada': '#FF0000',
            'detectada': '#FFFF00',
            'fondo': '#2C3E50'
        }
        
        # Configuración de clases
        self.chestnut_classes = []
        self.detected_objects = {}
        self.contaminated_memory = []
        
        self.load_interface_config()
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

        # Vincular eventos de mouse para dibujar línea
        self.video_label.bind('<Button-1>', self.on_video_click)
        self.video_label.bind('<Motion>', self.on_video_motion)
        
        # Panel derecho - Controles
        controls_frame = ttk.LabelFrame(top_frame, text="Controles", padding=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Crear widgets especializados
        self.camera_panel = CameraPanel(controls_frame, self.on_camera_changed)
        self.arduino_panel = ArduinoPanel(controls_frame)
        self.controls_panel = ControlsPanel(controls_frame)
        
        # Crear ventana de logs
        self.log_window = LogWindow(controls_frame)
        self.log_window.set_root_reference(self.root)
        setup_global_logging(self.log_window)
        
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
        
        # Crear paneles de detecciones y estadísticas
        self.detection_panel = DetectionPanel(bottom_frame)
        self.stats_panel = StatsPanel(bottom_frame)
    
    def load_interface_config(self):
        """Cargar configuración de interfaz desde JSON"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interface_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                interface_config = json.load(f)
                
                # Cargar colores desde configuración
                if "colors" in interface_config:
                    self.colors.update(interface_config["colors"])
                
                # Cargar clases de detección desde configuración
                if "detection_classes" in interface_config:
                    self.chestnut_classes = interface_config["detection_classes"]
                    print(f"✅ Clases de detección cargadas desde JSON: {self.chestnut_classes}")
                    
                    if not self.chestnut_classes:
                        print("⚠️  Advertencia: No hay clases de detección en interface_config.json")
                        self.chestnut_classes = ['apple', 'orange']
                        print(f"🔄 Usando clases por defecto: {self.chestnut_classes}")
                else:
                    print("❌ No se encontraron clases de detección en interface_config.json")
                    self.chestnut_classes = ['apple', 'orange']
                    print(f"🔄 Usando clases por defecto: {self.chestnut_classes}")
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error cargando interface_config.json: {e}")
            print("🔄 Usando configuración por defecto")
            if not self.chestnut_classes:
                self.chestnut_classes = ['apple', 'orange']
                print(f"🔄 Clases por defecto: {self.chestnut_classes}")
    
    def on_camera_changed(self, camera_id: int, camera_info: dict):
        """Callback cuando cambia la cámara"""
        self.stats_panel.update_camera_info(camera_id, camera_info['name'])
    
    def on_video_click(self, event):
        """Manejar clicks en el video para dibujar línea"""
        self.controls_panel.on_video_click(event)
    
    def on_video_motion(self, event):
        """Manejar movimiento del mouse para mostrar línea temporal"""
        self.controls_panel.on_video_motion(event)
    
    def toggle_detection(self):
        """Iniciar o detener la detección"""
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """Iniciar detección"""
        try:
            # Cargar modelo YOLO seleccionado
            selected_model = self.controls_panel.get_model()
            model_path = f'core/{selected_model.lower()}.pt'
            self.current_model_name = selected_model
            
            self.log_window.log(f"🔄 Cargando modelo: {selected_model}", "INFO")
            self.model = YOLO(model_path)
            
            # Configurar clases específicas
            self.configure_yolo_classes()
            
            # Obtener información de cámara
            camera_id = self.camera_panel.get_camera_id()
            camera_info = self.camera_panel.get_camera_info()
            
            # Inicializar cámara
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                messagebox.showerror("Error", f"No se pudo abrir la cámara {camera_id}")
                return
            
            # Configurar cámara
            self.configure_camera(camera_info)
            
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Iniciar hilo de detección
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            # Mostrar información de aceleración
            self.update_acceleration_status()
            self.stats_panel.update_model_status(selected_model)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar detección: {str(e)}")
    
    def configure_camera(self, camera_info: dict):
        """Configurar parámetros de cámara"""
        if camera_info:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_info['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_info['height'])
            
            # Configuraciones de performance
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            
            # Intentar forzar 720p @ 60 FPS
            try:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 60)
            except Exception:
                pass
    
    def update_acceleration_status(self):
        """Actualizar estado de aceleración"""
        try:
            import torch
            using_gpu = torch.cuda.is_available()
            accel_text = f"GPU FP16" if using_gpu else "CPU"
        except Exception:
            accel_text = "CPU"
        self.stats_panel.update_acceleration_status(accel_text)
    
    def configure_yolo_classes(self):
        """Configurar YOLO para detectar solo las clases deseadas"""
        if not self.model:
            return
            
        try:
            all_classes = list(self.model.names.values())
            print(f"📋 Clases YOLO disponibles: {all_classes}")
            
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
            else:
                print("⚠️  No se encontraron clases válidas en YOLO")
                
        except Exception as e:
            print(f"❌ Error configurando clases YOLO: {e}")
    
    def stop_detection(self):
        """Detener detección"""
        self.running = False
        if self.camera:
            self.camera.release()
        
        # Limpiar memoria y resetear contadores
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.contaminated_memory.clear()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(text="Cámara detenida")
    
    def detection_loop(self):
        """Bucle principal de detección"""
        frame_index = 0
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    current_time = time.time()
                    if current_time - self.last_frame_time > self.camera_timeout_threshold:
                        self.log_window.log("⚠️  Cámara colgada detectada, reiniciando...", "WARNING")
                        self.restart_camera()
                        continue
                    continue

                self.last_frame_time = time.time()
                self.frame_count += 1

                # Guardar dimensiones del frame
                frame_height, frame_width = frame.shape[:2]

                # Reinicio automático deshabilitado - solo reiniciar en caso de error
                # (el reinicio por timeout ya está manejado arriba)
                
                # Realizar detección
                results = self.run_detection(frame)
                
                # Procesar resultados
                detections_text, frame_detections, crossed_line_contaminated = self.process_detections(
                    results, frame, frame_width, frame_height
                )
                
                # Actualizar estadísticas
                self.update_detection_stats(frame_detections)
                
                # Enviar señales a Arduino
                self.handle_arduino_signals(crossed_line_contaminated, frame_detections)
                
                # Actualizar interfaz
                frame_index += 1
                render_every = max(1, self.controls_panel.get_render_every())
                if frame_index % render_every == 0:
                    self.root.after(0, self.update_ui, frame, detections_text)
                
            except Exception as e:
                self.log_window.log(f"Error en detección: {e}", "ERROR")
                try:
                    self.restart_camera()
                except Exception as restart_error:
                    self.log_window.log(f"Error reiniciando cámara: {restart_error}", "ERROR")
                time.sleep(0.1)
    
    def run_detection(self, frame):
        """Ejecutar detección YOLO"""
        # Configuración por modelo
        if self.current_model_name == "YOLO12l":
            predict_kwargs = {'conf': 0.4, 'imgsz': 640, 'device': 0}
        elif self.current_model_name == "YOLO12x":
            predict_kwargs = {'conf': 0.35, 'imgsz': 640, 'device': 0}
        else:  # YOLO12n
            predict_kwargs = {'conf': 0.5, 'imgsz': 640, 'device': 0}
        
        try:
            import torch
            if torch.cuda.is_available():
                predict_kwargs['half'] = True
        except Exception:
            pass
        
        return self.model.predict(frame, **predict_kwargs)
    
    def process_detections(self, results, frame, frame_width, frame_height):
        """Procesar resultados de detección"""
        detections_text = []
        frame_detections = {'sanas': 0, 'contaminadas': 0, 'detectadas': 0}
        processed_areas = []
        
        # Contador de debug global
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        # Purga de memoria de contaminados
        now_ts = time.time()
        self.contaminated_memory = [m for m in self.contaminated_memory if now_ts - m[2] < 3.0]
        
        crossed_line_contaminated = False
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = box.conf.item()
                    
                    # Filtrar clases
                    if class_name.lower() not in [cls.lower() for cls in self.chestnut_classes]:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Evitar duplicados
                    is_duplicate = False
                    for prev_center_x, prev_center_y, prev_area_size in processed_areas:
                        distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
                        if distance < 50 and confidence <= 0.7:
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue
                    
                    processed_areas.append((center_x, center_y, (x2 - x1) * (y2 - y1)))
                    
                    # Análisis de calidad
                    quality = self.analyze_object_quality(frame, x1, y1, x2, y2, center_x, center_y, class_name)
                    
                    # Contar por calidad
                    if quality == 'sana':
                        frame_detections['sanas'] += 1
                        color = self.colors['sana']
                        label = f"SANA ({confidence:.2f})"
                    elif quality == 'contaminada':
                        frame_detections['contaminadas'] += 1
                        color = self.colors['contaminada']
                        label = f"CONTAMINADA ({confidence:.2f})"
                        # Actualizar memoria de contaminados
                        self.update_contaminated_memory(center_x, center_y, now_ts)
                    else:
                        frame_detections['detectadas'] += 1
                        color = self.colors['detectada']
                        label = f"DETECTADA ({confidence:.2f})"

                    # Detectar cruce de línea
                    if self.controls_panel.has_line():
                        line_start, line_end = self.controls_panel.get_line_coordinates()
                        if self._debug_counter % 60 == 0:  # Log cada 60 frames para no saturar
                            print(f"📏 LÍNEA DIBUJADA: ({line_start[0]:.2f}, {line_start[1]:.2f}) -> ({line_end[0]:.2f}, {line_end[1]:.2f})")
                        is_near = self.is_near_line(center_x, center_y, line_start, line_end, frame_width, frame_height)
                        if is_near:
                            print(f"🔍 OBJETO CERCA DE LÍNEA: {label} en ({center_x}, {center_y})")
                            if quality == 'contaminada':
                                crossed_line_contaminated = True
                                print(f"🎯 CONTAMINADA CRUZANDO LÍNEA: {label}")
                    
                    # Dibujar rectángulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.hex_to_bgr(color), 3)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hex_to_bgr(color), 2)
                    
                    detections_text.append(f"{label} - {time.strftime('%H:%M:%S')}")
        
        return detections_text, frame_detections, crossed_line_contaminated
    
    def analyze_object_quality(self, frame, x1, y1, x2, y2, center_x, center_y, class_name):
        """Analizar calidad del objeto"""
        # Verificar memoria de contaminados
        for mem_cx, mem_cy, mem_ts in self.contaminated_memory:
            dist_mem = ((center_x - mem_cx) ** 2 + (center_y - mem_cy) ** 2) ** 0.5
            if dist_mem < 60:
                return 'contaminada'
        
        # Análisis RGB
        if x2 > x1 and y2 > y1:
            crop_img = frame[y1:y2, x1:x1+(x2-x1)]
            analysis_class = class_name.lower()
            if analysis_class in ['orange']:
                analysis_class = 'apple'
            return analyze_object_quality_with_logging(crop_img, analysis_class)
        
        return 'detectada'
    
    def update_contaminated_memory(self, center_x, center_y, now_ts):
        """Actualizar memoria de contaminados"""
        updated = False
        for i, (mem_cx, mem_cy, mem_ts) in enumerate(self.contaminated_memory):
            dist_mem = ((center_x - mem_cx) ** 2 + (center_y - mem_cy) ** 2) ** 0.5
            if dist_mem < 60:
                self.contaminated_memory[i] = (center_x, center_y, now_ts)
                updated = True
                break
        if not updated:
            self.contaminated_memory.append((center_x, center_y, now_ts))
    
    def is_near_line(self, center_x, center_y, line_start, line_end, frame_width, frame_height):
        """Verificar si el objeto está cerca de la línea"""
        try:
            # Convertir coordenadas normalizadas a absolutas
            lx1 = int(line_start[0] * frame_width)
            ly1 = int(line_start[1] * frame_height)
            lx2 = int(line_end[0] * frame_width)
            ly2 = int(line_end[1] * frame_height)
            
            # Calcular distancia del punto a la línea
            num = abs((lx2 - lx1) * (ly1 - center_y) - (lx1 - center_x) * (ly2 - ly1))
            den = ((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5
            dist = num / den if den != 0 else 1e9
            
            # Log de debug cada 30 frames para no saturar
            if self._debug_counter % 30 == 0:
                print(f"🔍 DISTANCIA A LÍNEA: objeto({center_x}, {center_y}) -> línea({lx1}, {ly1})-({lx2}, {ly2}) = {dist:.1f}px")
            
            detection_radius = self.controls_panel.get_line_detection_radius()
            is_near = dist < detection_radius
            if is_near and self._debug_counter % 30 == 0:
                print(f"✅ OBJETO CERCA DE LÍNEA: {dist:.1f}px < {detection_radius}px")
            
            return is_near
        except Exception as e:
            self.log_window.log(f"❌ Error en is_near_line: {e}", "ERROR")
            return False
    
    def update_detection_stats(self, frame_detections):
        """Actualizar estadísticas de detección"""
        self.stats_panel.update_stats(
            frames_processed=self.stats_panel.get_stats()['frames_processed'] + 1,
            sanas=self.stats_panel.get_stats()['sanas'] + frame_detections['sanas'],
            contaminadas=self.stats_panel.get_stats()['contaminadas'] + frame_detections['contaminadas'],
            total_detections=self.stats_panel.get_stats()['total_detections'] + sum(frame_detections.values())
        )
        
        # Calcular FPS
        self.fps_frame_count += 1
        if self.fps_frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                fps = self.fps_frame_count / elapsed
                self.stats_panel.update_fps(fps)
                self.fps_start_time = current_time
                self.fps_frame_count = 0
    
    def handle_arduino_signals(self, crossed_line_contaminated, frame_detections):
        """Manejar señales de Arduino"""
        try:
            if not self.arduino_panel.is_connected():
                return
            
            # Determinar estado deseado
            if self.controls_panel.has_line():
                desired_state_contaminated = crossed_line_contaminated
            else:
                desired_state_contaminated = False
            
            now_time = time.time()
            
            # Log del estado actual
            if crossed_line_contaminated:
                print(f"🔴 CRUCE DETECTADO: crossed_line_contaminated={crossed_line_contaminated}, desired_state={desired_state_contaminated}")
            
            # Debounce logic
            if desired_state_contaminated != self.last_detection_contaminated:
                self.state_stable_since = now_time
                self.last_detection_contaminated = desired_state_contaminated
                print(f"🔄 CAMBIO DE ESTADO: {self.last_detection_contaminated} -> {desired_state_contaminated}")
            
            debounce_s = self.controls_panel.get_arduino_debounce()
            cooldown_s = self.controls_panel.get_arduino_cooldown()
            
            time_stable = now_time - self.state_stable_since
            print(f"⏱️ TIEMPO ESTABLE: {time_stable:.1f}s (necesario: {debounce_s}s), COOLDOWN: {now_time >= self.arduino_block_until}")
            
            if (now_time - self.state_stable_since) >= debounce_s and now_time >= self.arduino_block_until:
                if desired_state_contaminated:
                    print("✅ ENVIANDO CONTAMINADO AL ARDUINO")
                    self.arduino_panel.send_signal('CONTAMINADO', delay_prevention=0.3)
                    self.arduino_block_until = now_time + cooldown_s
                elif self.controls_panel.has_line():
                    print("✅ ENVIANDO SANO AL ARDUINO")
                    self.arduino_panel.send_signal('SANO', delay_prevention=0.1)
                    self.arduino_block_until = now_time + cooldown_s
        except Exception as e:
            print(f"❌ Error en handle_arduino_signals: {e}")
    
    def update_ui(self, frame, detections_text):
        """Actualizar interfaz de usuario"""
        try:
            # Actualizar video
            self.update_video_display(frame)
            
            # Actualizar detecciones
            self.detection_panel.add_detections(detections_text)
            
            # Actualizar estadísticas de Arduino
            if self.arduino_panel.is_connected():
                manager = self.arduino_panel.get_manager()
                self.stats_panel.update_arduino_status(
                    True, 
                    manager.port or 'N/D', 
                    manager.last_signal or '-'
                )
            else:
                self.stats_panel.update_arduino_status(False)
            
        except Exception as e:
            print(f"Error actualizando UI: {e}")
    
    def update_video_display(self, frame):
        """Actualizar display de video"""
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width <= 1 or label_height <= 1:
            label_width = 600
            label_height = 400
        
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        if label_width / label_height > aspect_ratio:
            new_height = label_height - 10
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = label_width - 10
            new_height = int(new_width / aspect_ratio)
        
        new_width = max(new_width, 200)
        new_height = max(new_height, 150)
        
        # Convertir y redimensionar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Dibujar líneas si existen
        if self.controls_panel.has_line() or self.controls_panel.is_line_draw_mode():
            from PIL import ImageDraw
            draw = ImageDraw.Draw(frame_pil)
            
            # Dibujar línea final (si existe)
            if self.controls_panel.has_line():
                line_start, line_end = self.controls_panel.get_line_coordinates()
                x1 = int(line_start[0] * frame_width)
                y1 = int(line_start[1] * frame_height)
                x2 = int(line_end[0] * frame_width)
                y2 = int(line_end[1] * frame_height)
                draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=3)  # Línea roja sólida
            
            # Dibujar línea temporal (mientras se dibuja)
            elif self.controls_panel.is_line_draw_mode() and self.controls_panel.is_drawing_line:
                temp_start, temp_end = self.controls_panel.get_temp_line_coordinates()
                if temp_start and temp_end:
                    x1 = int(temp_start[0] * frame_width)
                    y1 = int(temp_start[1] * frame_height)
                    x2 = int(temp_end[0] * frame_width)
                    y2 = int(temp_end[1] * frame_height)
                    draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 0), width=2)  # Línea amarilla punteada

        frame_tk = ImageTk.PhotoImage(frame_pil.resize((new_width, new_height)))
        
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk
    
    def restart_camera(self):
        """Reiniciar la cámara"""
        try:
            if self.camera:
                self.camera.release()
            time.sleep(0.5)
            
            camera_id = self.camera_panel.get_camera_id()
            camera_info = self.camera_panel.get_camera_info()
            
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                print("❌ Error: No se pudo reabrir la cámara")
                return False
            
            self.configure_camera(camera_info)
            self.log_window.log("✅ Cámara reiniciada exitosamente", "INFO")
            return True
        except Exception as e:
            self.log_window.log(f"❌ Error reiniciando cámara: {e}", "ERROR")
            return False
    
    def hex_to_bgr(self, hex_color):
        """Convertir color hexadecimal a BGR para OpenCV"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

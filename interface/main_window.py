"""
Ventana principal que coordina todos los widgets - PyQt6 Version
"""

import sys
import time
import cv2
import json
import logging
import threading
import uuid
import numpy as np
import os
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont

from ultralytics import YOLO

import qtawesome as qta
from .styles import STYLES

from .camera_panel import CameraPanel
from .controls_panel import ControlsPanel
from .log_window import LogWindow, setup_global_logging


class SpatialGrid:
    """Grid espacial optimizado para detección de duplicados O(1)"""
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.grid = {} 
        self.max_detections_per_cell = 5 

    def add_detection(self, center_x, center_y, confidence, area):
        grid_x, grid_y = center_x // self.grid_size, center_y // self.grid_size
        if (grid_x, grid_y) not in self.grid:
            self.grid[(grid_x, grid_y)] = []

        if len(self.grid[(grid_x, grid_y)]) >= self.max_detections_per_cell:
            self.grid[(grid_x, grid_y)].sort(key=lambda x: x[2], reverse=True)
            self.grid[(grid_x, grid_y)] = self.grid[(grid_x, grid_y)][:self.max_detections_per_cell]

        self.grid[(grid_x, grid_y)].append((center_x, center_y, confidence, area))

    def is_duplicate(self, center_x, center_y, confidence, min_distance=50, min_confidence=0.7):
        grid_x, grid_y = center_x // self.grid_size, center_y // self.grid_size
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_grid = (grid_x + dx, grid_y + dy)
                if check_grid in self.grid:
                    for prev_center_x, prev_center_y, _prev_confidence, _prev_area in self.grid[check_grid]:
                        distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
                        if distance < min_distance and confidence <= min_confidence:
                            return True
        return False

    def clear(self):
        self.grid.clear()


class PerformanceMonitor:
    def __init__(self):
        self.fps_history = []
        self.memory_history = []
        self.max_history = 50 
        self.last_cleanup = time.time()
        self.cleanup_interval = 30.0

    def update_fps(self, fps):
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)

    def update_memory(self, memory_mb):
        self.memory_history.append(memory_mb)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)

    def should_cleanup(self):
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            self.last_cleanup = now
            return True
        return False


class CastañaSerialInterface(QMainWindow):
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.frame_ready.connect(self.set_video_frame)
        self.setWindowTitle("Detector de Calidad - Interfaz de Control")
        self.resize(1200, 800)
        
        # Aplicar tema Shadcn Light
        self.setStyleSheet(STYLES)

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

        # Variables de gestión de memoria
        self.memory_cleanup_interval = 25
        self.frame_count_since_cleanup = 0
        self.max_contaminated_memory = 5
        self.max_detection_history = 10
        self.performance_monitor = PerformanceMonitor()

        self.stats = {
            "frames_processed": 0,
            "sanas": 0,
            "contaminadas": 0,
            "total_detections": 0,
            "fps": 0.0,
        }

        # Colores para dibujado (CV2 usa BGR, aquí definimos para conversión)
        self.colors = {
            "sana": "#22c55e",       # Green 500
            "contaminada": "#ef4444", # Red 500
            "detectada": "#eab308",   # Yellow 500
            "fondo": "#ffffff",
        }
        self.quality_confidence_threshold = 0.8
        self.quality_area_threshold = 5000
        self.quality_display_map = {
            "sana": {"count_key": "sanas", "label": "SANA", "color_key": "sana"},
            "contaminada": {"count_key": "contaminadas", "label": "CONTAMINADA", "color_key": "contaminada"},
        }

        self.detections_db = None # DetectionDatabase disabled
        self.detection_session_id = None
        self.class_translations = {}

        self.chestnut_classes = []
        self.contaminated_memory = []
        self.spatial_grid = SpatialGrid(grid_size=50)

        # UI Initialization
        self.init_ui()
        self.load_interface_config()
        
        # Timer for UI updates if needed outside of thread
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_ui_update)
        self.update_timer.start(100) # 100ms

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 8, 16, 16)
        main_layout.setSpacing(8)

        # Top Section (Video + Controls)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(1)
        
        # Left: Video Frame
        video_container = QFrame()
        video_container.setObjectName("Card")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0,0,0,0)
        
        self.video_label = QLabel("Cámara no iniciada")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #fafafa; color: #a1a1aa; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 480)
        
        video_layout.addWidget(self.video_label)
        top_splitter.addWidget(video_container)
        
        # Right: Controls
        controls_scroll = QWidget() # Wrapper if we needed scroll, but QFrame is fine
        controls_container = QFrame()
        controls_container.setObjectName("Card")
        controls_container.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(12)
        
        # Panels
        self.camera_panel = CameraPanel(self, self.on_camera_changed)
        controls_layout.addWidget(self.camera_panel)
        
        self.controls_panel = ControlsPanel(self)
        controls_layout.addWidget(self.controls_panel)
        
        # Logs System (Internal)
        self.log_window = LogWindow()
        setup_global_logging(self.log_window)
        

        
        # Detection Controls
        controls_layout.addStretch() # Spacer
        
        self.btn_toggle = QPushButton(" Iniciar Detección")
        self.btn_toggle.setIcon(qta.icon('fa5s.play', color='#fafafa'))
        self.btn_toggle.setObjectName("Primary")
        self.btn_toggle.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.btn_toggle)
        
        top_splitter.addWidget(controls_container)
        main_layout.addWidget(top_splitter)

        # Auto-start detection after UI is ready
        QTimer.singleShot(1000, self.start_detection)

    def load_interface_config(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interface_config.json")
            with open(config_path, encoding="utf-8") as f:
                interface_config = json.load(f)
                if "colors" in interface_config:
                    self.colors.update(interface_config["colors"])
                if "detection_classes" in interface_config:
                    self.chestnut_classes = interface_config["detection_classes"]
        except Exception:
             self.chestnut_classes = []

    def on_camera_changed(self, camera_id: int, camera_info: dict):
        self.log_window.log(f"Cámara cambiada a ID {camera_id}: {camera_info.get('name')}")
        if self.running:
            self.stop_detection()
            # Slight delay to ensure release
            QTimer.singleShot(500, self.start_detection)



    def toggle_detection(self):
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        try:
            fruit = self.controls_panel.get_fruit()
            
            # Determinar ruta del modelo (Siembre usar el entrenado para la fruta)
            fruit_dir = os.path.join("core", fruit)
            # Buscar cualquier .pt en el directorio de la fruta
            pts = [f for f in os.listdir(fruit_dir) if f.endswith('.pt')]
            if not pts:
                raise FileNotFoundError(f"No se encontró modelo .pt en {fruit_dir}")
            
            # Priorizar 'best.pt' si existe, si no el primero que encuentre
            best_pt = [p for p in pts if 'best' in p.lower()]
            model_file = best_pt[0] if best_pt else pts[0]
            model_path = os.path.join(fruit_dir, model_file)
            
            # Cargar traducciones de clases
            json_path = os.path.join(fruit_dir, "classes.json")
            if os.path.exists(json_path):
                with open(json_path, encoding="utf-8") as f:
                    self.class_translations = json.load(f)
            else:
                self.class_translations = {}

            self.current_model_name = os.path.basename(model_path)
            
            self.log_window.log(f"🔄 Cargando sistema experto para: {fruit.upper()}", "INFO")
            self.model = YOLO(model_path)
            
            # Configure classes (YOLO names)
            self.configure_yolo_classes()
            
            camera_id = self.camera_panel.get_camera_id()
            camera_info = self.camera_panel.get_camera_info()
            
            # Force standard backend first as DSHOW caused instability on this system
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                 # Fallback to DSHOW if standard fails, just in case
                 self.log_window.log(f"⚠️ Fallo stándar, intentando DSHOW para ID {camera_id}...", "WARNING")
                 self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                self.log_window.log(f"❌ No se pudo abrir cámara {camera_id}", "ERROR")
                return
            
            self.configure_camera(camera_info)
            self.running = True
            
            self.btn_toggle.setText(" Detener Detección")
            self.btn_toggle.setIcon(qta.icon('fa5s.stop', color='#fafafa'))
            self.btn_toggle.setObjectName("Destructive")
            self.btn_toggle.setStyleSheet("background-color: #ef4444; color: white;") # Force update
            
            self.detection_session_id = uuid.uuid4().hex
            
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.log_window.log("🚀 Detección iniciada", "INFO")
            
        except Exception as e:
            self.log_window.log(f"Error iniciando detección: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def configure_camera(self, camera_info):
        try:
             # Basic config, DSHOW handles resolution better usually
             self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
             self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
             self.camera.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
             pass

    def configure_yolo_classes(self):
        if not self.model: return
        self.chestnut_class_ids = []
        if not self.chestnut_classes:
            # Si no hay filtro, usar todas las clases del modelo
            self.chestnut_class_ids = list(self.model.names.keys())
            return

        for desired_class in self.chestnut_classes:
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == desired_class.lower():
                    self.chestnut_class_ids.append(class_id)
                    break

    def stop_detection(self):
        self.running = False
        if self.camera:
            self.camera.release()
            
        # Update Button State
        self.btn_toggle.setText(" Iniciar Detección")
        self.btn_toggle.setIcon(qta.icon('fa5s.play', color='#fafafa'))
        self.btn_toggle.setObjectName("Primary") 
        self.btn_toggle.setStyleSheet("background-color: #18181b; color: white;")
        
        self.video_label.setText("Cámara detenida")
        self.video_label.setPixmap(QPixmap()) 
        
        if self.detection_session_id:
            self.detection_session_id = None
            
        self.log_window.log("🛑 Detección detenida", "INFO")

    def detection_loop(self):
        current_thread = threading.current_thread()
        read_error_count = 0
        
        while self.running and self.detection_thread == current_thread:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    read_error_count += 1
                    if read_error_count % 50 == 0: 
                        self.log_window.log(f"Error leyendo frame de cámara (x{read_error_count})", "WARNING")
                    time.sleep(0.1)
                    continue
                
                read_error_count = 0 
                
                self.last_frame_time = time.time()
                self.frame_count += 1
                
                if self.frame_count % self.memory_cleanup_interval == 0:
                     self.cleanup_memory()

                # Detection
                if self.model:
                     results = self.run_detection(frame)
                else:
                     results = []
                
                # Process
                _, frame_detections, _, detection_records = self.process_detections(results, frame)
                
                self.update_detection_stats(frame_detections)
                
                # Update UI
                try:
                    self.update_video_display(frame)
                except Exception:
                    pass 
                
            except Exception as e:
                self.log_window.log(f"Error loop: {e}", "ERROR")
                time.sleep(0.1)

    def run_detection(self, frame):
        """Ejecutar detección con parámetros optimizados"""
        if not self.model: return []
        return self.model.predict(source=frame, conf=0.6, iou=0.45, verbose=False)

    def process_detections(self, results, frame):
        detections_text = []
        frame_detections = {"sanas": 0, "contaminadas": 0}
        detection_records = []
        
        # Obtener dimensiones del frame para filtrado
        vh, vw = frame.shape[:2]
        max_box_area = (vw * vh) * 0.85 # Ignorar cuadros que cubran >85% de la pantalla (ruido)
        
        if len(self.spatial_grid.grid) > 100:
             self.spatial_grid.clear()
             
        now_ts = time.time()
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    # Filtrar por IDs configurados (Sistemas expertos)
                    if class_id not in self.chestnut_class_ids:
                        continue

                    class_name = result.names[class_id]
                    confidence = box.conf.item()
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Filtrar ruido de fondo (cuadros gigantescos)
                    if area > max_box_area:
                        continue
                        
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if self.spatial_grid.is_duplicate(center_x, center_y, confidence):
                        continue
                    self.spatial_grid.add_detection(center_x, center_y, confidence, area)
                    
                    # Obtener nombre traducido
                    class_data = self.class_translations.get(str(class_id))
                    if class_data:
                        display_name = class_data.get("es", class_data.get("name", class_name))
                    else:
                        display_name = class_name

                    # Quality Analysis
                    quality_key = "sana"
                    if confidence >= self.quality_confidence_threshold and area >= self.quality_area_threshold:
                         try:
                              quality_result = self.analyze_object_quality(frame, x1, y1, x2, y2, center_x, center_y, class_name, str(class_id))
                              if quality_result.lower() in self.quality_display_map:
                                   quality_key = quality_result.lower()
                         except: pass

                    quality_info = self.quality_display_map.get(quality_key, self.quality_display_map["sana"])
                    frame_detections[quality_info["count_key"]] += 1
                    
                    if quality_info["count_key"] == "contaminadas":
                         self.update_contaminated_memory(center_x, center_y, now_ts)
                         
                    color_hex = self.colors[quality_info["color_key"]]
                    cv_color = self.hex_to_bgr(color_hex)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 3)
                    # Usar display_name en lugar de hardcoded label if needed, or append it
                    label_text = f"{display_name} ({quality_info['label']})"
                    cv2.putText(frame, f"{label_text} {confidence:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 2)
                                
                    detection_records.append({
                        "class_name": class_name, "quality": quality_key, "confidence": confidence,
                        "display_name": display_name,
                        "area": area, "center_x": center_x, "center_y": center_y,
                        "width": x2-x1, "height": y2-y1
                    })

        return detections_text, frame_detections, None, detection_records

    def analyze_object_quality(self, frame, x1, y1, x2, y2, cx, cy, class_name, class_id=None):
        # 1. Prioridad: Configuración manual en classes.json
        if class_id and class_id in self.class_translations:
            manual_quality = self.class_translations[class_id].get("quality")
            if manual_quality:
                return manual_quality.lower()

        # 2. Filtrado espacial para evitar duplicados cercanos a contaminados
        for mem_cx, mem_cy, _ in self.contaminated_memory:
             dist = ((cx - mem_cx)**2 + (cy - mem_cy)**2)**0.5
             if dist < 60: return "contaminada"

        # 3. Fallback: Análisis por palabras clave
        cn_lower = class_name.lower()
        bad_keywords = ["bad", "rot", "orange", "canker", "blackspot", "greening", "mold", "stale", "damaged", "bruised", "wrinkled", "overripe"]
        if any(kw in cn_lower for kw in bad_keywords):
             return "contaminada"
        return "sana"

    def update_contaminated_memory(self, cx, cy, ts):
        updated = False
        for i, (mcx, mcy, _) in enumerate(self.contaminated_memory):
             if ((cx - mcx)**2 + (cy - mcy)**2)**0.5 < 60:
                  self.contaminated_memory[i] = (cx, cy, ts)
                  updated = True
                  break
        if not updated:
             self.contaminated_memory.append((cx, cy, ts))

    def update_detection_stats(self, frame_stats):
        self.stats["frames_processed"] += 1
        self.stats["sanas"] += frame_stats["sanas"]
        self.stats["contaminadas"] += frame_stats["contaminadas"]
        
        self.fps_frame_count += 1
        if self.fps_frame_count % 30 == 0:
             elapsed = time.time() - self.fps_start_time
             if elapsed > 0:
                  self.stats["fps"] = self.fps_frame_count / elapsed
                  self.fps_start_time = time.time()
                  self.fps_frame_count = 0

    def periodic_ui_update(self):
        # Update FPS label or status bar if exists
        pass

    def update_video_display(self, frame):
        if frame is None or frame.size == 0:
             return
        # Convert CV2 to QImage
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            # Must copy to decouple from cv2 memory if needed, but usually fine here
            self.frame_ready.emit(qt_image.copy())
        except Exception as e:
            pass

    @pyqtSlot(QImage)
    def set_video_frame(self, image):
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        scaled_pixmap = QPixmap.fromImage(image).scaled(
             lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setText("")


    def cleanup_memory(self):
         import gc
         gc.collect()
         self.spatial_grid.clear()

    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)


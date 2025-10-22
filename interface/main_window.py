"""
Ventana principal que coordina todos los widgets
"""

import json
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from functions.analysys import analyze_object_quality_with_logging

from .arduino_panel import ArduinoPanel

# Importar widgets
from .camera_panel import CameraPanel

# Los paneles de detecciones y estadísticas se han eliminado
# Las estadísticas ahora se muestran en la ventana de logs
from .controls_panel import ControlsPanel
from .log_window import LogWindow, setup_global_logging

# Configurar logging optimizado
logger = logging.getLogger(__name__)


class OptimizedLogger:
    """Logger optimizado para reducir overhead en tiempo real"""
    def __init__(self, log_window=None):
        self.log_window = log_window
        self.frame_count = 0
        self.log_interval = 30  # Log cada 30 frames
        self.buffered_messages = []
        self.max_buffer = 100

    def log_debug(self, message, force=False):
        """Log de debug optimizado"""
        self.frame_count += 1
        if force or self.frame_count % self.log_interval == 0:
            if self.log_window:
                self.log_window.log(f"[DEBUG] {message}", "DEBUG")
            else:
                print(f"[DEBUG] {message}")

    def log_info(self, message):
        """Log de información"""
        if self.log_window:
            self.log_window.log(f"{message}", "INFO")
        else:
            print(f"[INFO] {message}")

    def log_warning(self, message):
        """Log de advertencia"""
        if self.log_window:
            self.log_window.log(f"{message}", "WARNING")
        else:
            print(f"[WARNING] {message}")

    def log_error(self, message):
        """Log de error"""
        if self.log_window:
            self.log_window.log(f"{message}", "ERROR")
        else:
            print(f"[ERROR] {message}")

    def buffer_message(self, message, level="DEBUG"):
        """Bufferizar mensajes para procesamiento en lote"""
        self.buffered_messages.append((message, level))
        if len(self.buffered_messages) >= self.max_buffer:
            self.flush_buffer()

    def flush_buffer(self):
        """Procesar mensajes en buffer"""
        if self.buffered_messages and self.log_window:
            for message, level in self.buffered_messages:
                self.log_window.log(f"[{level}] {message}", level)
            self.buffered_messages.clear()


class PerformanceMonitor:
    """Monitor de rendimiento para detectar degradación"""
    def __init__(self):
        self.fps_history = []
        self.memory_history = []
        self.max_history = 100
        self.last_cleanup = time.time()
        self.cleanup_interval = 30.0  # Limpiar cada 30 segundos

    def update_fps(self, fps):
        """Actualizar historial de FPS"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)

    def update_memory(self, memory_mb):
        """Actualizar historial de memoria"""
        self.memory_history.append(memory_mb)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)

    def get_performance_stats(self):
        """Obtener estadísticas de rendimiento"""
        if not self.fps_history:
            return {"fps_avg": 0, "fps_trend": 0, "memory_avg": 0, "memory_trend": 0}

        fps_avg = sum(self.fps_history) / len(self.fps_history)
        fps_trend = 0
        if len(self.fps_history) >= 10:
            recent_avg = sum(self.fps_history[-10:]) / 10
            older_avg = sum(self.fps_history[-20:-10]) / 10 if len(self.fps_history) >= 20 else fps_avg
            fps_trend = recent_avg - older_avg

        memory_avg = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        memory_trend = 0
        if len(self.memory_history) >= 10:
            recent_mem = sum(self.memory_history[-10:]) / 10
            older_mem = sum(self.memory_history[-20:-10]) / 10 if len(self.memory_history) >= 20 else memory_avg
            memory_trend = recent_mem - older_mem

        return {
            "fps_avg": fps_avg,
            "fps_trend": fps_trend,
            "memory_avg": memory_avg,
            "memory_trend": memory_trend
        }

    def should_cleanup(self):
        """Determinar si es necesario limpiar memoria"""
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            self.last_cleanup = now
            return True
        return False


class SpatialGrid:
    """Grid espacial optimizado para detección de duplicados O(1)"""
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.grid = {}  # {(grid_x, grid_y): [(center_x, center_y, confidence, area), ...]}
        self.max_detections_per_cell = 5  # Límite para evitar saturación

    def add_detection(self, center_x, center_y, confidence, area):
        """Agregar detección al grid con límite de saturación"""
        grid_x, grid_y = center_x // self.grid_size, center_y // self.grid_size
        if (grid_x, grid_y) not in self.grid:
            self.grid[(grid_x, grid_y)] = []

        # Limitar detecciones por celda para evitar saturación
        if len(self.grid[(grid_x, grid_y)]) >= self.max_detections_per_cell:
            # Mantener solo las más recientes (por confianza)
            self.grid[(grid_x, grid_y)].sort(key=lambda x: x[2], reverse=True)
            self.grid[(grid_x, grid_y)] = self.grid[(grid_x, grid_y)][:self.max_detections_per_cell]

        self.grid[(grid_x, grid_y)].append((center_x, center_y, confidence, area))

    def is_duplicate(self, center_x, center_y, confidence, min_distance=50, min_confidence=0.7):
        """Verificar si es duplicado usando grid espacial O(1)"""
        grid_x, grid_y = center_x // self.grid_size, center_y // self.grid_size

        # Verificar celdas adyacentes (3x3)
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
        """Limpiar grid para nuevo frame"""
        self.grid.clear()


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

        # Variables de gestión de memoria y rendimiento (EXTREMA-AGRESIVAS)
        self._cpu_process = None
        self._last_cpu_time = time.time()
        self._last_cpu_percent = 0.0
        self._cpu_init_time = time.time()
        self.memory_cleanup_interval = 25   # Limpiar memoria cada 25 frames (extrema-frecuente)
        self.frame_count_since_cleanup = 0
        self.max_contaminated_memory = 5    # Máximo 5 objetos en memoria (extrema-restrictivo)
        self.max_detection_history = 10     # Máximo 10 detecciones en historial
        self.performance_monitor = PerformanceMonitor()

        # Variables de estadísticas (ahora se muestran en logs)
        self.stats = {
            'frames_processed': 0,
            'sanas': 0,
            'contaminadas': 0,
            'total_detections': 0,
            'fps': 0.0,
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'system_cpu_percent': 0.0,
            'threads': 0,
            'processes': 0,
            'gpu_name': 'N/A',
            'vram_total_gb': 0.0,
            'vram_used_mb': 0.0,
            'vram_free_mb': 0.0
        }

        # Variables para limpieza balanceada
        self.last_aggressive_cleanup = time.time()
        self.aggressive_cleanup_interval = 5.0   # Limpieza cada 5 segundos (extrema-frecuente)
        self.inference_time_history = []  # Historial de tiempos de inferencia
        self.max_inference_time_history = 10    # Solo 10 tiempos máximo (ultra-restrictivo)

        # Variables para reinicio automático (más permisivo)
        self.last_restart_time = time.time()
        self.restart_interval = 600.0  # Reiniciar cada 10 minutos si hay problemas
        self.crash_count = 0
        self.max_crashes = 5

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

        # Grid espacial para detección de duplicados optimizada
        self.spatial_grid = SpatialGrid(grid_size=50)

        # Logger optimizado temporal (se reemplazará después de crear log_window)
        self.optimized_logger = OptimizedLogger()

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

        # Actualizar logger optimizado con ventana de logs
        self.optimized_logger.log_window = self.log_window

        # Las estadísticas se mostrarán solo cuando haya problemas

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

        # Ya no necesitamos los paneles de detecciones y estadísticas
        # Las estadísticas se mostrarán en la ventana de logs

    def load_interface_config(self):
        """Cargar configuración de interfaz desde JSON"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interface_config.json")
            with open(config_path, encoding='utf-8') as f:
                interface_config = json.load(f)

                # Cargar colores desde configuración
                if "colors" in interface_config:
                    self.colors.update(interface_config["colors"])

                # Cargar clases de detección desde configuración
                if "detection_classes" in interface_config:
                    self.chestnut_classes = interface_config["detection_classes"]
                    self.optimized_logger.log_info(f"Clases de detección cargadas desde JSON: {self.chestnut_classes}")

                    if not self.chestnut_classes:
                        self.optimized_logger.log_warning("No hay clases de detección en interface_config.json")
                        self.chestnut_classes = ['apple', 'orange']
                        self.optimized_logger.log_info(f"Usando clases por defecto: {self.chestnut_classes}")
                else:
                    self.optimized_logger.log_error("No se encontraron clases de detección en interface_config.json")
                    self.chestnut_classes = ['apple', 'orange']
                    self.optimized_logger.log_info(f"Usando clases por defecto: {self.chestnut_classes}")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.optimized_logger.log_error(f"Error cargando interface_config.json: {e}")
            self.optimized_logger.log_info("Usando configuración por defecto")
            if not self.chestnut_classes:
                self.chestnut_classes = ['apple', 'orange']
                self.optimized_logger.log_info(f"Clases por defecto: {self.chestnut_classes}")

    def on_camera_changed(self, camera_id: int, camera_info: dict):
        """Callback cuando cambia la cámara"""
        # Actualizar estadísticas en logs
        self.stats['camera_id'] = camera_id
        self.stats['camera_name'] = camera_info['name']
        self.log_stats()

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
            # Actualizar estadísticas en logs
            self.stats['model'] = selected_model
            # Mostrar estadísticas iniciales
            self.optimized_logger.log_info("🚀 Iniciando detección - Estadísticas del sistema:")
            self.log_stats()

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
            accel_text = "GPU FP16" if using_gpu else "CPU"
        except Exception:
            accel_text = "CPU"
        # Log de información de aceleración
        self.optimized_logger.log_info(f"🚀 {accel_text}")

    def get_gpu_info(self):
        """Obtener información de GPU y VRAM"""
        try:
            import torch
            if torch.cuda.is_available():
                # Información de GPU
                gpu_name = torch.cuda.get_device_name(0)

                # Información de VRAM
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB
                vram_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024  # MB
                vram_cached = torch.cuda.memory_reserved(0) / 1024 / 1024  # MB
                vram_free = (vram_total * 1024) - vram_cached  # MB

                return {
                    'gpu_name': gpu_name,
                    'vram_total_gb': vram_total,
                    'vram_used_mb': vram_allocated,
                    'vram_free_mb': vram_free
                }
        except Exception:
            pass
        return {
            'gpu_name': 'No disponible',
            'vram_total_gb': 0.0,
            'vram_used_mb': 0.0,
            'vram_free_mb': 0.0
        }

    def get_system_info(self):
        """Obtener información del sistema"""
        try:
            import psutil
            
            # Inicializar proceso para CPU si no existe
            if self._cpu_process is None:
                self._cpu_process = psutil.Process()
                # Primera llamada para inicializar CPU percent
                self._cpu_process.cpu_percent()
                self._last_cpu_time = time.time()
                self._cpu_init_time = time.time()

            # Información del proceso actual
            threads = self._cpu_process.num_threads()
            memory_mb = self._cpu_process.memory_info().rss / 1024 / 1024
            
            # Medir CPU con intervalo mínimo y tiempo de calentamiento
            current_time = time.time()
            time_since_init = current_time - self._cpu_init_time
            
            # Esperar al menos 2 segundos para que CPU percent se calibre
            if time_since_init >= 2.0 and current_time - self._last_cpu_time >= 0.5:
                cpu_percent = self._cpu_process.cpu_percent()
                self._last_cpu_time = current_time
                self._last_cpu_percent = cpu_percent
            elif time_since_init < 2.0:
                # Durante los primeros 2 segundos, usar CPU del sistema
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self._last_cpu_percent = cpu_percent
            else:
                # Usar valor anterior
                cpu_percent = self._last_cpu_percent

            # Información del sistema
            processes = len(psutil.pids())
            system_cpu_percent = psutil.cpu_percent(interval=0.1)

            # Obtener RAM total y disponible del sistema
            system_memory = psutil.virtual_memory()
            total_ram_gb = system_memory.total / (1024**3)
            available_ram_gb = system_memory.available / (1024**3)
            used_ram_gb = system_memory.used / (1024**3)

            return {
                'threads': threads,
                'processes': processes,
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'system_cpu_percent': system_cpu_percent,
                'total_ram_gb': total_ram_gb,
                'available_ram_gb': available_ram_gb,
                'used_ram_gb': used_ram_gb
            }
        except Exception:
            return {
                'threads': 0,
                'processes': 0,
                'memory_mb': 0.0,
                'cpu_percent': 0.0,
                'total_ram_gb': 0.0,
                'available_ram_gb': 0.0,
                'used_ram_gb': 0.0
            }

    def log_stats(self):
        """Mostrar estadísticas en logs solo cuando hay problemas"""
        try:
            # Obtener información del sistema
            system_info = self.get_system_info()
            gpu_info = self.get_gpu_info()

            # Actualizar estadísticas internas
            self.stats.update(system_info)
            self.stats.update(gpu_info)

            # Mostrar estadísticas si hay problemas de rendimiento
            show_stats = False
            reason = ""

            if self.stats['memory_mb'] > 1000:
                show_stats = True
                reason += f"RAM alta ({self.stats['memory_mb']:.1f}MB) "

            if self.stats['cpu_percent'] > 50:
                show_stats = True
                reason += f"CPU alto ({self.stats['cpu_percent']:.1f}%) "

            if self.stats['fps'] > 0 and self.stats['fps'] < 10:
                show_stats = True
                reason += f"FPS bajo ({self.stats['fps']:.1f}) "

            if show_stats:
                # Información de VRAM
                vram_text = ""
                if gpu_info['gpu_name'] != 'No disponible':
                    vram_usage_percent = (gpu_info['vram_used_mb'] / (gpu_info['vram_total_gb'] * 1024)) * 100
                    vram_text = f", VRAM: {gpu_info['vram_used_mb']:.1f}MB/{gpu_info['vram_total_gb']:.1f}GB ({vram_usage_percent:.1f}%)"
                    
                    # Agregar alerta de VRAM alta
                    if vram_usage_percent > 85:
                        reason += f"VRAM CRÍTICA ({vram_usage_percent:.1f}%) "

                stats_text = (
                    f"📊 ESTADÍSTICAS CRÍTICAS ({reason.strip()}): "
                    f"RAM: {self.stats['memory_mb']:.1f}MB, "
                    f"RAM Total: {self.stats['total_ram_gb']:.1f}GB, "
                    f"RAM Disponible: {self.stats['available_ram_gb']:.1f}GB, "
                    f"CPU Proc: {self.stats['cpu_percent']:.1f}%, "
                    f"CPU Sist: {self.stats.get('system_cpu_percent', 0.0):.1f}%, "
                    f"Hilos: {self.stats['threads']}"
                    f"{vram_text}, "
                    f"FPS: {self.stats['fps']:.1f}, "
                    f"Frames: {self.stats['frames_processed']}, "
                    f"Detectadas: {self.stats['total_detections']}"
                )
                self.optimized_logger.log_warning(stats_text)

        except Exception as e:
            self.optimized_logger.log_error(f"Error obteniendo estadísticas: {e}")

    def configure_yolo_classes(self):
        """Configurar YOLO para detectar solo las clases deseadas"""
        if not self.model:
            return

        try:
            all_classes = list(self.model.names.values())
            self.optimized_logger.log_info(f"Clases YOLO disponibles: {all_classes}")

            # Buscar IDs de clases de castañas
            self.chestnut_class_ids = []
            for desired_class in self.chestnut_classes:
                for class_id, class_name in self.model.names.items():
                    if class_name.lower() == desired_class.lower():
                        self.chestnut_class_ids.append(class_id)
                        self.optimized_logger.log_info(f"Clase encontrada: {class_name} (ID: {class_id})")
                        break
                else:
                    self.optimized_logger.log_warning(f"Clase no encontrada en YOLO: {desired_class}")

            if self.chestnut_class_ids:
                self.optimized_logger.log_info(f"✅ YOLO configurado para detectar SOLO: {[self.model.names[i] for i in self.chestnut_class_ids]}")
                self.optimized_logger.log_info(f"📊 IDs de clases: {self.chestnut_class_ids}")
            else:
                self.optimized_logger.log_warning("⚠️ No se encontraron clases válidas en YOLO")
                self.chestnut_class_ids = []

        except Exception as e:
            self.optimized_logger.log_error(f"Error configurando clases YOLO: {e}")
            self.chestnut_class_ids = []

    def stop_detection(self):
        """Detener detección"""
        self.running = False
        if self.camera:
            self.camera.release()

        # Mostrar estadísticas finales
        self.optimized_logger.log_info("🛑 Detección detenida - Estadísticas finales:")
        self.log_stats()

        # Limpiar memoria y resetear contadores
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.contaminated_memory.clear()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(text="Cámara detenida")

    def detection_loop(self):
        """Bucle principal de detección optimizado con gestión de memoria"""
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
                self.frame_count_since_cleanup += 1

                # Guardar dimensiones del frame
                frame_height, frame_width = frame.shape[:2]

                # Limpieza automática de memoria cada N frames
                if self.frame_count_since_cleanup >= self.memory_cleanup_interval:
                    self.cleanup_memory()
                    self.frame_count_since_cleanup = 0

                # Limpieza basada en tiempo (cada 30 segundos)
                if self.performance_monitor.should_cleanup():
                    self.cleanup_memory()

                # Limpieza ultra-agresiva si hay degradación severa
                now_time = time.time()
                if now_time - self.last_aggressive_cleanup > self.aggressive_cleanup_interval:
                    # Verificar si hay degradación severa
                    if len(self.inference_time_history) >= 5:
                        recent_avg = sum(self.inference_time_history[-5:]) / 5
                        if recent_avg > 100.0:  # Si promedio > 100ms (más permisivo)
                            self.ultra_aggressive_cleanup()
                            self.optimized_logger.log_warning(f"⚠️ Degradación detectada (promedio: {recent_avg:.1f}ms). Limpieza ultra-agresiva aplicada.")

                # Reinicio automático si hay problemas persistentes
                if now_time - self.last_restart_time > self.restart_interval:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > 3000:  # Si RAM > 3GB (más permisivo)
                            self.optimized_logger.log_warning(f"🔄 RAM crítica ({memory_mb:.1f}MB). Reiniciando detección...")
                            self.restart_detection()
                            self.last_restart_time = now_time
                    except Exception:
                        pass

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
        """Ejecutar detección YOLO con monitoreo de rendimiento y filtrado de clases"""
        import time

        # Medir tiempo de inferencia
        start_time = time.time()

        # Configuración por modelo con optimización de VRAM
        gpu_info = self.get_gpu_info()
        vram_total_gb = gpu_info.get('vram_total_gb', 0)
        
        # Ajustar resolución según VRAM disponible
        if vram_total_gb >= 8:  # GPU con 8GB+ VRAM
            img_size = 640
        elif vram_total_gb >= 4:  # GPU con 4-8GB VRAM
            img_size = 512
        else:  # GPU con menos de 4GB VRAM
            img_size = 416
            
        if self.current_model_name == "YOLO12l":
            predict_kwargs = {'conf': 0.4, 'imgsz': img_size, 'device': 0}
        elif self.current_model_name == "YOLO12x":
            predict_kwargs = {'conf': 0.35, 'imgsz': img_size, 'device': 0}
        else:  # YOLO12n
            predict_kwargs = {'conf': 0.5, 'imgsz': img_size, 'device': 0}

        # OPTIMIZACIÓN CRÍTICA: Filtrar clases a nivel de YOLO
        if hasattr(self, 'chestnut_class_ids') and self.chestnut_class_ids:
            predict_kwargs['classes'] = self.chestnut_class_ids  # Solo detectar clases de castañas
            self.optimized_logger.log_debug(f"🔍 YOLO filtrando clases: {self.chestnut_class_ids}")

        try:
            import torch
            if torch.cuda.is_available():
                predict_kwargs['half'] = True
                # OPTIMIZACIÓN VRAM: Limpiar caché antes de inferencia
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Ejecutar detección (SOLO CLASES DE CASTAÑAS)
        results = self.model.predict(frame, **predict_kwargs)
        
        # OPTIMIZACIÓN VRAM: Limpiar caché después de inferencia
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Medir tiempo total
        inference_time = (time.time() - start_time) * 1000  # Convertir a ms
        
        # Log periódico de VRAM cada 500 frames
        if self.stats['frames_processed'] % 500 == 0:
            gpu_info = self.get_gpu_info()
            if gpu_info['gpu_name'] != 'No disponible':
                vram_usage_percent = (gpu_info['vram_used_mb'] / (gpu_info['vram_total_gb'] * 1024)) * 100
                self.optimized_logger.log_info(f"🎮 GPU: {gpu_info['gpu_name']}, VRAM: {gpu_info['vram_used_mb']:.1f}MB/{gpu_info['vram_total_gb']:.1f}GB ({vram_usage_percent:.1f}%)")

        # Agregar a historial
        self.inference_time_history.append(inference_time)
        if len(self.inference_time_history) > self.max_inference_time_history:
            self.inference_time_history.pop(0)

        # Log de tiempos altos con estadísticas del sistema
        if inference_time > 200.0:  # Si > 200ms (más permisivo)
            system_info = self.get_system_info()
            gpu_info = self.get_gpu_info()
            self.optimized_logger.log_warning(f"⚠️ Tiempo de inferencia alto: {inference_time:.1f}ms")
            self.optimized_logger.log_warning(f"📊 SISTEMA: RAM: {system_info['memory_mb']:.1f}MB, CPU: {system_info['cpu_percent']:.1f}%, Hilos: {system_info['threads']}")
            self.optimized_logger.log_warning(f"💾 RAM Total: {system_info['total_ram_gb']:.1f}GB, Disponible: {system_info['available_ram_gb']:.1f}GB")
            if gpu_info['gpu_name'] != 'No disponible':
                vram_usage_percent = (gpu_info['vram_used_mb'] / (gpu_info['vram_total_gb'] * 1024)) * 100
                self.optimized_logger.log_warning(f"🎮 GPU: {gpu_info['gpu_name']}, VRAM: {gpu_info['vram_used_mb']:.1f}MB/{gpu_info['vram_total_gb']:.1f}GB ({vram_usage_percent:.1f}%)")
                
                # Alerta de VRAM alta
                if vram_usage_percent > 85:
                    self.optimized_logger.log_warning(f"🚨 VRAM CRÍTICA: {vram_usage_percent:.1f}% - Considera reducir resolución o cambiar a modelo más ligero")

        return results

    def process_detections(self, results, frame, frame_width, frame_height):
        """Procesar resultados de detección"""
        detections_text = []
        frame_detections = {'sanas': 0, 'contaminadas': 0, 'detectadas': 0}

        # Contador de debug global
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1

        # Limpiar grid espacial para nuevo frame (OPTIMIZADO)
        # Solo limpiar si está muy lleno para evitar limpieza innecesaria
        if len(self.spatial_grid.grid) > 100:  # Solo limpiar si hay muchas celdas
            self.spatial_grid.clear()

        # Purga de memoria de contaminados (optimizada)
        now_ts = time.time()
        # Limitar tamaño de memoria de contaminados (CORREGIDO)
        if len(self.contaminated_memory) > self.max_contaminated_memory:
            # Mantener solo los más recientes (ordenar por timestamp)
            self.contaminated_memory.sort(key=lambda x: x[2], reverse=True)  # x[2] es timestamp
            self.contaminated_memory = self.contaminated_memory[:self.max_contaminated_memory]

        # Filtrar por tiempo (mantener solo los últimos 3 segundos)
        self.contaminated_memory = [m for m in self.contaminated_memory if now_ts - m[2] < 3.0]

        crossed_line_contaminated = False

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = box.conf.item()

                    # Ya no necesitamos filtrar clases aquí porque YOLO ya filtra
                    # Solo procesamos las clases de castañas que YOLO detectó

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)

                    # Evitar duplicados usando grid espacial O(1)
                    if self.spatial_grid.is_duplicate(center_x, center_y, confidence):
                        continue

                    # Agregar detección al grid
                    self.spatial_grid.add_detection(center_x, center_y, confidence, area)

                    # Análisis de calidad (ULTRA-OPTIMIZADO - solo críticos)
                    if confidence > 0.8 and area > 5000:  # Solo objetos grandes y muy confiables
                        try:
                            quality = self.analyze_object_quality(frame, x1, y1, x2, y2, center_x, center_y, class_name)
                        except Exception:
                            quality = "SANAS"  # Fallback si hay error
                    else:
                        quality = "SANAS"  # Asumir sano para la mayoría de objetos

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
                            self.optimized_logger.log_debug(f"LÍNEA DIBUJADA: ({line_start[0]:.2f}, {line_start[1]:.2f}) -> ({line_end[0]:.2f}, {line_end[1]:.2f})")
                        is_near = self.is_near_line(center_x, center_y, line_start, line_end, frame_width, frame_height)
                        if is_near:
                            self.optimized_logger.log_debug(f"OBJETO CERCA DE LÍNEA: {label} en ({center_x}, {center_y})")
                            if quality == 'contaminada':
                                crossed_line_contaminated = True
                                self.optimized_logger.log_info(f"CONTAMINADA CRUZANDO LÍNEA: {label}")

                    # Dibujar rectángulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.hex_to_bgr(color), 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hex_to_bgr(color), 2)

                    detections_text.append(f"{label} - {time.strftime('%H:%M:%S')}")

        return detections_text, frame_detections, crossed_line_contaminated

    def analyze_object_quality(self, frame, x1, y1, x2, y2, center_x, center_y, class_name):
        """Analizar calidad del objeto"""
        # Verificar memoria de contaminados
        for mem_cx, mem_cy, _mem_ts in self.contaminated_memory:
            dist_mem = ((center_x - mem_cx) ** 2 + (center_y - mem_cy) ** 2) ** 0.5
            if dist_mem < 60:
                return 'contaminada'

        # Análisis RGB
        if x2 > x1 and y2 > y1:
            crop_img = frame[y1:y2, x1:x1 + (x2 - x1)]
            analysis_class = class_name.lower()
            if analysis_class in ['orange']:
                analysis_class = 'apple'
            return analyze_object_quality_with_logging(crop_img, analysis_class)

        return 'detectada'

    def update_contaminated_memory(self, center_x, center_y, now_ts):
        """Actualizar memoria de contaminados"""
        updated = False
        for i, (mem_cx, mem_cy, _mem_ts) in enumerate(self.contaminated_memory):
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
                self.optimized_logger.log_debug(f"DISTANCIA A LÍNEA: objeto({center_x}, {center_y}) -> línea({lx1}, {ly1})-({lx2}, {ly2}) = {dist:.1f}px")

            detection_radius = self.controls_panel.get_line_detection_radius()
            is_near = dist < detection_radius
            if is_near and self._debug_counter % 30 == 0:
                self.optimized_logger.log_debug(f"OBJETO CERCA DE LÍNEA: {dist:.1f}px < {detection_radius}px")

            return is_near
        except Exception as e:
            self.log_window.log(f"❌ Error en is_near_line: {e}", "ERROR")
            return False

    def update_detection_stats(self, frame_detections):
        """Actualizar estadísticas de detección"""
        # Actualizar estadísticas internas
        self.stats['frames_processed'] += 1
        self.stats['sanas'] += frame_detections['sanas']
        self.stats['contaminadas'] += frame_detections['contaminadas']
        self.stats['total_detections'] += sum(frame_detections.values())

        # Calcular FPS
        self.fps_frame_count += 1
        if self.fps_frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                fps = self.fps_frame_count / elapsed
                self.stats['fps'] = fps
                # Actualizar monitor de rendimiento
                self.performance_monitor.update_fps(fps)
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
                self.optimized_logger.log_debug(f"CRUCE DETECTADO: crossed_line_contaminated={crossed_line_contaminated}, desired_state={desired_state_contaminated}")

            # Debounce logic
            if desired_state_contaminated != self.last_detection_contaminated:
                self.state_stable_since = now_time
                self.last_detection_contaminated = desired_state_contaminated
                self.optimized_logger.log_info(f"CAMBIO DE ESTADO: {self.last_detection_contaminated} -> {desired_state_contaminated}")

            debounce_s = self.controls_panel.get_arduino_debounce()
            cooldown_s = self.controls_panel.get_arduino_cooldown()

            time_stable = now_time - self.state_stable_since
            self.optimized_logger.log_debug(f"TIEMPO ESTABLE: {time_stable:.1f}s (necesario: {debounce_s}s), COOLDOWN: {now_time >= self.arduino_block_until}")

            if (now_time - self.state_stable_since) >= debounce_s and now_time >= self.arduino_block_until:
                if desired_state_contaminated:
                    self.optimized_logger.log_info("ENVIANDO CONTAMINADO AL ARDUINO")
                    self.arduino_panel.send_signal('CONTAMINADO', delay_prevention=0.3)
                    self.arduino_block_until = now_time + cooldown_s
                elif self.controls_panel.has_line():
                    self.optimized_logger.log_info("ENVIANDO SANO AL ARDUINO")
                    self.arduino_panel.send_signal('SANO', delay_prevention=0.1)
                    self.arduino_block_until = now_time + cooldown_s
        except Exception as e:
            self.optimized_logger.log_error(f"Error en handle_arduino_signals: {e}")

    def update_ui(self, frame, detections_text):
        """Actualizar interfaz de usuario"""
        try:
            # Actualizar video
            self.update_video_display(frame)

            # Actualizar detecciones (ya no se muestran en logs)

            # Actualizar estado Arduino en panel de estadísticas
            if self.arduino_panel.is_connected():
                manager = self.arduino_panel.get_manager()
                self.stats['arduino_status'] = f"Conectado ({manager.port or 'N/D'})"
            else:
                self.stats['arduino_status'] = "Desconectado"

            # Actualizar información de rendimiento (cada 10 frames para no saturar)
            if self.frame_count % 10 == 0:
                try:
                    # Obtener información del sistema
                    system_info = self.get_system_info()
                    self.stats.update(system_info)

                    # Mostrar estadísticas cada 100 frames para monitoreo
                    if self.frame_count % 100 == 0:
                        self.log_stats()
                except Exception:
                    pass  # Ignorar errores de psutil

        except Exception as e:
            self.optimized_logger.log_error(f"Error actualizando UI: {e}")

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
                self.optimized_logger.log_error("No se pudo reabrir la cámara")
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
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def cleanup_memory(self):
        """Limpieza automática de memoria para mantener rendimiento"""
        try:
            # Limpiar memoria de contaminados (más agresivo)
            now_ts = time.time()
            self.contaminated_memory = [m for m in self.contaminated_memory if now_ts - m[2] < 1.0]  # Solo 1 segundo

            # Limpiar buffer de logs
            if hasattr(self.optimized_logger, 'buffered_messages'):
                self.optimized_logger.buffered_messages.clear()

            # Limpiar cache de imágenes si existe
            try:
                from functions.image_cache import get_image_cache
                image_cache = get_image_cache()
                if hasattr(image_cache, 'clear_cache'):
                    image_cache.clear_cache()
            except Exception:
                pass

            # Limpiar grid espacial
            self.spatial_grid.clear()

            # Limpiar historial de tiempos de inferencia
            if len(self.inference_time_history) > self.max_inference_time_history:
                self.inference_time_history = self.inference_time_history[-self.max_inference_time_history:]

            # Forzar garbage collection
            import gc
            gc.collect()

            # Actualizar estadísticas de rendimiento
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.performance_monitor.update_memory(memory_mb)

            # Log de limpieza (solo cada 5 limpiezas para no saturar)
            if self.frame_count % (self.memory_cleanup_interval * 5) == 0:
                self.optimized_logger.log_info(f"Limpieza de memoria completada. RAM: {memory_mb:.1f}MB")

        except Exception as e:
            self.optimized_logger.log_error(f"Error en limpieza de memoria: {e}")

    def aggressive_cleanup(self):
        """Limpieza agresiva para casos de degradación severa"""
        try:
            self.optimized_logger.log_info("🧹 Iniciando limpieza agresiva de memoria...")

            # Limpiar TODO
            self.contaminated_memory.clear()
            self.inference_time_history.clear()
            self.spatial_grid.clear()

            # Limpiar buffers de logs
            if hasattr(self.optimized_logger, 'buffered_messages'):
                self.optimized_logger.buffered_messages.clear()

            # Limpiar cache de imágenes
            try:
                from functions.image_cache import get_image_cache
                image_cache = get_image_cache()
                if hasattr(image_cache, 'clear_cache'):
                    image_cache.clear_cache()
            except Exception:
                pass

            # Limpiar cache de CUDA si está disponible
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.optimized_logger.log_info("✅ Cache de CUDA limpiado")
            except Exception:
                pass

            # Forzar garbage collection múltiple
            import gc
            for _ in range(3):
                gc.collect()

            # Reiniciar contadores
            self.frame_count_since_cleanup = 0
            self.last_aggressive_cleanup = time.time()

            # Actualizar estadísticas
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.performance_monitor.update_memory(memory_mb)

            self.optimized_logger.log_info(f"✅ Limpieza agresiva completada. RAM: {memory_mb:.1f}MB")

        except Exception as e:
            self.optimized_logger.log_error(f"Error en limpieza agresiva: {e}")

    def ultra_aggressive_cleanup(self):
        """Limpieza ultra-agresiva para casos críticos"""
        try:
            self.optimized_logger.log_info("🚨 Iniciando limpieza ULTRA-AGRESIVA...")

            # Limpiar TODO de forma más agresiva
            self.contaminated_memory.clear()
            self.inference_time_history.clear()
            self.spatial_grid.clear()

            # Limpiar buffers de logs
            if hasattr(self.optimized_logger, 'buffered_messages'):
                self.optimized_logger.buffered_messages.clear()

            # Limpiar cache de imágenes
            try:
                from functions.image_cache import get_image_cache
                image_cache = get_image_cache()
                if hasattr(image_cache, 'clear_cache'):
                    image_cache.clear_cache()
            except Exception:
                pass

            # Limpiar cache de CUDA múltiple
            try:
                import torch
                if torch.cuda.is_available():
                    for _ in range(5):  # 5 veces para asegurar limpieza
                        torch.cuda.empty_cache()
                    self.optimized_logger.log_info("✅ Cache de CUDA limpiado 5 veces")
            except Exception:
                pass

            # Garbage collection ultra-agresivo
            import gc
            for _ in range(10):  # 10 veces para limpieza completa
                gc.collect()

            # Reiniciar contadores
            self.frame_count_since_cleanup = 0
            self.last_aggressive_cleanup = time.time()

            # Actualizar estadísticas
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.performance_monitor.update_memory(memory_mb)

            self.optimized_logger.log_info(f"✅ Limpieza ultra-agresiva completada. RAM: {memory_mb:.1f}MB")

        except Exception as e:
            self.optimized_logger.log_error(f"Error en limpieza ultra-agresiva: {e}")

    def restart_detection(self):
        """Reiniciar detección completamente"""
        try:
            self.optimized_logger.log_info("🔄 Reiniciando detección...")

            # Detener detección actual
            self.running = False
            if self.camera:
                self.camera.release()

            # Limpiar todo
            self.ultra_aggressive_cleanup()

            # Reiniciar cámara
            time.sleep(1.0)  # Esperar 1 segundo
            camera_id = self.camera_panel.get_camera_id()
            camera_info = self.camera_panel.get_camera_info()

            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                self.optimized_logger.log_error("❌ No se pudo reabrir la cámara")
                # Mostrar estadísticas del sistema cuando hay problemas con la cámara
                system_info = self.get_system_info()
                gpu_info = self.get_gpu_info()
                self.optimized_logger.log_error(f"📊 SISTEMA: RAM: {system_info['memory_mb']:.1f}MB, CPU: {system_info['cpu_percent']:.1f}%, Hilos: {system_info['threads']}")
                if gpu_info['gpu_name'] != 'No disponible':
                    self.optimized_logger.log_error(f"🎮 GPU: {gpu_info['gpu_name']}, VRAM: {gpu_info['vram_used_mb']:.1f}MB/{gpu_info['vram_total_gb']:.1f}GB")
                return False

            self.configure_camera(camera_info)

            # Reiniciar detección
            self.running = True
            self.optimized_logger.log_info("✅ Detección reiniciada exitosamente")
            return True

        except Exception as e:
            self.optimized_logger.log_error(f"Error reiniciando detección: {e}")
            return False

    def get_performance_info(self):
        """Obtener información de rendimiento actual"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            stats = self.performance_monitor.get_performance_stats()

            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "fps_avg": stats["fps_avg"],
                "fps_trend": stats["fps_trend"],
                "memory_trend": stats["memory_trend"],
                "contaminated_memory_size": len(self.contaminated_memory),
                "frame_count": self.frame_count
            }
        except Exception as e:
            return {"error": str(e)}

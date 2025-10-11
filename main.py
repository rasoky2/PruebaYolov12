import cv2
from ultralytics import YOLO
import os
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
import time
from image import FilterManager
from arduino import ArduinoManager

# Variables globales para Arduino
arduino_manager = None

# Variable global para dispositivo de procesamiento
device = "cpu"  # Por defecto CPU

def init_arduino_manager():
    """Inicializar el gestor de Arduino"""
    global arduino_manager
    arduino_manager = ArduinoManager()
    return arduino_manager

def load_camera_config() -> Optional[Dict[str, Any]]:
    """Cargar configuración de cámaras desde JSON"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "camera_config.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("[WARNING] Archivo camera_config.json no encontrado. Usando configuración por defecto.")
        return None
    except json.JSONDecodeError:
        print("[ERROR] Error al leer camera_config.json. Usando configuración por defecto.")
        return None

def get_camera_name(cam_id: int, config: Optional[Dict[str, Any]]) -> str:
    """Obtener nombre personalizado de la cámara"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)]["name"]
    return f"Dispositivo {cam_id}"

def get_camera_description(cam_id: int, config: Optional[Dict[str, Any]]) -> str:
    """Obtener descripción de la cámara"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)].get("description", "")
    return ""

def is_favorite_camera(cam_id: int, config: Optional[Dict[str, Any]]) -> bool:
    """Verificar si la cámara es favorita"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)].get("is_favorite", False)
    return False

def get_favorite_camera(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """Obtener la cámara favorita"""
    if config and "cameras" in config:
        for cam_id, cam_info in config["cameras"].items():
            if cam_info.get("is_favorite", False):
                return int(cam_id)
    return None

def assign_color(chestnut_type: str) -> Tuple[int, int, int]:
    """Asignar colores específicos para castañas"""
    color_map = {
        'sana': (0, 255, 0),        # Verde - Castaña sana
        'contaminada': (0, 0, 255),  # Rojo - Castaña contaminada
        'castaña': (255, 165, 0),    # Naranja - Castaña general
    }
    return color_map.get(chestnut_type, (255, 255, 255))  # Blanco por defecto

# Funciones de filtros movidas a image/filters.py

def analyze_chestnut_quality(crop_img: Optional[np.ndarray]) -> str:
    """Analizar calidad de castaña usando análisis HSV y textura"""
    if crop_img is None or crop_img.size == 0:
        return 'sana'  # Por defecto si no se puede analizar
    
    # Convertir a HSV para análisis de color
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)  # Usar _ para variable no usada
    
    # Calcular estadísticas de color
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_v = np.std(v)  # Desviación estándar del brillo
    
    # Análisis de textura simple (detección de bordes)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Heurística para clasificar (ajustable según tus imágenes)
    # Castañas contaminadas suelen tener:
    # - Menor brillo (valores V bajos)
    # - Menor saturación (colores más grisáceos)
    # - Más bordes (manchas, moho, irregularidades)
    
    contamination_score = 0
    
    # Factor 1: Brillo bajo (posible moho o manchas oscuras)
    if mean_v < 70:
        contamination_score += 2
    elif mean_v < 100:
        contamination_score += 1
    
    # Factor 2: Saturación baja (colores apagados)
    if mean_s < 40:
        contamination_score += 2
    elif mean_s < 60:
        contamination_score += 1
    
    # Factor 3: Mucha variación en brillo (manchas)
    if std_v > 35:
        contamination_score += 1
    
    # Factor 4: Alta densidad de bordes (textura irregular)
    if edge_density > 0.15:
        contamination_score += 1
    
    # Clasificar basado en puntuación
    return 'contaminada' if contamination_score >= 1 else 'sana'

def analyze_chestnut_quality_dual(crop_img: Optional[np.ndarray], filter_manager: FilterManager = None) -> str:
    """Análisis dual: RGB normal + simulación UV para máxima precisión"""
    if crop_img is None or crop_img.size == 0:
        return 'sana'
    
    # Análisis RGB normal
    rgb_result = analyze_chestnut_quality(crop_img)
    
    # Análisis con simulación UV usando FilterManager
    if filter_manager:
        uv_img, _, _ = filter_manager.apply_filter(crop_img, "uv")
        uv_result = analyze_chestnut_quality(uv_img)
    else:
        # Fallback si no hay FilterManager
        uv_result = rgb_result
    
    # Lógica de decisión dual
    final_result = 'contaminada' if (rgb_result == 'contaminada' or uv_result == 'contaminada') else 'sana'
    
    return final_result

def classify_chestnut(label: str, confidence: float, crop_img: Optional[np.ndarray] = None, filter_manager: FilterManager = None) -> Optional[str]:
    """Clasificar castañas en sanas o contaminadas usando YOLO + análisis RGB"""
    label_lower = label.lower()
    
    # Clases que detectan castañas (similares por forma) - Actualizado con análisis YOLO12n
    chestnut_classes = ['sports ball', 'apple', 'orange', 'donut', 'bowl']
    
    # Verificar si es una clase que puede ser castaña
    if label_lower not in chestnut_classes:
        return None  # No es una castaña
    
    # Si tenemos la imagen recortada, usar análisis dual RGB + UV
    if crop_img is not None:
        return analyze_chestnut_quality_dual(crop_img, filter_manager)
    
    # Configuración de clasificación por clase y confianza
    classification_rules = {
        'sports ball': {'sana': 0.6, 'contaminada': 0.4},
        'apple': {'sana': 0.7, 'contaminada': 0.5},
        'orange': {'sana': 0.4, 'contaminada': 0.6},
        'donut': {'sana': 0.8, 'contaminada': 0.6},
        'bowl': {'sana': 0.8, 'contaminada': 0.6}
    }
    
    # Aplicar reglas de clasificación
    rules = classification_rules.get(label_lower, {})
    if not rules:
        return None
    
    if confidence >= rules.get('sana', 0):
        return 'sana'
    elif confidence >= rules.get('contaminada', 0):
        return 'contaminada'
    
    return None  # No clasificar si confianza muy baja

def switch_camera(new_camera_id: int, available_cameras: list, current_camera_id: int, camera_live) -> Tuple[int, Any]:
    """Cambiar a una nueva cámara"""
    if new_camera_id in available_cameras and new_camera_id != current_camera_id:
        # Cerrar cámara actual
        camera_live.release()
        
        # Abrir nueva cámara
        new_camera = cv2.VideoCapture(new_camera_id)
        if new_camera.isOpened():
            new_camera.set(3, 1280)  # Ancho
            new_camera.set(4, 720)   # Alto
            print(f"[OK] Cambiado a cámara {new_camera_id}")
            return new_camera_id, new_camera
        else:
            print(f"[ERROR] No se pudo abrir cámara {new_camera_id}")
            # Restaurar cámara anterior
            old_camera = cv2.VideoCapture(current_camera_id)
            old_camera.set(3, 1280)
            old_camera.set(4, 720)
            return current_camera_id, old_camera
    else:
        print(f"[ERROR] Cámara {new_camera_id} no disponible o ya seleccionada")
        return current_camera_id, camera_live

def create_dual_view(normal_img: np.ndarray, model: YOLO, conf: float = 0.5, filter_type: str = "nir", filter_manager: FilterManager = None) -> Tuple[np.ndarray, int, int, int]:
    """Crear vista dual: cámara normal + filtro especializado con detecciones"""
    # Usar FilterManager (requerido)
    if filter_manager:
        # Manejar pipeline especial
        if filter_type == "pipeline":
            filtered_processed, contamination_scores, pipeline_desc = filter_manager.apply_fungal_contamination_pipeline(normal_img.copy())
            filter_name = "PIPELINE COMPLETO"
            filter_desc = f"Score: {contamination_scores.get('total', 0.0):.2f}"
            
            # Mostrar reporte detallado cada 60 frames
            if hasattr(create_dual_view, 'frame_count'):
                create_dual_view.frame_count += 1
            else:
                create_dual_view.frame_count = 1
                
            if create_dual_view.frame_count % 60 == 0:  # Cada 2 segundos aprox
                print("\n" + pipeline_desc)
        else:
            filtered_processed, filter_name, filter_desc = filter_manager.apply_filter(normal_img.copy(), filter_type)
    else:
        # Error si no hay FilterManager
        print("[ERROR] FilterManager es requerido para create_dual_view")
        return normal_img, 0, 0, 0
    
    # ⚡ OPTIMIZACIÓN CRÍTICA: Solo una predicción YOLO (mejora 2x velocidad)
    # Usar solo la imagen normal para detección, aplicar filtro solo para visualización
    normal_result = model.predict(normal_img, conf=conf, verbose=False, device=device)
    # filtered_result = model.predict(filtered_processed, conf=conf, verbose=False, device=device if 'device' in globals() else 'cpu')  # ELIMINADO
    
    # Dibujar detecciones en imagen normal
    normal_with_detections = normal_img.copy()
    filtered_with_detections = filtered_processed.copy()
    
    total_castanas = 0
    sana_count = 0
    contaminada_count = 0
    
    # Procesar detecciones en imagen normal
    for result in normal_result:
        if result.boxes is not None:
            for box in result.boxes:
                label = f"{result.names[int(box.cls[0])]}"
                score = box.conf.item()
                
                # Recortar la región detectada para análisis dual
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                
                # Asegurar que las coordenadas están dentro de la imagen
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(normal_img.shape[1], x2)
                y2 = min(normal_img.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    # Recortar imagen de la castaña para análisis dual
                    crop_img = normal_img[y1:y2, x1:x2]
                    
                    # Clasificar castaña usando análisis dual
                    chestnut_type = classify_chestnut(label, score, crop_img, filter_manager)
                    
                    if chestnut_type is not None:
                        total_castanas += 1
                        if chestnut_type == 'sana':
                            sana_count += 1
                        elif chestnut_type == 'contaminada':
                            contaminada_count += 1
                        
                        color = assign_color(chestnut_type)
                        
                        # Dibujar en imagen normal
                        cv2.rectangle(normal_with_detections, (x1, y1), (x2, y2), color, 2)
                        label_text = f"{chestnut_type.upper()}: {score:.2f}"
                        cv2.putText(normal_with_detections, label_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Dibujar en imagen filtrada (mismo rectángulo, sin clasificación adicional)
                        cv2.rectangle(filtered_with_detections, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(filtered_with_detections, label_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Enviar señal al Arduino basada en el estado general del frame
    if arduino_manager and arduino_manager.enabled:
        arduino_manager.send_detection_signals(total_castanas, contaminada_count)
    
    # Crear vista dual lado a lado
    height, width = normal_with_detections.shape[:2]
    dual_view = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Colocar imágenes lado a lado
    dual_view[:, :width] = normal_with_detections
    dual_view[:, width:] = filtered_with_detections
    
    # Agregar títulos con colores distintivos
    cv2.putText(dual_view, "CAMARA NORMAL", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)  # Verde
    cv2.putText(dual_view, filter_name, (width + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)  # Magenta
    
    # Agregar subtítulos explicativos
    cv2.putText(dual_view, "(RGB Natural)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(dual_view, filter_desc, (width + 10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Línea divisoria
    cv2.line(dual_view, (width, 0), (width, height), (255, 255, 255), 2)
    
    return dual_view, total_castanas, sana_count, contaminada_count

def predict(model: YOLO, img: np.ndarray, conf: float = 0.5, filter_manager: FilterManager = None) -> Tuple[np.ndarray, Any, int, int, int]:
    """Realizar predicción y visualizar SOLO detecciones de castañas (sanas y contaminadas)"""
    
    sana_count = 0
    contaminada_count = 0
    total_castanas = 0
    
    # Realizar predicción
    results = model.predict(img, conf=conf)
    
    # Procesar resultados - SOLO castañas
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                label = f"{result.names[int(box.cls[0])]}"
                score = box.conf.item()
                
                # Recortar la región detectada para análisis RGB
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                
                # Asegurar que las coordenadas están dentro de la imagen
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                
                # Recortar imagen para análisis
                crop_img = img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                # Clasificar si es castaña y de qué tipo usando análisis RGB
                chestnut_type = classify_chestnut(label, score, crop_img, filter_manager)
                
                if chestnut_type is not None:  # Solo procesar castañas
                    total_castanas += 1
                    
                    # Contar por tipo
                    if chestnut_type == 'sana':
                        sana_count += 1
                        emoji = "[OK]"
                    elif chestnut_type == 'contaminada':
                        contaminada_count += 1
                        emoji = "[ERROR]"
                    else:
                        emoji = "[CHESTNUT]"
                    
                    # Obtener color según el tipo
                    color = assign_color(chestnut_type)
                    
                    # Dibujar rectángulo (más grueso para destacar)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=6)
                    
                    # Mostrar etiqueta con información
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label_text = f"{emoji} Castaña {chestnut_type} ({score:.2f})"
                    
                    cv2.putText(img, label_text, (x1, y1 - 10), font, 0.8, color=color, thickness=2)
    
    # Mostrar información SOLO de castañas en la imagen
    info_text = f"Castañas: {total_castanas} | Sanas: {sana_count} | Contaminadas: {contaminada_count}"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return img, results, total_castanas, sana_count, contaminada_count

def main_func():
    """Función principal para detección de castañas sanas y contaminadas"""
    print("[CHESTNUT] Detector de Castañas - Análisis Dual RGB + UV Simulado + Arduino")
    print("=" * 70)
    print("[OK] Verde: Castañas SANAS (análisis HSV + textura + simulación UV)")
    print("[ERROR] Rojo: Castañas CONTAMINADAS (moho, manchas, fluorescencia detectada)")
    print("[AI] Método: YOLO12n detecta → Análisis dual RGB + UV simulado clasifica")
    print("[TIP] UV Simulado: Detecta fluorescencia de moho y contaminación orgánica")
    print("[ARDUINO] Arduino: Activa servo cuando detecta contaminación")
    print("=" * 70)
    
    # Cargar configuración de cámaras
    camera_config = load_camera_config()
    
    # Inicializar sistema de filtros
    filter_manager = FilterManager()
    print(f"[FILTER] Sistema de filtros inicializado (GPU: {'[OK]' if filter_manager.cuda_available else '[ERROR]'})")
    
    # Cargar modelo YOLO12 preentrenado
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "core", "yolo12n.pt")
    
    print(f"[SEARCH] Cargando modelo YOLO12n: {model_path}")
    
    try:
        # Verificar disponibilidad de GPU
        import torch
        global device  # Declarar global al inicio
        print(f"[SEARCH] Verificando GPU...")
        print(f"   - PyTorch versión: {torch.__version__}")
        print(f"   - CUDA disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   - GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"   - Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device = "cuda"
            print("[FAST] Usando GPU para YOLO")
        else:
            print("[WARNING] GPU no disponible, usando CPU")
            device = "cpu"
        
        # Cargar modelo con dispositivo específico
        model = YOLO(model_path)
        
        # Mover modelo a GPU si está disponible
        if torch.cuda.is_available():
            model.to(device)
            print("[OK] Modelo YOLO12n cargado en GPU exitosamente")
        else:
            print("[OK] Modelo YOLO12n cargado en CPU exitosamente")
            
    except Exception as e:
        print(f"[ERROR] Error cargando modelo YOLO12: {e}")
        print("🔄 Intentando con modelo YOLOv8 como respaldo...")
        try:
            backup_path = os.path.join(script_dir, "core", "yolov8n.pt")
            model = YOLO(backup_path)
            
            # Mover modelo de respaldo a GPU si está disponible
            if torch.cuda.is_available():
                model.to(device)
                print("[OK] Modelo YOLOv8 de respaldo cargado en GPU exitosamente")
            else:
                print("[OK] Modelo YOLOv8 de respaldo cargado en CPU exitosamente")
        except Exception as e2:
            print(f"[ERROR] Error cargando modelo de respaldo: {e2}")
            return
    
    # Mostrar información del modelo
    print(f"📊 Información del modelo:")
    print(f"   - Dispositivo: {device.upper()}")
    print(f"   - Clases disponibles: {list(model.names.values())}")
    print(f"   - Número de clases: {len(model.names)}")
    
    # Mostrar rendimiento esperado
    if device == "cuda":
        print(f"[FAST] Rendimiento: GPU acelerado - Detección ultra-rápida")
    else:
        print(f"[WARNING] Rendimiento: CPU - Detección más lenta")
    
    # Inicializar gestor de Arduino
    global arduino_manager
    arduino_manager = init_arduino_manager()
    
    # Configuración interactiva de Arduino
    arduino_manager.interactive_setup()
    
    # Detectar cámaras disponibles con información detallada
    available_cameras = []
    camera_info = {}
    
    print("[SEARCH] Detectando cámaras disponibles...")
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Obtener información de la cámara
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Obtener nombre personalizado desde configuración
            device_name = get_camera_name(i, camera_config)
            device_description = get_camera_description(i, camera_config)
            is_favorite = is_favorite_camera(i, camera_config)
            
            camera_info[i] = {
                'name': device_name,
                'description': device_description,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'width': width,
                'height': height,
                'is_favorite': is_favorite
            }
            
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("[ERROR] No se encontraron cámaras disponibles")
        return
    
    # Mostrar información detallada de las cámaras
    print(f"\n📷 Cámaras disponibles ({len(available_cameras)} encontradas):")
    print("-" * 70)
    
    favorite_cam_id = None
    for cam_id in available_cameras:
        info = camera_info[cam_id]
        favorite_marker = " ⭐ FAVORITA" if info['is_favorite'] else ""
        print(f"   📹 Cámara {cam_id}: {info['name']}{favorite_marker}")
        
        if info['description']:
            print(f"      📝 {info['description']}")
        
        print(f"      📐 Resolución: {info['resolution']} | FPS: {info['fps']:.1f}")
        
        if info['is_favorite']:
            favorite_cam_id = cam_id
        print()
    print("-" * 70)
    
    # Seleccionar cámara (siempre preguntar al usuario)
    camera_id = None
    
    while camera_id is None:
        try:
            print(f"\n🎯 Selección de cámara:")
            print(f"   Cámaras disponibles: {available_cameras}")
            
            # Mostrar resúmen rápido de cada cámara
            for cam_id in available_cameras:
                info = camera_info[cam_id]
                favorite_marker = " ⭐" if info['is_favorite'] else ""
                print(f"   {cam_id}: {info['name']}{favorite_marker} - {info['resolution']} @ {info['fps']:.0f}fps")
            
            # Sugerir cámara favorita o por defecto
            default_cam = favorite_cam_id if favorite_cam_id else available_cameras[0]
            default_text = f"cámara favorita ({default_cam})" if favorite_cam_id else f"cámara {default_cam}"
            
            camera_choice = input(f"\nSelecciona una cámara ({available_cameras[0]}-{available_cameras[-1]}) o presiona Enter para usar la {default_text}: ").strip()
            
            if camera_choice == "":
                # Usar cámara favorita o por defecto
                camera_id = default_cam
                info = camera_info[camera_id]
                favorite_text = " (FAVORITA)" if info['is_favorite'] else ""
                print(f"[OK] Usando {default_text}: {info['name']} ({info['resolution']} @ {info['fps']:.0f}fps){favorite_text}")
            else:
                camera_id = int(camera_choice)
                if camera_id in available_cameras:
                    info = camera_info[camera_id]
                    favorite_text = " (FAVORITA)" if info['is_favorite'] else ""
                    print(f"[OK] Cámara {camera_id} seleccionada: {info['name']} ({info['resolution']} @ {info['fps']:.0f}fps){favorite_text}")
                else:
                    print(f"[ERROR] Cámara {camera_id} no disponible. Intenta con una de estas: {available_cameras}")
                    camera_id = None  # Reiniciar el bucle
                    
        except ValueError:
            print("[ERROR] Por favor ingresa un número válido")
            camera_id = None  # Reiniciar el bucle
        except (EOFError, KeyboardInterrupt):
            print("\n[WARNING] Selección cancelada. Usando cámara favorita...")
            camera_id = default_cam
            info = camera_info[camera_id]
            favorite_text = " (FAVORITA)" if info['is_favorite'] else ""
            print(f"[OK] Usando cámara por defecto: {info['name']} ({info['resolution']} @ {info['fps']:.0f}fps){favorite_text}")
    
    camera_live = cv2.VideoCapture(camera_id)
    
    if not camera_live.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_id}")
        return
    
    # Usar la resolución nativa de la cámara seleccionada
    selected_info = camera_info[camera_id]
    native_width = selected_info['width']
    native_height = selected_info['height']
    native_fps = selected_info['fps']
    
    # Intentar configurar la resolución nativa
    camera_live.set(cv2.CAP_PROP_FRAME_WIDTH, native_width)
    camera_live.set(cv2.CAP_PROP_FRAME_HEIGHT, native_height)
    camera_live.set(cv2.CAP_PROP_FPS, native_fps)
    
    # Verificar la resolución configurada
    actual_width = int(camera_live.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera_live.get(cv2.CAP_PROP_FPS)
    
    print(f"[OK] Cámara configurada:")
    print(f"   📹 Dispositivo: {selected_info['name']}")
    if selected_info['description']:
        print(f"   📝 Descripción: {selected_info['description']}")
    print(f"   📐 Resolución: {actual_width}x{actual_height}")
    print(f"   🎬 FPS: {actual_fps:.1f}")
    if selected_info['is_favorite']:
        print(f"   ⭐ Estado: CÁMARA FAVORITA")
    print("\n📋 Controles:")
    print("   - ESC: Salir")
    print("   - 'c': Cambiar confianza")
    print("   - 's': Guardar captura")
    print("   - 'i': Mostrar información del modelo")
    print("   - 'm': Cambiar cámara")
    print("   - 'f': Cambiar filtro (NIR/Spectral/Contraste/UV)")
    print("   - 'o': Optimización manual (limpiar memoria)")
    print("   - 'F': Mostrar información de filtros favoritos")
    
    frame_count = 0
    total_sanas = 0
    total_contaminadas = 0
    frames_con_castanas = 0
    confidence = 0.5
    current_filter = filter_manager.default_filter  # Filtro por defecto desde configuración
    
    # Variables para medir rendimiento
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Variables para optimización de memoria
    frame_skip_counter = 0
    frame_skip_interval = 2  # Procesar 1 de cada 2 frames para reducir carga
    memory_cleanup_interval = 300  # Limpiar memoria cada 300 frames (~10 segundos a 30fps)
    frame_count_memory = 0
    
    # Variables para control de carga del sistema
    processing_times = []
    max_processing_time = 0.05  # Máximo 50ms por frame (20 FPS mínimo)
    
    while camera_live.isOpened():
        read, frame = camera_live.read()
        if not read:
            continue
        
        frame_count += 1
        fps_counter += 1
        frame_count_memory += 1
        frame_skip_counter += 1
        
        # Medir tiempo de procesamiento
        frame_start_time = time.time()
        
        # OPTIMIZACIÓN 1: Saltar frames para reducir carga del sistema
        if frame_skip_counter < frame_skip_interval:
            # Mostrar frame anterior sin procesar
            if 'dual_view' in locals():
                cv2.imshow('[CHESTNUT] Vista Dual: Normal + UV Simulado', dual_view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
            continue
        
        frame_skip_counter = 0  # Resetear contador
        
        # OPTIMIZACIÓN 2: Limpieza periódica de memoria
        if frame_count_memory >= memory_cleanup_interval:
            import gc
            gc.collect()  # Forzar garbage collection
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()  # Limpiar cache de GPU
            frame_count_memory = 0
            print(f"[OPTIMIZACIÓN] Memoria limpiada en frame {frame_count}")
        
        # Crear vista dual: cámara normal + filtro especializado
        dual_view, total_castanas, sanas, contaminadas = create_dual_view(frame, model, conf=confidence, filter_type=current_filter, filter_manager=filter_manager)
        
        # OPTIMIZACIÓN 3: Medir tiempo de procesamiento y ajustar dinámicamente
        frame_processing_time = time.time() - frame_start_time
        processing_times.append(frame_processing_time)
        
        # Mantener solo los últimos 30 tiempos de procesamiento
        if len(processing_times) > 30:
            processing_times.pop(0)
        
        # Calcular tiempo promedio de procesamiento
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Ajustar dinámicamente el intervalo de salto de frames si el sistema está sobrecargado
        if avg_processing_time > max_processing_time:
            frame_skip_interval = min(frame_skip_interval + 1, 4)  # Máximo salto de 3 frames
            if frame_count % 60 == 0:  # Cada 2 segundos aproximadamente
                print(f"[OPTIMIZACIÓN] Sistema sobrecargado ({avg_processing_time:.3f}s), aumentando intervalo a {frame_skip_interval}")
        elif avg_processing_time < max_processing_time * 0.5 and frame_skip_interval > 1:
            frame_skip_interval = max(frame_skip_interval - 1, 1)  # Reducir intervalo si el sistema está bien
            if frame_count % 60 == 0:
                print(f"[OPTIMIZACIÓN] Sistema estable ({avg_processing_time:.3f}s), reduciendo intervalo a {frame_skip_interval}")
        
        # Calcular FPS cada 30 frames
        if fps_counter >= 30:
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            current_fps = fps_counter / elapsed_time
            fps_counter = 0
            fps_start_time = current_time
        
        # Acumular estadísticas
        total_sanas += sanas
        total_contaminadas += contaminadas
        if total_castanas > 0:
            frames_con_castanas += 1
        
        # Mostrar información adicional con FPS y rendimiento
        fps_text = f"FPS: {current_fps:.1f}" if current_fps > 0 else "FPS: Calculando..."
        processing_text = f"Proc: {avg_processing_time:.3f}s | Skip: {frame_skip_interval}"
        status_text = f"Frame: {frame_count} | {fps_text} | {processing_text} | Confianza: {confidence:.2f}"
        stats_text = f"Sanas: {total_sanas} | Contaminadas: {total_contaminadas}"
        camera_text = f"Cámara: {camera_id} | Filtro: {current_filter.upper()} | Presiona 'f' para cambiar filtro"
        
        # Información de Arduino
        if arduino_manager and arduino_manager.enabled:
            status_info = arduino_manager.get_status_info()
            arduino_status = "[ARDUINO] Arduino: CONECTADO"
            signal_color = status_info['signal_color']
            arduino_text = f"Última señal: {status_info['last_signal']}" if status_info['last_signal'] else "Esperando detección..."
            behavior_text = status_info['behavior_text']
        else:
            arduino_status = "[ARDUINO] Arduino: DESCONECTADO"
            signal_color = (128, 128, 128)
            arduino_text = "Solo detección visual"
            behavior_text = ""
        
        cv2.putText(dual_view, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(dual_view, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.putText(dual_view, stats_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(dual_view, stats_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(dual_view, camera_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(dual_view, camera_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(dual_view, arduino_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
        cv2.putText(dual_view, arduino_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, signal_color, 2)
        if behavior_text:
            cv2.putText(dual_view, behavior_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.35, signal_color, 1)
        
        # Mostrar vista dual
        cv2.imshow('[CHESTNUT] Vista Dual: Normal + UV Simulado', dual_view)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            # Cambiar confianza
            try:
                new_conf = float(input(f"\nNueva confianza (actual: {confidence:.2f}): "))
                if 0.1 <= new_conf <= 1.0:
                    confidence = new_conf
                    print(f"[OK] Confianza actualizada: {confidence:.2f}")
                else:
                    print("[ERROR] Confianza debe estar entre 0.1 y 1.0")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("[ERROR] Valor inválido o entrada cancelada")
        elif key == ord('s'):
            # Guardar captura dual
            filename = f"captura_dual_{frame_count}.jpg"
            cv2.imwrite(filename, dual_view)
            print(f"💾 Captura dual guardada: {filename}")
        elif key == ord('i'):
            # Mostrar información
            print(f"\n📊 Información actual:")
            print(f"   - Frame actual: {frame_count}")
            print(f"   - Confianza: {confidence:.2f}")
            print(f"   - Castañas SANAS detectadas: {total_sanas}")
            print(f"   - Castañas CONTAMINADAS detectadas: {total_contaminadas}")
            print(f"   - Frames con castañas: {frames_con_castanas}")
            print(f"   - Cámara actual: {camera_id}")
            print(f"   - Clases detectadas: sports ball, apple, orange, donut, bowl")
            print(f"   - IDs YOLO: 32, 47, 49, 54, 45")
            print(f"   - Filtro actual: {current_filter.upper()}")
            print(f"   - Clasificación Dual RGB + {current_filter.upper()}:")
            print(f"     • Verde (Sana): análisis HSV + textura + filtro favorable")
            print(f"     • Rojo (Contaminada): moho, manchas, contaminación detectada")
            print(f"   - Factores analizados:")
            print(f"     • RGB: Brillo (V), Saturación (S), Variación, Bordes")
            if current_filter == "nir":
                print(f"     • NIR Avanzado: Haralick, Watershed, Sobel, Morfología")
            elif current_filter == "spectral":
                print(f"     • Spectral: PCA, NDVI, Anomalías espectrales")
            elif current_filter == "contrast":
                print(f"     • Textura: LBP, Gradientes, Patrones irregulares")
            elif current_filter == "mold":
                print(f"     • Moho: Manchas oscuras, colores verdes/azules, morfología")
            elif current_filter == "rot":
                print(f"     • Podredumbre: Baja textura, amarronamiento, suavización")
            elif current_filter == "fungal":
                print(f"     • Hongos: Círculos (esporas), redes (micelio), patrones")
            elif current_filter == "mycotoxin":
                print(f"     • Micotoxinas: Decoloraciones amarronadas/verdosas, texturas granulares")
            elif current_filter == "aflatoxin":
                print(f"     • Aflatoxinas: Fluorescencia azul-verde, patrones de red (Aspergillus)")
            elif current_filter == "discoloration":
                print(f"     • Decoloración: Análisis multi-espectral, patrones de difusión")
            elif current_filter == "mold_texture":
                print(f"     • Textura Moho: Granular (Penicillium), Fibroso (Aspergillus), Algodonoso (Fusarium)")
            elif current_filter == "spore_detection":
                print(f"     • Esporas: Detección de círculos pequeños, agrupaciones, análisis de forma")
            elif current_filter == "brazil_chestnut":
                print(f"     • Castaña Brasileña: Color marrón-dorado, grietas en cáscara, manchas específicas")
            elif current_filter == "pipeline":
                print(f"     • Pipeline Completo: Combina 7 filtros relevantes para análisis exhaustivo")
                print(f"       - Normalización → Decoloración → Textura Moho → Hongos → Esporas → Micotoxinas → Aflatoxinas")
                print(f"       - Score total de contaminación con recomendaciones automáticas")
            else:
                print(f"     • UV: Fluorescencia, contraste mejorado")
        elif key == ord('m'):
            # Cambiar cámara
            print(f"\n📷 Cámaras disponibles: {available_cameras}")
            print(f"   Cámara actual: {camera_id}")
            try:
                new_camera = input(f"Selecciona una nueva cámara ({available_cameras[0]}-{available_cameras[-1]}): ").strip()
                if new_camera != "":
                    new_camera_id = int(new_camera)
                    if new_camera_id == camera_id:
                        print(f"[WARNING] Ya estás usando la cámara {camera_id}")
                    else:
                        camera_id, camera_live = switch_camera(new_camera_id, available_cameras, camera_id, camera_live)
            except (ValueError, EOFError, KeyboardInterrupt):
                print("[ERROR] Entrada inválida o cancelada")
        elif key == ord('f'):
            # Cambiar filtro usando configuración
            print(f"\n🔬 Filtros disponibles:")
            
            # Obtener filtros por categoría desde configuración
            categories = filter_manager.get_filters_by_category()
            favorite_filters = filter_manager.get_favorite_filters()
            
            filter_counter = 1
            filter_mapping = {}
            
            # Mostrar filtros por categoría
            category_names = {
                "básico": "FILTROS BÁSICOS",
                "hongos": "FILTROS PARA HONGOS", 
                "avanzado": "FILTROS AVANZADOS PARA CASTAÑAS BRASILEÑAS",
                "especializado": "FILTROS ESPECIALIZADOS",
                "pipeline": "PIPELINE AUTOMATIZADO"
            }
            
            for category, filters in categories.items():
                if filters:  # Solo mostrar categorías con filtros
                    category_display = category_names.get(category, category.upper())
                    print(f"   === {category_display} ===")
                    
                    for filter_info in filters:
                        favorite_marker = " ⭐" if filter_info["is_favorite"] else ""
                        print(f"   {filter_counter}. {filter_info['name']}{favorite_marker} - {filter_info['description']}")
                        filter_mapping[str(filter_counter)] = filter_info["type"]
                        filter_counter += 1
                    print()
            
            print(f"   Filtro actual: {current_filter.upper()}")
            if favorite_filters:
                print(f"   Filtros favoritos: {', '.join(favorite_filters)}")
            try:
                filter_choice = input(f"Selecciona filtro (1-{filter_counter-1}): ").strip()
                
                # Usar mapeo dinámico para seleccionar filtro
                if filter_choice in filter_mapping:
                    selected_filter = filter_mapping[filter_choice]
                    current_filter = selected_filter
                    
                    # Obtener información del filtro desde configuración
                    filter_info = filter_manager.get_filter_info_from_config(selected_filter)
                    filter_name = filter_info["name"]
                    filter_desc = filter_info.get("description", "")
                    
                    print(f"[OK] Cambiado a filtro {filter_name} {filter_desc}")
                    
                    # Mostrar información adicional si es pipeline
                    if selected_filter == "pipeline":
                        print("[INFO] El pipeline combina automáticamente los 7 filtros más relevantes")
                        print("[INFO] Análisis completo: normalización → decoloración → textura moho → hongos → esporas → micotoxinas → aflatoxinas")
                    
                else:
                    print("[ERROR] Opción inválida")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("[ERROR] Entrada inválida o cancelada")
        elif key == ord('F'):
            # Mostrar información de filtros favoritos
            print(f"\n⭐ Filtros favoritos configurados:")
            favorite_filters = filter_manager.get_favorite_filters()
            
            if favorite_filters:
                for filter_type in favorite_filters:
                    filter_info = filter_manager.get_filter_info_from_config(filter_type)
                    category = filter_info.get("category", "desconocido")
                    recommended = filter_info.get("recommended_for", "")
                    print(f"   • {filter_info['name']} ({filter_type})")
                    print(f"     Categoría: {category}")
                    if recommended:
                        print(f"     Recomendado para: {recommended}")
                    print()
                
                print(f"📋 Filtro por defecto: {filter_manager.default_filter}")
                print(f"💡 Para cambiar favoritos, edita filter_config.json")
            else:
                print("   No hay filtros marcados como favoritos")
                print(f"   Filtro por defecto: {filter_manager.default_filter}")
        elif key == ord('o'):
            # Optimización manual
            print(f"\n🔧 Optimización manual:")
            print(f"   - Frame actual: {frame_count}")
            print(f"   - FPS actual: {current_fps:.1f}")
            print(f"   - Tiempo promedio procesamiento: {avg_processing_time:.3f}s")
            print(f"   - Intervalo de salto: {frame_skip_interval}")
            print(f"   - Memoria GPU/CPU: Limpiando...")
            
            # Limpieza forzada de memoria
            import gc
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            
            # Resetear contadores de rendimiento
            processing_times.clear()
            frame_skip_interval = 2  # Resetear a valor por defecto
            
            print(f"   ✅ Memoria limpiada y contadores reseteados")
    
    # Limpiar recursos
    camera_live.release()
    cv2.destroyAllWindows()
    if arduino_manager:
        arduino_manager.disconnect()
    
    print(f"\n📊 Estadísticas finales:")
    print(f"   - Frames procesados: {frame_count}")
    print(f"   - Frames con castañas: {frames_con_castanas}")
    print(f"   - Total castañas SANAS: {total_sanas}")
    print(f"   - Total castañas CONTAMINADAS: {total_contaminadas}")
    print(f"   - Total castañas detectadas: {total_sanas + total_contaminadas}")
    if (total_sanas + total_contaminadas) > 0:
        print(f"   - Porcentaje de castañas sanas: {(total_sanas/(total_sanas + total_contaminadas)*100):.1f}%")
        print(f"   - Porcentaje de castañas contaminadas: {(total_contaminadas/(total_sanas + total_contaminadas)*100):.1f}%")
    print(f"   - Porcentaje de frames con detección: {(frames_con_castanas/frame_count*100):.1f}%")
    print("[OK] Detección de castañas completada")

if __name__ == "__main__":
    main_func()


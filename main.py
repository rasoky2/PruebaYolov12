import cv2
from ultralytics import YOLO
import os
import json
import numpy as np
from typing import Optional, Any, Union
import time
from arduino import ArduinoManager
from functions.analysys import analyze_apple_quality_with_logging
from utils.logger import (
    info, success, warning, error,
    arduino_info,
    detection_info, detection_error,
    camera_log, camera_ok, camera_error,
    performance_info, yolo_info,
    separator, title
)

# Variables globales para Arduino
arduino_manager = None

# Variable global para dispositivo de procesamiento
device = "cpu"  # Por defecto CPU

def init_arduino_manager():
    """Inicializar el gestor de Arduino"""
    global arduino_manager
    arduino_manager = ArduinoManager()
    return arduino_manager

def load_camera_config() -> Optional[dict[str, Any]]:
    """Cargar configuración de cámaras desde JSON"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "camera_config.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        warning("Archivo camera_config.json no encontrado. Usando configuración por defecto.")
        return None
    except json.JSONDecodeError:
        error("Error al leer camera_config.json. Usando configuración por defecto.")
        return None

def get_camera_name(cam_id: int, config: Optional[dict[str, Any]]) -> str:
    """Obtener nombre personalizado de la cámara"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)]["name"]
    return f"Dispositivo {cam_id}"

def get_camera_description(cam_id: int, config: Optional[dict[str, Any]]) -> str:
    """Obtener descripción de la cámara"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)].get("description", "")
    return ""

def is_favorite_camera(cam_id: int, config: Optional[dict[str, Any]]) -> bool:
    """Verificar si la cámara es favorita"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)].get("is_favorite", False)
    return False

def get_favorite_camera(config: Optional[dict[str, Any]]) -> Optional[int]:
    """Obtener la cámara favorita"""
    if config and "cameras" in config:
        for cam_id, cam_info in config["cameras"].items():
            if cam_info.get("is_favorite", False):
                return int(cam_id)
    return None

def assign_color(apple_type: str) -> tuple[int, int, int]:
    """Asignar colores específicos para manzanas"""
    color_map = {
        'verde': (0, 255, 0),        # Verde - Manzana verde/fresca
        'malograda': (139, 69, 19),  # Marrón - Manzana malograda/dañada
        'indeterminada': (128, 128, 128),  # Gris - No clasificada
    }
    return color_map.get(apple_type, (255, 255, 255))  # Blanco por defecto

# Funciones de análisis RGB movidas a functions/analysys.py

def classify_apple(label: str, confidence: float, crop_img: Optional[np.ndarray] = None) -> Optional[str]:
    """Clasificar manzanas en verde o malograda usando YOLO + análisis RGB"""
    label_lower = label.lower()
    
    # Usar las clases 'apple' y 'orange' para detectar manzanas
    if label_lower not in ['apple', 'orange']:
        return None  # No es una manzana o naranja
    
    # Si tenemos la imagen recortada, usar análisis RGB directo
    if crop_img is not None:
        return analyze_apple_quality_with_logging(crop_img)
    
    # Configuración de clasificación por confianza para manzanas
    if confidence >= 0.7:
        return 'verde'
    else:
        return 'malograda'  # Si la confianza es baja, probablemente esté dañada
    
    return None  # No clasificar si confianza muy baja

def switch_camera(new_camera_id: int, available_cameras: list[int], current_camera_id: int, camera_live: Any) -> tuple[int, Any]:
    """Cambiar a una nueva cámara"""
    if new_camera_id in available_cameras and new_camera_id != current_camera_id:
        # Cerrar cámara actual
        camera_live.release()
        
        # Abrir nueva cámara
        new_camera = cv2.VideoCapture(new_camera_id)
        if new_camera.isOpened():
            new_camera.set(3, 1280)  # Ancho
            new_camera.set(4, 720)   # Alto
            camera_ok(f"Cambiado a cámara {new_camera_id}")
            return new_camera_id, new_camera
        else:
            camera_error(f"No se pudo abrir cámara {new_camera_id}")
            # Restaurar cámara anterior
            old_camera = cv2.VideoCapture(current_camera_id)
            old_camera.set(3, 1280)
            old_camera.set(4, 720)
            return current_camera_id, old_camera
    else:
        camera_error(f"Cámara {new_camera_id} no disponible o ya seleccionada")
        return current_camera_id, camera_live

def create_simple_view(img: np.ndarray, model: YOLO, conf: float = 0.5) -> tuple[np.ndarray, int, int, int]:
    """Crear vista simple con detección específica de manzanas"""
    
    # Realizar predicción YOLO
    results = model.predict(img, conf=conf, verbose=False, device=device)
    
    # Crear copia de la imagen para dibujar
    img_with_detections = img.copy()
    
    total_manzanas = 0
    verde_count = 0
    malograda_count = 0
    
    # Procesar detecciones
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Obtener información de la detección
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                score = box.conf.item()
                
                # Coordenadas del bounding box
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                
                # Asegurar que las coordenadas están dentro de la imagen
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    # Recortar imagen de la manzana para análisis de calidad
                    crop_img = img[y1:y2, x1:x2]
                    
                    # Procesar detecciones de manzanas (apple y orange)
                    if class_name.lower() in ['apple', 'orange']:
                        # Usar análisis RGB para determinar el estado de la manzana
                        apple_quality = analyze_apple_quality_with_logging(crop_img)
                        
                        total_manzanas += 1
                        if apple_quality == 'verde':
                            verde_count += 1
                            color = assign_color('verde')
                            label_text = f"MANZANA VERDE: {score:.2f}"
                        elif apple_quality == 'malograda':
                            malograda_count += 1
                            color = assign_color('malograda')
                            label_text = f"MANZANA MALOGRADA: {score:.2f}"
                        else:
                            color = assign_color('indeterminada')
                            label_text = f"MANZANA INDETERMINADA: {score:.2f}"
                        
                        # Dibujar rectángulo y etiqueta
                        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(img_with_detections, label_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Agregar información adicional si está malograda
                        if apple_quality == 'malograda':
                            cv2.putText(img_with_detections, "Malograda detectada", (x1, y2+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    # Si no es una manzana, no procesar
    
    # Enviar señal al Arduino basada en el estado general del frame
    if arduino_manager and arduino_manager.enabled:
        arduino_manager.send_detection_signals(total_manzanas, malograda_count)
    
    return img_with_detections, total_manzanas, verde_count, malograda_count

def predict(model: YOLO, img: np.ndarray, conf: float = 0.5) -> tuple[np.ndarray, Any, int, int]:
    """Realizar predicción y visualizar SOLO detecciones de manzanas (verde, malograda)"""
    
    verde_count = 0
    malograda_count = 0
    total_manzanas = 0
    
    # Realizar predicción
    results = model.predict(img, conf=conf)
    
    # Procesar resultados - SOLO manzanas
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
                
                # Clasificar si es manzana y de qué tipo usando análisis RGB directo
                apple_type = classify_apple(label, score, crop_img)
                
                if apple_type is not None:  # Solo procesar manzanas
                    total_manzanas += 1
                    
                    # Contar por tipo
                    if apple_type == 'verde':
                        verde_count += 1
                        emoji = "[VERDE]"
                    elif apple_type == 'malograda':
                        malograda_count += 1
                        emoji = "[MALOGRADA]"
                    else:
                        emoji = "[MANZANA]"
                    
                    # Obtener color según el tipo
                    color = assign_color(apple_type)
                    
                    # Dibujar rectángulo (más grueso para destacar)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=6)
                    
                    # Mostrar etiqueta con información
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label_text = f"{emoji} Manzana {apple_type} ({score:.2f})"
                    
                    cv2.putText(img, label_text, (x1, y1 - 10), font, 0.8, color=color, thickness=2)
    
    # Mostrar información SOLO de manzanas en la imagen
    info_text = f"Manzanas: {total_manzanas} | Verdes: {verde_count} | Malogradas: {malograda_count}"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return img, results, total_manzanas, verde_count

def main_func():
    """Función principal para detección de manzanas con interfaz gráfica"""
    title("Detector de Manzanas - YOLO12n + Análisis RGB")
    detection_info("Verde: Manzanas VERDES (frescas y saludables)")
    detection_error("Marrón: Manzanas MALOGRADAS (dañadas, podridas o arrugadas)")
    detection_error("Método: YOLO12n detecta clase 'apple' → Análisis RGB + Textura")
    yolo_info("Clases detectadas: apple, orange (para manzanas arrugadas)")
    info("Análisis: RGB + detección de arrugas (bordes y textura)")
    
    # Intentar abrir interfaz gráfica predeterminadamente
    try:
        info("Iniciando interfaz gráfica...")
        import interface
        interface.main()
        return
    except ImportError as e:
        warning(f"No se pudo cargar interfaz gráfica: {e}")
        warning("Continuando con modo consola...")
    except Exception as e:
        error(f"Error en interfaz gráfica: {e}")
        warning("Continuando con modo consola...")
    
    # Cargar configuración de cámaras
    camera_config = load_camera_config()
    
    # Cargar modelo YOLO12n preentrenado (sin entrenamiento específico)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "core", "yolo12n.pt")
    
    yolo_info(f"Cargando modelo YOLO12n: {model_path}")
    
    try:
        # Verificar disponibilidad de GPU
        import torch
        global device  # Declarar global al inicio
        info("Verificando GPU...")
        info(f"   - PyTorch versión: {torch.__version__}")
        info(f"   - CUDA disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            info(f"   - GPU detectada: {torch.cuda.get_device_name(0)}")
            info(f"   - Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device = "cuda"
            performance_info("Usando GPU para YOLO")
        else:
            warning("GPU no disponible, usando CPU")
            device = "cpu"
        
        # Cargar modelo YOLO12n preentrenado
        model = YOLO(model_path)
        
        # Mover modelo a GPU si está disponible
        if torch.cuda.is_available():
            model.to(device)
            yolo_info("Modelo YOLO12n cargado en GPU exitosamente")
        else:
            yolo_info("Modelo YOLO12n cargado en CPU exitosamente")
            
    except Exception as e:
        error(f"Error cargando modelo YOLO12: {e}")
        info("Intentando con modelo YOLOv8 como respaldo...")
        try:
            backup_path = os.path.join(script_dir, "core", "yolov8n.pt")
            model = YOLO(backup_path)
            
            # Mover modelo de respaldo a GPU si está disponible
            if torch.cuda.is_available():
                model.to(device)
                yolo_info("Modelo YOLOv8 de respaldo cargado en GPU exitosamente")
            else:
                yolo_info("Modelo YOLOv8 de respaldo cargado en CPU exitosamente")
        except Exception as e2:
            error(f"Error cargando modelo de respaldo: {e2}")
            return
    
    # Mostrar información del modelo
    info("Información del modelo:")
    info(f"   - Dispositivo: {device.upper()}")
    info(f"   - Clases disponibles: {list(model.names.values())}")
    info(f"   - Número de clases: {len(model.names)}")
    
    # Mostrar rendimiento esperado
    if device == "cuda":
        performance_info("Rendimiento: GPU acelerado - Detección ultra-rápida")
    else:
        warning("Rendimiento: CPU - Detección más lenta")
    
    # Inicializar gestor de Arduino
    global arduino_manager
    arduino_manager = init_arduino_manager()
    
    # Configuración interactiva de Arduino
    arduino_manager.interactive_setup()
    
    # Detectar cámaras disponibles con información detallada
    available_cameras = []
    camera_info = {}
    
    info("Detectando cámaras disponibles...")
    
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
        camera_error("No se encontraron cámaras disponibles")
        return
    
    # Mostrar información detallada de las cámaras
    info(f"\nCámaras disponibles ({len(available_cameras)} encontradas):")
    separator("-", 70, "CYAN")
    
    favorite_cam_id = None
    for cam_id in available_cameras:
        cam_info = camera_info[cam_id]
        favorite_marker = " (FAVORITA)" if cam_info['is_favorite'] else ""
        camera_log(f"Cámara {cam_id}: {cam_info['name']}{favorite_marker}")
        
        if cam_info['description']:
            info(f"      {cam_info['description']}")
        
        info(f"      Resolución: {cam_info['resolution']} | FPS: {cam_info['fps']:.1f}")
        
        if cam_info['is_favorite']:
            favorite_cam_id = cam_id
        print()
    separator("-", 70, "CYAN")
    
    # Seleccionar cámara (siempre preguntar al usuario)
    camera_id = None
    
    while camera_id is None:
        try:
            info("\nSelección de cámara:")
            info(f"   Cámaras disponibles: {available_cameras}")
            
            # Mostrar resúmen rápido de cada cámara
            for cam_id in available_cameras:
                cam_info = camera_info[cam_id]
                favorite_marker = " (FAVORITA)" if cam_info['is_favorite'] else ""
                info(f"   {cam_id}: {cam_info['name']}{favorite_marker} - {cam_info['resolution']} @ {cam_info['fps']:.0f}fps")
            
            # Sugerir cámara favorita o por defecto
            default_cam = favorite_cam_id if favorite_cam_id else available_cameras[0]
            default_text = f"cámara favorita ({default_cam})" if favorite_cam_id else f"cámara {default_cam}"
            
            camera_choice = input(f"\nSelecciona una cámara ({available_cameras[0]}-{available_cameras[-1]}) o presiona Enter para usar la {default_text}: ").strip()
            
            if camera_choice == "":
                # Usar cámara favorita o por defecto 
                camera_id = default_cam
                cam_info = camera_info[camera_id]
                favorite_text = " (FAVORITA)" if cam_info['is_favorite'] else ""
                camera_ok(f"Usando {default_text}: {cam_info['name']} ({cam_info['resolution']} @ {cam_info['fps']:.0f}fps){favorite_text}")
            else:
                camera_id = int(camera_choice)
                if camera_id in available_cameras:
                    cam_info = camera_info[camera_id]
                    favorite_text = " (FAVORITA)" if cam_info['is_favorite'] else ""
                    camera_ok(f"Cámara {camera_id} seleccionada: {cam_info['name']} ({cam_info['resolution']} @ {cam_info['fps']:.0f}fps){favorite_text}")
                else:
                    camera_error(f"Cámara {camera_id} no disponible. Intenta con una de estas: {available_cameras}")
                    camera_id = None  # Reiniciar el bucle
                    
        except ValueError:
            error("Por favor ingresa un número válido")
            camera_id = None  # Reiniciar el bucle
        except (EOFError, KeyboardInterrupt):
            warning("\nSelección cancelada. Usando cámara favorita...")
            camera_id = default_cam
            cam_info = camera_info[camera_id]
            favorite_text = " (FAVORITA)" if cam_info['is_favorite'] else ""
            camera_ok(f"Usando cámara por defecto: {cam_info['name']} ({cam_info['resolution']} @ {cam_info['fps']:.0f}fps){favorite_text}")
    
    camera_live = cv2.VideoCapture(camera_id)
    
    if not camera_live.isOpened():
        camera_error(f"No se pudo abrir la cámara {camera_id}")
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
    
    camera_ok("Cámara configurada:")
    info(f"   Dispositivo: {selected_info['name']}")
    if selected_info['description']:
        info(f"   Descripción: {selected_info['description']}")
    info(f"   Resolución: {actual_width}x{actual_height}")
    info(f"   FPS: {actual_fps:.1f}")
    if selected_info['is_favorite']:
        info("   Estado: CÁMARA FAVORITA")
    
    info("\nControles:")
    info("   - ESC: Salir")
    info("   - 'c': Cambiar confianza")
    info("   - 's': Guardar captura")
    info("   - 'i': Mostrar información del modelo")
    info("   - 'm': Cambiar cámara")
    info("   - 'o': Optimización manual (limpiar memoria)")
    
    frame_count = 0
    total_verdes = 0
    total_malogradas = 0
    frames_con_manzanas = 0
    confidence = 0.5
    
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
            if 'processed_frame' in locals():
                cv2.imshow('[CHESTNUT] Detector de Castañas - Análisis RGB', processed_frame)
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
            performance_info(f"Memoria limpiada en frame {frame_count}")
        
        # Crear vista simple con análisis RGB directo
        processed_frame, total_manzanas, verdes, malogradas = create_simple_view(frame, model, conf=confidence)
        
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
                performance_info(f"Sistema sobrecargado ({avg_processing_time:.3f}s), aumentando intervalo a {frame_skip_interval}")
        elif avg_processing_time < max_processing_time * 0.5 and frame_skip_interval > 1:
            frame_skip_interval = max(frame_skip_interval - 1, 1)  # Reducir intervalo si el sistema está bien
            if frame_count % 60 == 0:
                performance_info(f"Sistema estable ({avg_processing_time:.3f}s), reduciendo intervalo a {frame_skip_interval}")
        
        # Calcular FPS cada 30 frames
        if fps_counter >= 30:
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            current_fps = fps_counter / elapsed_time
            fps_counter = 0
            fps_start_time = current_time
        
        # Acumular estadísticas
        total_verdes += verdes
        total_malogradas += malogradas
        if total_manzanas > 0:
            frames_con_manzanas += 1
        
        # Mostrar información adicional con FPS y rendimiento
        fps_text = f"FPS: {current_fps:.1f}" if current_fps > 0 else "FPS: Calculando..."
        processing_text = f"Proc: {avg_processing_time:.3f}s | Skip: {frame_skip_interval}"
        status_text = f"Frame: {frame_count} | {fps_text} | {processing_text} | Confianza: {confidence:.2f}"
        stats_text = f"Verdes: {total_verdes} | Malogradas: {total_malogradas}"
        camera_text = f"Cámara: {camera_id} | Modelo: YOLO12n"
        
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
        
        cv2.putText(processed_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(processed_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.putText(processed_frame, stats_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(processed_frame, stats_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(processed_frame, camera_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(processed_frame, camera_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(processed_frame, arduino_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
        cv2.putText(processed_frame, arduino_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, signal_color, 2)
        if behavior_text:
            cv2.putText(processed_frame, behavior_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.35, signal_color, 1)
        
        # Mostrar vista simple
        cv2.imshow('[APPLE] Detector de Manzanas - YOLO12n', processed_frame)
        
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
                    success(f"Confianza actualizada: {confidence:.2f}")
                else:
                    error("Confianza debe estar entre 0.1 y 1.0")
            except (ValueError, EOFError, KeyboardInterrupt):
                error("Valor inválido o entrada cancelada")
        elif key == ord('s'):
            # Guardar captura
            filename = f"captura_{frame_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            success(f"Captura guardada: {filename}")
        elif key == ord('i'):
            # Mostrar información
            info(f"\nInformación actual:")
            info(f"   - Frame actual: {frame_count}")
            info(f"   - Confianza: {confidence:.2f}")
            info(f"   - Manzanas VERDES detectadas: {total_verdes}")
            info(f"   - Manzanas MALOGRADAS detectadas: {total_malogradas}")
            info(f"   - Frames con manzanas: {frames_con_manzanas}")
            info(f"   - Cámara actual: {camera_id}")
            info("   - Clases detectadas: apple, orange")
            info("   - ID YOLO: 47")
            info("   - Modelo: YOLO12n preentrenado (sin entrenamiento específico)")
            info("   - Clasificación de Calidad:")
            info("     • Verde: Manzanas frescas y saludables (RGB + sin arrugas)")
            info("     • Marrón: Manzanas malogradas/dañadas/arrugadas (RGB + textura)")
            info("   - Factores analizados:")
            info("     • RGB: Verde (140-220, 160-240, 80-150) + verde dominante")
            info("     • TEXTURA: Detección de arrugas (bordes Canny + gradientes)")
            info("     • Arrugas: Bordes >0.08+variación >80+gradientes >30, o bordes >0.12, o variación >100")
        elif key == ord('m'):
            # Cambiar cámara
            info(f"\nCámaras disponibles: {available_cameras}")
            info(f"   Cámara actual: {camera_id}")
            try:
                new_camera = input(f"Selecciona una nueva cámara ({available_cameras[0]}-{available_cameras[-1]}): ").strip()
                if new_camera != "":
                    new_camera_id = int(new_camera)
                    if new_camera_id == camera_id:
                        warning(f"Ya estás usando la cámara {camera_id}")
                    else:
                        camera_id, camera_live = switch_camera(new_camera_id, available_cameras, camera_id, camera_live)
            except (ValueError, EOFError, KeyboardInterrupt):
                error("Entrada inválida o cancelada")
        elif key == ord('o'):
            # Optimización manual
            info("\nOptimización manual:")
            info(f"   - Frame actual: {frame_count}")
            info(f"   - FPS actual: {current_fps:.1f}")
            info(f"   - Tiempo promedio procesamiento: {avg_processing_time:.3f}s")
            info(f"   - Intervalo de salto: {frame_skip_interval}")
            info("   - Memoria GPU/CPU: Limpiando...")
            
            # Limpieza forzada de memoria
            import gc
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            
            # Resetear contadores de rendimiento
            processing_times.clear()
            frame_skip_interval = 2  # Resetear a valor por defecto
            
            success("   ✅ Memoria limpiada y contadores reseteados")
        
    
    # Limpiar recursos
    camera_live.release()
    cv2.destroyAllWindows()
    if arduino_manager:
        arduino_manager.disconnect()
    
    info("\nEstadísticas finales:")
    info(f"   - Frames procesados: {frame_count}")
    info(f"   - Frames con manzanas: {frames_con_manzanas}")
    info(f"   - Total manzanas VERDES: {total_verdes}")
    info(f"   - Total manzanas MALOGRADAS: {total_malogradas}")
    info(f"   - Total manzanas detectadas: {total_verdes + total_malogradas}")
    if (total_verdes + total_malogradas) > 0:
        info(f"   - Porcentaje de manzanas verdes: {(total_verdes/(total_verdes + total_malogradas)*100):.1f}%")
        info(f"   - Porcentaje de manzanas malogradas: {(total_malogradas/(total_verdes + total_malogradas)*100):.1f}%")
    info(f"   - Porcentaje de frames con detección: {(frames_con_manzanas/frame_count*100):.1f}%")
    success("Detección de manzanas completada")

if __name__ == "__main__":
    main_func()


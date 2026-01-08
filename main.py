import json
import os
from typing import Any
from utils.logger import (
    detection_error,
    detection_info,
    error,
    info,
    title,
    warning,
    yolo_info,
)

# Patrón singleton para YOLO (Context7 best practice)
class YOLOModelManager:
    """Gestor singleton para modelos YOLO - Optimización Context7"""
    _instance = None
    _model = None
    _model_path = None
    _config_manager = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_config_manager(self):
        """Obtener gestor de configuración"""
        if self._config_manager is None:
            from functions.config_manager import get_config_manager
            self._config_manager = get_config_manager()
        return self._config_manager
    
    def get_model(self, model_path: str = None):
        """Obtener modelo YOLO con carga optimizada desde configuración"""
        if model_path is None:
            # Usar configuración del JSON
            config_manager = self._get_config_manager()
            yolo_config = config_manager.get_yolo_config()
            model_path = yolo_config.get("selected_model", "yolo12n.pt")
        
        if self._model is None or self._model_path != model_path:
            try:
                from ultralytics import YOLO
                self._model = YOLO(model_path, task="detect")
                self._model_path = model_path
                info(f"Modelo YOLO cargado: {model_path}")
            except ImportError as e:
                error(f"Error cargando YOLO: {e}")
                return None
        return self._model
    
    def predict_optimized(self, source, conf=None, iou=None, imgsz=None):
        """Predicción optimizada con parámetros desde configuración JSON"""
        if self._model is None:
            return None
        
        # Obtener parámetros desde configuración si no se especifican
        if conf is None or iou is None or imgsz is None:
            config_manager = self._get_config_manager()
            yolo_config = config_manager.get_yolo_config()
            conf = conf or yolo_config.get("confidence_threshold", 0.75)
            iou = iou or yolo_config.get("iou_threshold", 0.45)
            imgsz = imgsz or yolo_config.get("image_size", 640)
        
        try:
            results = self._model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                save=False,
                verbose=False
            )
            return results
        except Exception as e:
            error(f"Error en predicción YOLO: {e}")
            return None


# Instancia global del gestor YOLO
yolo_manager = YOLOModelManager()


def load_camera_config() -> dict[str, Any] | None:
    """Cargar configuración de cámaras desde interface_config.json"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "interface_config.json")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            # Extraer solo la configuración de cámaras
            return {
                "cameras": config.get("cameras", {}),
                "settings": config.get("settings", {}),
            }
    except FileNotFoundError:
        warning("Archivo interface_config.json no encontrado. Usando configuración por defecto.")
        return None
    except json.JSONDecodeError:
        error("Error al leer interface_config.json. Usando configuración por defecto.")
        return None


def get_camera_name(cam_id: int, config: dict[str, Any] | None) -> str:
    """Obtener nombre personalizado de la cámara"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)]["name"]
    return f"Dispositivo {cam_id}"


def get_camera_description(cam_id: int, config: dict[str, Any] | None) -> str:
    """Obtener descripción de la cámara"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)].get("description", "")
    return ""


def is_favorite_camera(cam_id: int, config: dict[str, Any] | None) -> bool:
    """Verificar si la cámara es favorita"""
    if config and "cameras" in config and str(cam_id) in config["cameras"]:
        return config["cameras"][str(cam_id)].get("is_favorite", False)
    return False


def get_favorite_camera(config: dict[str, Any] | None) -> int | None:
    """Obtener la cámara favorita"""
    if config and "cameras" in config:
        for cam_id, cam_info in config["cameras"].items():
            if cam_info.get("is_favorite", False):
                return int(cam_id)
    return None

# Todas las funciones de detección y análisis movidas a interface.py


def main_func():
    """Función principal para detección de manzanas con interfaz gráfica"""
    title("Detector de Manzanas - YOLO12n + Análisis RGB")
    detection_info("Verde: Manzanas VERDES (frescas y saludables)")
    detection_error("Marrón: Manzanas MALOGRADAS (dañadas, podridas o arrugadas)")
    detection_error("Método: YOLO12n detecta clase 'apple' → Análisis RGB + Textura")
    yolo_info("Clases detectadas: apple, orange (para manzanas arrugadas)")
    info("Análisis: RGB + detección de arrugas (bordes y textura)")

    # Cargar interfaz gráfica (PyQt6)
    try:
        info("Iniciando interfaz gráfica (PyQt6)...")
        import interface
        interface.main()
    except Exception as e:
        error(f"ERROR CRÍTICO: Error en interfaz gráfica: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main_func()

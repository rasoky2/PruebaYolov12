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

# Variables globales (ahora manejadas por interface.py)


def load_camera_config() -> dict[str, Any] | None:
    """Cargar configuración de cámaras desde JSON"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "camera_config.json")

    try:
        with open(config_path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        warning("Archivo camera_config.json no encontrado. Usando configuración por defecto.")
        return None
    except json.JSONDecodeError:
        error("Error al leer camera_config.json. Usando configuración por defecto.")
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

    # Cargar interfaz gráfica (OBLIGATORIO)
    try:
        info("Iniciando interfaz gráfica...")
        import interface
        interface.main()
        return
    except ImportError as e:
        error(f"ERROR CRÍTICO: No se pudo cargar interfaz gráfica: {e}")
        error("La interfaz gráfica es obligatoria. Verifica que tkinter esté instalado.")
        return
    except Exception as e:
        error(f"ERROR CRÍTICO: Error en interfaz gráfica: {e}")
        error("La interfaz gráfica es obligatoria. Verifica la instalación.")
        return


if __name__ == "__main__":
    main_func()

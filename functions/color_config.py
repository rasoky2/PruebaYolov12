"""
Módulo para manejo de configuración de clasificación de colores
Carga y procesa reglas de clasificación desde archivos JSON/YAML
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ColorClassificationConfig:
    """Gestor de configuración de clasificación de colores"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._cache = {}  # Cache para evaluaciones frecuentes

    def _get_default_config_path(self) -> str:
        """Obtener ruta por defecto del archivo de configuración"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(current_dir, "config", "color_classification.json")
    
    def _load_config(self) -> dict[str, Any]:
        """Cargar configuración desde archivo JSON"""
        try:
            with open(self.config_path, encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Configuración de colores cargada desde: {self.config_path}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error cargando configuración de colores: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict[str, Any]:
        """Configuración por defecto si falla la carga"""
        return {
            "apple": {
                "sana": {
                    "ranges": [
                        {"name": "verde_dominante", "condition": "g == max_channel && g > 100", "priority": 1},
                        {"name": "verde_fresco", "r_min": 120, "r_max": 240, "g_min": 140, "g_max": 255, "b_min": 60, "b_max": 180, "priority": 2}
                    ]
                },
                "contaminada": {
                    "ranges": [
                        {"name": "colores_muy_oscuros", "r_max": 100, "g_max": 100, "b_max": 80, "priority": 1}
                    ]
                }
            },
            "thresholds": {
                "dark_spots": {"threshold": 120, "ratio_threshold": 0.35}
            }
        }
    
    def get_object_config(self, object_class: str) -> dict[str, Any]:
        """Obtener configuración para un tipo de objeto específico"""
        return self.config.get(object_class.lower(), self.config.get("generic", {}))
    
    def get_thresholds(self) -> dict[str, Any]:
        """Obtener umbrales de detección"""
        return self.config.get("thresholds", {})
    
    def evaluate_color_ranges(self, rgb_color: tuple[int, int, int], object_class: str, quality_type: str) -> bool:
        """
        Evaluar si un color RGB cumple con los rangos de un tipo de calidad específico

        Args:
            rgb_color: Color RGB (R, G, B)
            object_class: Tipo de objeto ('apple', 'orange', etc.)
            quality_type: Tipo de calidad ('sana', 'contaminada')

        Returns:
            bool: True si cumple con los rangos
        """
        r, g, b = rgb_color
        object_config = self.get_object_config(object_class)
        quality_config = object_config.get(quality_type, {})
        ranges = quality_config.get("ranges", [])
        
        # Ordenar por prioridad (menor número = mayor prioridad)
        ranges = sorted(ranges, key=lambda x: x.get("priority", 999))

        for range_config in ranges:
            if self._evaluate_single_range(r, g, b, range_config):
                return True
        
        return False
    
    def _evaluate_single_range(self, r: int, g: int, b: int, range_config: dict[str, Any]) -> bool:
        """Evaluar un rango específico de color"""
        # Evaluar condiciones simples de rango
        if "r_min" in range_config and r < range_config["r_min"]:
            return False
        if "r_max" in range_config and r > range_config["r_max"]:
            return False
        if "g_min" in range_config and g < range_config["g_min"]:
            return False
        if "g_max" in range_config and g > range_config["g_max"]:
            return False
        if "b_min" in range_config and b < range_config["b_min"]:
            return False
        if "b_max" in range_config and b > range_config["b_max"]:
            return False
        
        # Evaluar condiciones complejas si existen
        if "condition" in range_config:
            if not self._evaluate_condition(r, g, b, range_config["condition"]):
                return False
        
        return True
    
    def _evaluate_condition(self, r: int, g: int, b: int, condition: str) -> bool:
        """Evaluar condición compleja de color"""
        try:
            # Reemplazar variables en la condición
            condition = condition.replace("r", str(r))
            condition = condition.replace("g", str(g))
            condition = condition.replace("b", str(b))
            condition = condition.replace("max_channel", str(max(r, g, b)))
            
            # Convertir operadores de C/JavaScript a Python
            condition = condition.replace("&&", "and")
            condition = condition.replace("||", "or")
            condition = condition.replace("==", "==")
            condition = condition.replace("!=", "!=")
            
            # Evaluar la condición (usar eval con precaución)
            return bool(eval(condition))
        except Exception as e:
            logger.warning(f"Error evaluando condición '{condition}': {e}")
            return False

    def classify_color(self, rgb_color: tuple[int, int, int], object_class: str) -> str:
        """
        Clasificar color usando configuración externa
        
        Args:
            rgb_color: Color RGB (R, G, B)
            object_class: Tipo de objeto

        Returns:
            str: 'sana', 'contaminada' o 'indeterminada'
        """
        # Primero verificar si es contaminada (mayor prioridad)
        if self.evaluate_color_ranges(rgb_color, object_class, "contaminada"):
            return "contaminada"

        # Luego verificar si es sana
        if self.evaluate_color_ranges(rgb_color, object_class, "sana"):
            return "sana"

        return "indeterminada"

    def get_threshold(self, category: str, threshold_name: str, default_value: Any = None) -> Any:
        """Obtener umbral específico de configuración"""
        thresholds = self.get_thresholds()
        category_thresholds = thresholds.get(category, {})
        return category_thresholds.get(threshold_name, default_value)

    def reload_config(self):
        """Recargar configuración desde archivo"""
        self.config = self._load_config()
        self._cache.clear()
        logger.info("Configuración de colores recargada")

    def get_available_object_classes(self) -> list[str]:
        """Obtener lista de clases de objetos disponibles"""
        return [key for key in self.config.keys() if key != "thresholds"]

    def get_quality_types(self, object_class: str) -> list[str]:
        """Obtener tipos de calidad disponibles para un objeto"""
        object_config = self.get_object_config(object_class)
        return list(object_config.keys())


# Instancia global del gestor de configuración
color_config = ColorClassificationConfig()


def get_color_config() -> ColorClassificationConfig:
    """Obtener instancia global del gestor de configuración"""
    return color_config


def reload_color_config():
    """Recargar configuración de colores"""
    color_config.reload_config()

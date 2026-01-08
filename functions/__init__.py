"""
Paquete de funciones auxiliares para detección de manzanas
"""

from .analysys import (
    # Funciones del sistema consolidado optimizado
    analyze_object_quality_consolidated,
    analyze_rgb_vectorized,
    calculate_color_confidence,
    classify_object_by_color_and_type,
    detect_wrinkled_texture_optimized,
    get_dominant_rgb_color_from_rgb,
    analyze_color_distribution_from_rgb,
)

from .config_manager import (
    # Gestor de configuración principal
    get_config_manager,
    get_yolo_config,
    get_camera_config,
    get_processing_config,
    get_setting,
    set_setting,
    save_config,
    reload_config,
)


__all__ = [
    # Funciones del sistema consolidado optimizado
    "analyze_object_quality_consolidated",
    "analyze_rgb_vectorized",
    "calculate_color_confidence",
    "classify_object_by_color_and_type",
    "detect_wrinkled_texture_optimized",
    "get_dominant_rgb_color_from_rgb",
    "analyze_color_distribution_from_rgb",
    # Gestor de configuración principal
    "get_config_manager",
    "get_yolo_config",
    "get_camera_config",
    "get_processing_config",
    "get_setting",
    "set_setting",
    "save_config",
    "reload_config",
]

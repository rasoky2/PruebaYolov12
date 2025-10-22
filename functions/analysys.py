"""
Análisis RGB para clasificación de calidad de manzanas
Sistema de detección de estado de madurez y daño basado en análisis de color RGB
"""

import json
import logging
import os
from typing import Any

import cv2
import numpy as np

from .color_config import get_color_config
from .image_cache import get_image_cache

# Configurar logger optimizado
logger = logging.getLogger(__name__)


# Sistema de logging optimizado para análisis frecuente
class AnalysisLogger:
    """Logger optimizado para análisis de objetos que se ejecuta frecuentemente"""
    def __init__(self):
        self.analysis_count = 0
        self.log_interval = 50  # Log cada 50 análisis
        self.buffered_analyses = []
        self.max_buffer = 20

    def log_analysis(self, message, force=False):
        """Log de análisis optimizado"""
        self.analysis_count += 1
        if force or self.analysis_count % self.log_interval == 0:
            logger.info(f"[ANÁLISIS] {message}")

    def buffer_analysis(self, analysis_type, details):
        """Bufferizar análisis para procesamiento en lote"""
        self.buffered_analyses.append((analysis_type, details))
        if len(self.buffered_analyses) >= self.max_buffer:
            self.flush_analyses()

    def flush_analyses(self):
        """Procesar análisis en buffer"""
        if self.buffered_analyses:
            summary = {}
            for analysis_type, _details in self.buffered_analyses:
                if analysis_type not in summary:
                    summary[analysis_type] = 0
                summary[analysis_type] += 1

            for analysis_type, count in summary.items():
                logger.info(f"[RESUMEN] {analysis_type}: {count} detecciones")
            self.buffered_analyses.clear()


# Instancia global del logger de análisis
analysis_logger = AnalysisLogger()

# Cache global para configuración (optimización de I/O)
_detection_classes_cache = None


def load_interface_config() -> dict:
    """Cargar configuración desde interface_config.json"""
    try:
        # Obtener la ruta del archivo de configuración
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(current_dir, "interface_config.json")

        with open(config_path, encoding='utf-8') as f:
            return json.load(f)
            # print(f"✅ Configuración cargada desde: {config_path}")  # Log reducido
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error cargando interface_config.json: {e}")
        logger.info("Usando configuración por defecto")
        return {
            "detection_classes": ["apple", "orange"],
            "colors": {
                "sana": "#00FF00",
                "contaminada": "#FF0000",
                "detectada": "#FFFF00"
            }
        }


def get_detection_classes() -> list[str]:
    """Obtener las clases de detección desde la configuración (OPTIMIZADO)"""
    global _detection_classes_cache
    if _detection_classes_cache is None:
        config = load_interface_config()
        _detection_classes_cache = config.get("detection_classes", ["apple", "orange"])
    return _detection_classes_cache


def analyze_object_quality(crop_img: np.ndarray | None, object_class: str = "apple") -> str:
    """
    Analizar calidad de objeto detectado usando análisis RGB + detección de textura + huecos
    VERSIÓN OPTIMIZADA: Reutiliza conversiones BGR→RGB y redimensionamientos

    Args:
        crop_img: Imagen recortada del objeto detectado
        object_class: Tipo de objeto ('apple', 'orange', 'sports ball', etc.)

    Returns:
        str: 'sana', 'contaminada' o 'indeterminada'
    """
    if crop_img is None or crop_img.size == 0:
        return 'indeterminada'

    # OPTIMIZACIÓN: Análisis consolidado que reutiliza conversiones
    analysis_data = analyze_object_quality_consolidated(crop_img, object_class)

    # PRIORIDAD 1: Detectar huecos/agujeros (daño severo)
    if analysis_data['hole_analysis']['has_holes'] and analysis_data['hole_analysis']['hole_confidence'] > 0.9:
        return 'contaminada'

    # PRIORIDAD 2: Detectar textura dañada/arrugada
    if analysis_data['texture_analysis']['is_wrinkled']:
        return 'contaminada'

    # PRIORIDAD 3: Detectar manchas oscuras/podredumbre
    if analysis_data['dark_spots']:
        return 'contaminada'

    # PRIORIDAD 4: Detectar rayas verticales de podredumbre
    if analysis_data['rot_streaks']:
        return 'contaminada'

    # PRIORIDAD 5: Análisis RGB específico por tipo de objeto
    color_result = analysis_data['color_result']

    # Debug reducido: Solo mostrar análisis si hay contaminación detectada
    if color_result in ['contaminada', 'detectada']:
        analysis_logger.buffer_analysis(f"{object_class.upper()}_{color_result.upper()}", {
            'rgb': analysis_data['dominant_color'],
            'holes': analysis_data['hole_analysis']['has_holes'],
            'wrinkled': analysis_data['texture_analysis']['is_wrinkled'],
            'dark_spots': analysis_data['dark_spots'],
            'rot_streaks': analysis_data['rot_streaks']
        })

    return color_result


def analyze_object_quality_consolidated(crop_img: np.ndarray, object_class: str) -> dict[str, Any]:
    """
    Análisis consolidado que reutiliza conversiones BGR→RGB y redimensionamientos
    OPTIMIZACIÓN CRÍTICA: Una sola conversión BGR→RGB para todos los análisis
    
    Args:
        crop_img: Imagen recortada del objeto
        object_class: Tipo de objeto
        
    Returns:
        Dict con todos los resultados de análisis
    """
    # Obtener cache de imágenes
    image_cache = get_image_cache()

    # UNA SOLA conversión BGR→RGB (reutilizada para todos los análisis)
    rgb_img = image_cache.get_rgb_image(crop_img)

    # Redimensionamientos optimizados (reutilizados)
    small_rgb = image_cache.get_resized_image(rgb_img, (50, 50))
    image_cache.get_resized_image(rgb_img, (100, 100))

    # Análisis de huecos (usando imagen original)
    hole_analysis = detect_holes_and_cavities_optimized(crop_img)

    # Análisis de textura (usando imagen original)
    texture_analysis = detect_wrinkled_texture_optimized(crop_img)

    # Análisis de manchas oscuras (usando imagen original)
    dark_spots = detect_dark_spots_optimized(crop_img)

    # Análisis de rayas de podredumbre (usando imagen original)
    rot_streaks = detect_vertical_rot_streaks_optimized(crop_img)

    # Análisis de color RGB (usando imagen RGB ya convertida)
    dominant_color = get_dominant_rgb_color_from_rgb(small_rgb)
    color_result = classify_object_by_color_and_type(dominant_color, object_class)

    return {
        'hole_analysis': hole_analysis,
        'texture_analysis': texture_analysis,
        'dark_spots': dark_spots,
        'rot_streaks': rot_streaks,
        'dominant_color': dominant_color,
        'color_result': color_result
    }


def analyze_apple_quality(crop_img: np.ndarray | None) -> str:
    """
    Analizar calidad de manzana usando análisis RGB + detección de arrugas (COMPATIBILIDAD)
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        
    Returns:
        str: 'verde' o 'malograda'
    """
    result = analyze_object_quality(crop_img, "apple")
    # Convertir resultado nuevo a formato anterior para compatibilidad
    if result == 'sana':
        return 'verde'
    if result == 'contaminada':
        return 'malograda'
    return 'indeterminada'


def detect_holes_and_cavities(crop_img: np.ndarray) -> dict[str, any]:
    """
    Detectar huecos, agujeros y cavidades en objetos usando análisis de contornos
    
    Args:
        crop_img: Imagen recortada del objeto
        
    Returns:
        dict: Información sobre huecos detectados y confianza
    """
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro gaussiano para suavizar
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar bordes usando Canny
        edges = cv2.Canny(blurred, 30, 100)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'has_holes': False,
                'hole_count': 0,
                'largest_hole_area': 0,
                'hole_confidence': 0.0
            }

        # Ordenar contornos por área (de mayor a menor)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # El contorno más grande debería ser el objeto principal
        main_contour = contours[0]
        main_area = cv2.contourArea(main_contour)

        # Obtener configuración de umbrales
        color_config = get_color_config()
        area_ratio_min = color_config.get_threshold("holes", "area_ratio_min", 0.40)
        confidence_multiplier = color_config.get_threshold("holes", "confidence_multiplier", 5)

        # Buscar contornos internos (huecos)
        holes = []
        for contour in contours[1:]:  # Excluir el contorno principal
            area = cv2.contourArea(contour)
            # Solo considerar contornos significativos usando configuración
            if area > main_area * area_ratio_min:
                holes.append(contour)

        # Análisis de huecos
        hole_count = len(holes)
        has_holes = hole_count > 0

        # Calcular área del hueco más grande
        largest_hole_area = 0
        if holes:
            largest_hole_area = max(cv2.contourArea(hole) for hole in holes)

        # Calcular confianza basada en el tamaño relativo del hueco
        hole_confidence = 0.0
        if has_holes and main_area > 0:
            relative_hole_area = largest_hole_area / main_area
            # Usar multiplicador configurable para confianza
            hole_confidence = min(1.0, relative_hole_area * confidence_multiplier)

        return {
            'has_holes': has_holes,
            'hole_count': hole_count,
            'largest_hole_area': largest_hole_area,
            'hole_confidence': hole_confidence,
            'relative_hole_area': largest_hole_area / main_area if main_area > 0 else 0
        }

    except Exception as e:
        logger.error(f"Error detectando huecos: {e}")
        return {
            'has_holes': False,
            'hole_count': 0,
            'largest_hole_area': 0,
            'hole_confidence': 0.0
        }


def detect_dark_spots(crop_img: np.ndarray) -> bool:
    """
    Detectar manchas oscuras/podredumbre en objetos (OPTIMIZADO)
    
    Args:
        crop_img: Imagen recortada del objeto
        
    Returns:
        bool: True si tiene manchas oscuras significativas
    """
    try:
        # Obtener configuración de umbrales
        color_config = get_color_config()
        dark_threshold = color_config.get_threshold("dark_spots", "threshold", 120)
        ratio_threshold = color_config.get_threshold("dark_spots", "ratio_threshold", 0.35)

        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral para detectar píxeles oscuros
        dark_pixels = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)[1]

        # Contar píxeles oscuros
        dark_pixel_count = cv2.countNonZero(dark_pixels)
        total_pixels = crop_img.shape[0] * crop_img.shape[1]
        dark_pixel_ratio = dark_pixel_count / total_pixels

        # Usar umbral configurable
        return dark_pixel_ratio > ratio_threshold

    except Exception as e:
        logger.error(f"Error detectando manchas oscuras: {e}")
        return False


def detect_vertical_rot_streaks(crop_img: np.ndarray) -> bool:
    """
    Detectar rayas verticales de podredumbre en objetos
    
    Args:
        crop_img: Imagen recortada del objeto
        
    Returns:
        bool: True si tiene rayas verticales de podredumbre
    """
    try:
        # Obtener configuración de umbrales
        color_config = get_color_config()
        dark_threshold = color_config.get_threshold("vertical_rot_streaks", "threshold", 110)
        height_ratio_min = color_config.get_threshold("vertical_rot_streaks", "height_ratio_min", 0.20)
        intensity_max = color_config.get_threshold("vertical_rot_streaks", "intensity_max", 90)
        area_min = color_config.get_threshold("vertical_rot_streaks", "area_min", 25)

        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral para detectar píxeles oscuros
        dark_pixels = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)[1]

        # Aplicar morfología para conectar píxeles cercanos (kernel más alto y apertura+cierre)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25))  # Kernel vertical
        morphed = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Analizar cada contorno
        for contour in contours:
            # Calcular área del contorno
            area = cv2.contourArea(contour)

            # Calcular bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Verificar si es una raya vertical (más alta que ancha)
            if h > w * 1.5 and area > area_min:
                # Verificar si ocupa una buena parte de la altura del objeto
                height_ratio = h / crop_img.shape[0]
                if height_ratio > height_ratio_min:
                    # Comprobar que la franja sea realmente OSCURA
                    stripe_roi = gray[y:y + h, x:x + w]
                    if stripe_roi.size == 0:
                        continue
                    mean_intensity = float(np.mean(stripe_roi))
                    if mean_intensity < intensity_max:
                        return True
                    # Raya clara: ignorar (reduce falsos positivos)
                    continue

        return False

    except Exception as e:
        logger.error(f"Error detectando rayas de podredumbre: {e}")
        return False


def detect_irregular_contours(crop_img: np.ndarray) -> dict[str, any]:
    """
    Detectar contornos irregulares que pueden indicar daño severo
    
    Args:
        crop_img: Imagen recortada del objeto
        
    Returns:
        dict: Información sobre irregularidades en contornos
    """
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar bordes
        edges = cv2.Canny(blurred, 30, 100)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'is_irregular': False,
                'irregularity_score': 0.0,
                'contour_solidity': 1.0
            }

        # Encontrar el contorno más grande (objeto principal)
        main_contour = max(contours, key=cv2.contourArea)

        # Calcular solidez del contorno (área / área del casco convexo)
        area = cv2.contourArea(main_contour)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 1.0

        # Calcular irregularidad basada en la solidez
        # Solidez cercana a 1.0 = contorno regular
        # Solidez baja = contorno irregular (posible daño)
        irregularity_score = 1.0 - solidity

        # Considerar irregular si la solidez es menor a 0.3 (muy permisivo)
        is_irregular = solidity < 0.3

        return {
            'is_irregular': is_irregular,
            'irregularity_score': irregularity_score,
            'contour_solidity': solidity
        }

    except Exception as e:
        logger.error(f"Error detectando contornos irregulares: {e}")
        return {
            'is_irregular': False,
            'irregularity_score': 0.0,
            'contour_solidity': 1.0
        }


def detect_wrinkled_texture(crop_img: np.ndarray) -> dict[str, any]:
    """
    Detectar si una manzana está arrugada usando análisis de textura (OPTIMIZADO)
    
    Args:
        crop_img: Imagen recortada de la manzana
        
    Returns:
        dict: Información sobre si está arrugada y confianza
    """
    try:
        # Obtener configuración de umbrales
        color_config = get_color_config()
        edge_density_min = color_config.get_threshold("wrinkled_texture", "edge_density_min", 0.08)
        intensity_std_min = color_config.get_threshold("wrinkled_texture", "intensity_std_min", 80)
        avg_gradient_min = color_config.get_threshold("wrinkled_texture", "avg_gradient_min", 30)
        edge_density_alt = color_config.get_threshold("wrinkled_texture", "edge_density_alt", 0.12)
        intensity_std_alt = color_config.get_threshold("wrinkled_texture", "intensity_std_alt", 100)

        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro gaussiano para suavizar
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar bordes usando Canny (arrugas crean muchos bordes)
        edges = cv2.Canny(blurred, 30, 100)

        # Contar píxeles de bordes
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels

        # Análisis de variación de intensidad (arrugas = más variación)
        intensity_std = np.std(gray)

        # Análisis de gradientes locales (arrugas = gradientes pronunciados)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)

        # Criterios para detectar arrugas usando configuración
        is_wrinkled = (
            edge_density > edge_density_min and intensity_std > intensity_std_min and avg_gradient > avg_gradient_min
        ) or (
            edge_density > edge_density_alt
        ) or (
            intensity_std > intensity_std_alt
        )

        # Calcular confianza de arrugas
        wrinkle_confidence = min(1.0, (edge_density * 15 + intensity_std / 30 + avg_gradient / 20) / 3)

        return {
            'is_wrinkled': is_wrinkled,
            'edge_density': edge_density,
            'intensity_std': intensity_std,
            'avg_gradient': avg_gradient,
            'wrinkle_confidence': wrinkle_confidence
        }

    except Exception:
        # Si hay error, asumir sin arrugas
        return {
            'is_wrinkled': False,
            'edge_density': 0.0,
            'intensity_std': 0.0,
            'avg_gradient': 0.0,
            'wrinkle_confidence': 0.0
        }


def get_dominant_rgb_color(crop_img: np.ndarray) -> tuple[int, int, int]:
    """
    Obtener el color RGB dominante de la imagen (OPTIMIZADO)
    
    Args:
        crop_img: Imagen recortada
        
    Returns:
        Tuple[int, int, int]: RGB promedio dominante
    """
    # Redimensionar para análisis más rápido
    small_img = cv2.resize(crop_img, (50, 50))

    # Convertir BGR a RGB (OpenCV usa BGR por defecto)
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    # Calcular promedio vectorizado (más eficiente que 3 llamadas separadas)
    rgb_means = np.mean(rgb_img, axis=(0, 1))
    return int(rgb_means[0]), int(rgb_means[1]), int(rgb_means[2])


def get_dominant_rgb_color_from_rgb(rgb_img: np.ndarray) -> tuple[int, int, int]:
    """
    Obtener el color RGB dominante desde imagen ya convertida (ULTRA OPTIMIZADO)
    
    Args:
        rgb_img: Imagen ya en formato RGB
        
    Returns:
        Tuple[int, int, int]: RGB promedio dominante
    """
    # Calcular promedio vectorizado directamente (sin conversiones adicionales)
    rgb_means = np.mean(rgb_img, axis=(0, 1))
    return int(rgb_means[0]), int(rgb_means[1]), int(rgb_means[2])


def classify_object_by_color_and_type(rgb_color: tuple[int, int, int], object_class: str) -> str:
    """
    Clasificar objeto según su color RGB dominante y tipo usando configuración externa
    
    Args:
        rgb_color: Color RGB (R, G, B)
        object_class: Tipo de objeto ('apple', 'orange', 'sports ball', etc.)
        
    Returns:
        str: 'sana', 'contaminada' o 'indeterminada'
    """
    # Usar configuración externa para clasificación
    color_config = get_color_config()
    return color_config.classify_color(rgb_color, object_class)


def classify_apple_by_color(rgb_color: tuple[int, int, int]) -> str:
    """
    Clasificar manzana según su color RGB dominante
    Solo clasifica: Verde (fresca) o Malograda (dañada/podrida)
    
    Args:
        rgb_color: Color RGB (R, G, B)
        
    Returns:
        str: 'sana' o 'contaminada'
    """
    r, g, b = rgb_color

    # PRIORIDAD 1: Si tiene colores muy oscuros (definitivamente contaminada)
    if r < 100 and g < 100 and b < 80:  # Colores muy oscuros/marrones (más estricto)
        return 'contaminada'

    # PRIORIDAD 2: Si tiene colores marrones/terrosos (contaminada)
    if r < 140 and g < 120 and b < 90:  # Colores marrones claramente (más estricto)
        return 'contaminada'

    # PRIORIDAD 2.5: Si tiene manchas marrones (variación de color alta)
    # Detectar si hay zonas muy oscuras vs zonas claras (manchas)
    # HACER MÁS PERMISIVO para manzanas verdes
    if r < 120 and g < 100 and b < 70:  # Solo colores muy oscuros/marrones
        return 'contaminada'

    # PRIORIDAD 3: Si el verde es dominante (probablemente sana)
    max_channel = max(r, g, b)

    # Si el verde es el canal más alto y es significativo
    if g == max_channel and g > 100:  # Verde dominante (más permisivo)
        return 'sana'

    # PRIORIDAD 4: Rango específico para manzanas VERDES frescas (más amplio)
    # Verde: RGB (120-240, 140-255, 60-180) - Manzanas frescas y saludables
    if 120 <= r <= 240 and 140 <= g <= 255 and 60 <= b <= 180:
        return 'sana'

    # PRIORIDAD 5: Si tiene colores verde-amarillentos (probablemente sana)
    if g > r and g > b and g > 110:  # Verde dominante con buen nivel (más permisivo)
        return 'sana'

    # PRIORIDAD 6: Si tiene colores claros pero no claramente verde
    if r > 120 and g > 120 and b > 80:  # Colores claros/bright (más permisivo)
        return 'sana'  # Asumir que es sana si es claro

    # PRIORIDAD 7: Manzanas amarillo-verdosas (como la de la imagen)
    if r > 140 and g > 140 and b > 90 and g >= r - 30:  # Amarillo-verdoso (más permisivo)
        return 'sana'

    # PRIORIDAD 8: Si tiene colores cálidos pero claros (no marrones)
    if r > 120 and g > 120 and b > 70 and r + g + b > 350:  # Colores cálidos claros (más permisivo)
        return 'sana'

    # Por defecto: Si tiene colores oscuros o marrones, es contaminada
    if r < 130 or g < 110 or b < 80:  # Más estricto para detectar podredumbre
        return 'contaminada'

    # Si llega aquí, probablemente es sana
    return 'sana'


def classify_orange_by_color(rgb_color: tuple[int, int, int]) -> str:
    """
    Clasificar naranja según su color RGB dominante
    
    Args:
        rgb_color: Color RGB (R, G, B)
        
    Returns:
        str: 'sana' o 'contaminada'
    """
    r, g, b = rgb_color

    # PRIORIDAD 1: Colores muy oscuros/marrones (contaminada)
    if r < 80 and g < 80 and b < 60:
        return 'contaminada'

    # PRIORIDAD 2: Rango específico para naranjas SANAS
    # Naranja: RGB (200-255, 120-200, 0-100) - Naranjas frescas
    if 200 <= r <= 255 and 120 <= g <= 200 and 0 <= b <= 100:
        return 'sana'

    # PRIORIDAD 3: Naranjas amarillentas (sanas)
    if r > 180 and g > 100 and b < 80 and r > g:
        return 'sana'

    # PRIORIDAD 4: Colores cálidos claros (sanas)
    if r > 160 and g > 80 and b < 100 and r + g + b > 300:
        return 'sana'

    # Por defecto: Si es oscuro, es contaminada
    if r < 140 or g < 60:
        return 'contaminada'

    return 'sana'


def classify_generic_object_by_color(rgb_color: tuple[int, int, int]) -> str:
    """
    Clasificación genérica para objetos no específicos
    
    Args:
        rgb_color: Color RGB (R, G, B)
        
    Returns:
        str: 'sana' o 'contaminada'
    """
    r, g, b = rgb_color

    # PRIORIDAD 1: Colores muy oscuros (contaminada)
    if r < 60 and g < 60 and b < 60:
        return 'contaminada'

    # PRIORIDAD 2: Colores claros/vibrantes (sanos)
    if r > 150 or g > 150 or b > 150:
        return 'sana'

    # PRIORIDAD 3: Colores medios (evaluar por brillo total)
    total_brightness = r + g + b
    if total_brightness > 300:  # Brillo alto
        return 'sana'
    if total_brightness < 150:  # Brillo bajo
        return 'contaminada'
    return 'sana'  # Por defecto, asumir sano


def get_apple_analysis_details(crop_img: np.ndarray | None) -> dict[str, any]:
    """
    Obtener detalles completos del análisis de manzana
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        
    Returns:
        Dict: Detalles del análisis (color dominante, clasificación, confianza)
    """
    if crop_img is None or crop_img.size == 0:
        return {
            'dominant_color': (0, 0, 0),
            'classification': 'indeterminada',
            'confidence': 0.0,
            'color_analysis': {}
        }

    # Obtener color dominante
    dominant_color = get_dominant_rgb_color(crop_img)

    # Clasificar
    classification = classify_apple_by_color(dominant_color)

    # Calcular confianza basada en qué tan cerca está del rango ideal
    confidence = calculate_color_confidence(dominant_color, classification)

    # Análisis detallado de color
    color_analysis = analyze_color_distribution(crop_img)

    return {
        'dominant_color': dominant_color,
        'classification': classification,
        'confidence': confidence,
        'color_analysis': color_analysis
    }


def calculate_color_confidence(rgb_color: tuple[int, int, int], classification: str) -> float:
    """
    Calcular confianza de la clasificación basada en proximidad a rangos ideales
    
    Args:
        rgb_color: Color RGB dominante
        classification: Clasificación obtenida
        
    Returns:
        float: Confianza entre 0.0 y 1.0
    """
    r, g, b = rgb_color

    if classification == 'verde':
        # Rango ideal: RGB (180, 205, 115)
        ideal_r, ideal_g, ideal_b = 180, 205, 115
        distance = np.sqrt((r - ideal_r)**2 + (g - ideal_g)**2 + (b - ideal_b)**2)
        max_distance = 100  # Distancia máxima para confianza 0

    elif classification == 'malograda':
        # Rango ideal: RGB (90, 60, 45)
        ideal_r, ideal_g, ideal_b = 90, 60, 45
        distance = np.sqrt((r - ideal_r)**2 + (g - ideal_g)**2 + (b - ideal_b)**2)
        max_distance = 80

    else:  # indeterminada
        return 0.3

    confidence = max(0.0, 1.0 - (distance / max_distance))
    return round(confidence, 2)


def analyze_color_distribution(crop_img: np.ndarray) -> dict[str, float]:
    """
    Analizar distribución de colores en la imagen (OPTIMIZADO)
    
    Args:
        crop_img: Imagen recortada
        
    Returns:
        Dict: Análisis de distribución de colores
    """
    # Redimensionar para análisis
    small_img = cv2.resize(crop_img, (100, 100))
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    # Calcular estadísticas por canal (vectorizado)
    r_channel = rgb_img[:, :, 0]
    g_channel = rgb_img[:, :, 1]
    b_channel = rgb_img[:, :, 2]

    return {
        'r_mean': float(np.mean(r_channel)),
        'r_std': float(np.std(r_channel)),
        'g_mean': float(np.mean(g_channel)),
        'g_std': float(np.std(g_channel)),
        'b_mean': float(np.mean(b_channel)),
        'b_std': float(np.std(b_channel)),
        'brightness': float(np.mean(rgb_img)),
        'contrast': float(np.std(rgb_img))
    }


def analyze_color_distribution_from_rgb(rgb_img: np.ndarray) -> dict[str, float]:
    """
    Analizar distribución de colores desde imagen RGB ya convertida (ULTRA OPTIMIZADO)
    
    Args:
        rgb_img: Imagen ya en formato RGB
        
    Returns:
        Dict: Análisis de distribución de colores
    """
    # Calcular estadísticas por canal (vectorizado) - sin conversiones adicionales
    r_channel = rgb_img[:, :, 0]
    g_channel = rgb_img[:, :, 1]
    b_channel = rgb_img[:, :, 2]

    return {
        'r_mean': float(np.mean(r_channel)),
        'r_std': float(np.std(r_channel)),
        'g_mean': float(np.mean(g_channel)),
        'g_std': float(np.std(g_channel)),
        'b_mean': float(np.mean(b_channel)),
        'b_std': float(np.std(b_channel)),
        'brightness': float(np.mean(rgb_img)),
        'contrast': float(np.std(rgb_img))
    }


def configure_apple_color_ranges(
    green_range: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((160, 200), (190, 220), (100, 130)),
    malograda_range: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((60, 120), (40, 80), (30, 60))
) -> dict[str, any]:
    """
    Configurar rangos de color personalizados para clasificación de manzanas
    
    Args:
        green_range: Rango RGB para manzanas verdes ((r_min, r_max), (g_min, g_max), (b_min, b_max))
        malograda_range: Rango RGB para manzanas malogradas/dañadas
        
    Returns:
        Dict: Configuración de rangos de color
    """
    return {
        'green_range': green_range,
        'malograda_range': malograda_range
    }


def classify_apple_with_custom_ranges(rgb_color: tuple[int, int, int], color_ranges: dict[str, any]) -> str:
    """
    Clasificar manzana usando rangos de color personalizados
    
    Args:
        rgb_color: Color RGB (R, G, B)
        color_ranges: Diccionario con rangos personalizados
        
    Returns:
        str: 'verde' o 'malograda'
    """
    r, g, b = rgb_color

    # Verificar rango verde
    green_range = color_ranges['green_range']
    if (green_range[0][0] <= r <= green_range[0][1] and
        green_range[1][0] <= g <= green_range[1][1] and
        green_range[2][0] <= b <= green_range[2][1]):
        return 'verde'

    # Verificar rango malograda
    malograda_range = color_ranges['malograda_range']
    if (malograda_range[0][0] <= r <= malograda_range[0][1] and
        malograda_range[1][0] <= g <= malograda_range[1][1] and
        malograda_range[2][0] <= b <= malograda_range[2][1]):
        return 'malograda'

    return 'malograda'  # Por defecto, si no está claramente verde, es malograda


def analyze_apple_quality_with_logging(crop_img: np.ndarray | None) -> str:
    """
    Análisis RGB + textura para clasificar manzanas con logging detallado (COMPATIBILIDAD)
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        
    Returns:
        str: 'verde' o 'malograda'
    """
    result = analyze_object_quality_with_logging(crop_img, "apple")
    # Convertir resultado nuevo a formato anterior para compatibilidad
    if result == 'sana':
        return 'verde'
    if result == 'contaminada':
        return 'malograda'
    return 'indeterminada'


def analyze_object_quality_with_logging(crop_img: np.ndarray | None, object_class: str = "apple") -> str:
    """
    Análisis RGB + textura para clasificar cualquier objeto con logging detallado
    
    Args:
        crop_img: Imagen recortada del objeto detectado
        object_class: Tipo de objeto ('apple', 'orange', 'sports ball', etc.)
        
    Returns:
        str: 'sana', 'contaminada' o 'indeterminada'
    """
    if crop_img is None or crop_img.size == 0:
        return 'indeterminada'

    # Análisis completo (RGB + textura)
    result = analyze_object_quality(crop_img, object_class)

    # Logging detallado con diagnóstico completo
    dominant_color = get_dominant_rgb_color(crop_img)
    texture_analysis = detect_wrinkled_texture(crop_img)
    hole_analysis = detect_holes_and_cavities(crop_img)
    contour_analysis = detect_irregular_contours(crop_img)

    if result == 'contaminada':
        # Verificar la causa específica del daño
        # Log detallado solo para casos críticos o cada cierto intervalo
        if hole_analysis['has_holes'] and hole_analysis['hole_confidence'] > 0.8:
            analysis_logger.log_analysis(f"HUECOS: {object_class.upper()} (confianza: {hole_analysis['hole_confidence']:.2f})")
        elif contour_analysis['is_irregular'] and contour_analysis['irregularity_score'] > 0.5:
            analysis_logger.log_analysis(f"CONTORNOS IRREGULARES: {object_class.upper()} (puntuación: {contour_analysis['irregularity_score']:.2f})")
        elif texture_analysis['is_wrinkled']:
            analysis_logger.log_analysis(f"ARRUGADO: {object_class.upper()} (confianza: {texture_analysis['wrinkle_confidence']:.2f})")
        else:
            analysis_logger.log_analysis(f"CONTAMINADO POR COLOR: {object_class.upper()} RGB: {dominant_color}")
    else:
        analysis_logger.buffer_analysis(f"SANA_{object_class.upper()}", {'rgb': dominant_color})

    return result


def get_apple_quality_details(crop_img: np.ndarray | None) -> dict[str, any]:
    """
    Obtener detalles del análisis de calidad de manzana
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        
    Returns:
        Dict: Detalles del análisis (color dominante, clasificación, confianza)
    """
    return get_apple_analysis_details(crop_img)


def adjust_classification_thresholds(
    brightness_threshold_low: int = 70,
    brightness_threshold_medium: int = 100,
    variation_threshold: float = 35.0,
    edge_density_threshold: float = 0.15,
    contamination_threshold: int = 1
) -> dict[str, any]:
    """
    Configurar umbrales de clasificación personalizados
    
    Args:
        brightness_threshold_low: Umbral de brillo bajo (puntuación +2)
        brightness_threshold_medium: Umbral de brillo medio (puntuación +1)
        variation_threshold: Umbral de variación de brillo (puntuación +1)
        edge_density_threshold: Umbral de densidad de bordes (puntuación +1)
        contamination_threshold: Umbral mínimo para clasificar como contaminada
        
    Returns:
        dict: Configuración de umbrales
    """
    return {
        'brightness_threshold_low': brightness_threshold_low,
        'brightness_threshold_medium': brightness_threshold_medium,
        'variation_threshold': variation_threshold,
        'edge_density_threshold': edge_density_threshold,
        'contamination_threshold': contamination_threshold
    }


def analyze_apple_quality_custom(crop_img: np.ndarray | None, color_ranges: dict[str, any]) -> str:
    """
    Análisis RGB con rangos de color personalizados
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        color_ranges: Diccionario con rangos de color personalizados
        
    Returns:
        str: 'verde' o 'malograda'
    """
    if crop_img is None or crop_img.size == 0:
        return 'indeterminada'

    # Obtener color dominante
    dominant_color = get_dominant_rgb_color(crop_img)

    # Clasificar usando rangos personalizados
    return classify_apple_with_custom_ranges(dominant_color, color_ranges)


def reload_color_classification_config():
    """
    Recargar configuración de clasificación de colores desde archivo
    Útil para cambios dinámicos sin reiniciar la aplicación
    """
    from .color_config import reload_color_config
    reload_color_config()
    logger.info("Configuración de clasificación de colores recargada")


def get_color_classification_info() -> dict[str, any]:
    """
    Obtener información sobre la configuración actual de clasificación de colores
    
    Returns:
        dict: Información de configuración actual
    """
    color_config = get_color_config()
    return {
        "available_objects": color_config.get_available_object_classes(),
        "config_path": color_config.config_path,
        "thresholds": color_config.get_thresholds()
    }


# ============================================================================
# FUNCIONES OPTIMIZADAS PARA REUTILIZACIÓN DE CONVERSIONES BGR→RGB
# ============================================================================

def detect_holes_and_cavities_optimized(crop_img: np.ndarray) -> dict[str, any]:
    """
    Detectar huecos y cavidades (VERSIÓN OPTIMIZADA)
    Reutiliza la misma lógica pero sin conversiones BGR→RGB adicionales
    """
    return detect_holes_and_cavities(crop_img)


def detect_wrinkled_texture_optimized(crop_img: np.ndarray) -> dict[str, any]:
    """
    Detectar textura arrugada (VERSIÓN OPTIMIZADA)
    Reutiliza la misma lógica pero sin conversiones BGR→RGB adicionales
    """
    return detect_wrinkled_texture(crop_img)


def detect_dark_spots_optimized(crop_img: np.ndarray) -> bool:
    """
    Detectar manchas oscuras (VERSIÓN OPTIMIZADA)
    Reutiliza la misma lógica pero sin conversiones BGR→RGB adicionales
    """
    return detect_dark_spots(crop_img)


def detect_vertical_rot_streaks_optimized(crop_img: np.ndarray) -> bool:
    """
    Detectar rayas de podredumbre (VERSIÓN OPTIMIZADA)
    Reutiliza la misma lógica pero sin conversiones BGR→RGB adicionales
    """
    return detect_vertical_rot_streaks(crop_img)


def get_image_cache_stats() -> dict[str, int]:
    """
    Obtener estadísticas del cache de imágenes
    
    Returns:
        dict: Estadísticas del cache
    """
    image_cache = get_image_cache()
    return image_cache.get_cache_stats()


def clear_image_cache_optimized():
    """
    Limpiar cache de imágenes optimizado
    """
    image_cache = get_image_cache()
    image_cache.clear_cache()
    logger.info("Cache de imágenes optimizado limpiado")

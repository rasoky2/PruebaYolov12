"""
Análisis RGB para clasificación de calidad de manzanas
Sistema de detección de estado de madurez y daño basado en análisis de color RGB
"""

import cv2
import numpy as np
from typing import Optional


def analyze_apple_quality(crop_img: Optional[np.ndarray]) -> str:
    """
    Analizar calidad de manzana usando análisis RGB + detección de arrugas
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        
    Returns:
        str: 'verde' o 'malograda'
    """
    if crop_img is None or crop_img.size == 0:
        return 'indeterminada'
    
    # PRIORIDAD 1: Detectar arrugas/textura (manzanas muy arrugadas)
    texture_analysis = detect_wrinkled_texture(crop_img)
    if texture_analysis['is_wrinkled']:
        return 'malograda'  # Si está arrugada, es malograda
    
    # PRIORIDAD 2: Análisis RGB para el resto
    dominant_color = get_dominant_rgb_color(crop_img)
    return classify_apple_by_color(dominant_color)


def detect_wrinkled_texture(crop_img: np.ndarray) -> dict[str, any]:
    """
    Detectar si una manzana está arrugada usando análisis de textura
    
    Args:
        crop_img: Imagen recortada de la manzana
        
    Returns:
        dict: Información sobre si está arrugada y confianza
    """
    try:
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
        
        # Criterios para detectar arrugas (más conservador)
        # Solo detectar arrugas si hay evidencia MUY clara
        is_wrinkled = (
            # Criterio 1: Muchos bordes Y alta variación (arrugas evidentes)
            edge_density > 0.08 and          # Muchos bordes
            intensity_std > 80 and           # Variación muy alta
            avg_gradient > 30                # Gradientes muy pronunciados
        ) or (
            # Criterio 2: Densidad de bordes extremadamente alta
            edge_density > 0.12              # MUCHÍSIMOS bordes
        ) or (
            # Criterio 3: Variación extremadamente alta (daño severo)
            intensity_std > 100              # Variación extrema
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
        
    except Exception as e:
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
    Obtener el color RGB dominante de la imagen
    
    Args:
        crop_img: Imagen recortada
        
    Returns:
        Tuple[int, int, int]: RGB promedio dominante
    """
    # Redimensionar para análisis más rápido
    small_img = cv2.resize(crop_img, (50, 50))
    
    # Convertir BGR a RGB (OpenCV usa BGR por defecto)
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    
    # Calcular promedio de cada canal
    mean_r = np.mean(rgb_img[:, :, 0])
    mean_g = np.mean(rgb_img[:, :, 1])
    mean_b = np.mean(rgb_img[:, :, 2])
    
    return int(mean_r), int(mean_g), int(mean_b)


def classify_apple_by_color(rgb_color: tuple[int, int, int]) -> str:
    """
    Clasificar manzana según su color RGB dominante
    Solo clasifica: Verde (fresca) o Malograda (dañada/podrida)
    
    Args:
        rgb_color: Color RGB (R, G, B)
        
    Returns:
        str: 'verde' o 'malograda'
    """
    r, g, b = rgb_color
    
    # PRIORIDAD 1: Si tiene colores muy oscuros (definitivamente malograda)
    if r < 80 and g < 80 and b < 60:  # Colores muy oscuros/marrones
        return 'malograda'
    
    # PRIORIDAD 2: Si tiene colores marrones/terrosos (malograda)
    if r < 120 and g < 100 and b < 70:  # Colores marrones claramente
        return 'malograda'
    
    # PRIORIDAD 2.5: Si tiene manchas marrones (variación de color alta)
    # Detectar si hay zonas muy oscuras vs zonas claras (manchas)
    if r < 150 and g < 140 and b < 90:  # Colores más oscuros/marrones
        return 'malograda'
    
    # PRIORIDAD 3: Si el verde es dominante (probablemente verde)
    max_channel = max(r, g, b)
    
    # Si el verde es el canal más alto y es significativo
    if g == max_channel and g > 120:  # Verde dominante
        return 'verde'
    
    # PRIORIDAD 4: Rango específico para manzanas VERDES frescas (más amplio)
    # Verde: RGB (140-220, 160-240, 80-150) - Manzanas frescas y saludables
    if 140 <= r <= 220 and 160 <= g <= 240 and 80 <= b <= 150:
        return 'verde'
    
    # PRIORIDAD 5: Si tiene colores verde-amarillentos (probablemente verde)
    if g > r and g > b and g > 130:  # Verde dominante con buen nivel
        return 'verde'
    
    # PRIORIDAD 6: Si tiene colores claros pero no claramente verde
    if r > 150 and g > 150 and b > 100:  # Colores claros/bright
        return 'verde'  # Asumir que es verde si es claro
    
    # PRIORIDAD 7: Manzanas amarillo-verdosas (como la de la imagen)
    if r > 180 and g > 180 and b > 120 and g >= r - 20:  # Amarillo-verdoso
        return 'verde'
    
    # PRIORIDAD 8: Si tiene colores cálidos pero claros (no marrones)
    if r > 160 and g > 160 and b > 100 and r + g + b > 450:  # Colores cálidos claros
        return 'verde'
    
    # Por defecto: Solo si es muy oscuro o marrón, es malograda
    if r < 140 or g < 120 or b < 80:  # Solo si es realmente oscuro
        return 'malograda'
    
    # Si llega aquí, probablemente es verde
    return 'verde'


def get_apple_analysis_details(crop_img: Optional[np.ndarray]) -> dict[str, any]:
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
    Analizar distribución de colores en la imagen
    
    Args:
        crop_img: Imagen recortada
        
    Returns:
        Dict: Análisis de distribución de colores
    """
    # Redimensionar para análisis
    small_img = cv2.resize(crop_img, (100, 100))
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    
    # Calcular estadísticas por canal
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


def analyze_apple_quality_with_logging(crop_img: Optional[np.ndarray]) -> str:
    """
    Análisis RGB + textura para clasificar manzanas con logging detallado
    
    Args:
        crop_img: Imagen recortada de la manzana detectada
        
    Returns:
        str: 'verde' o 'malograda'
    """
    if crop_img is None or crop_img.size == 0:
        return 'indeterminada'
    
    # Análisis completo (RGB + textura)
    result = analyze_apple_quality(crop_img)
    
    # Logging detallado con diagnóstico completo
    dominant_color = get_dominant_rgb_color(crop_img)
    texture_analysis = detect_wrinkled_texture(crop_img)
    
    if result == 'malograda':
        # Verificar si fue por arrugas o por color
        if texture_analysis['is_wrinkled']:
            print(f"[ARRUGAS] Manzana ARRUGADA detectada (confianza: {texture_analysis['wrinkle_confidence']:.2f})")
            print(f"  - RGB: {dominant_color}")
            print(f"  - Densidad bordes: {texture_analysis['edge_density']:.3f} (límite: 0.08)")
            print(f"  - Variación intensidad: {texture_analysis['intensity_std']:.1f} (límite: 80/100)")
            print(f"  - Gradiente promedio: {texture_analysis['avg_gradient']:.1f} (límite: 30)")
        else:
            print(f"[COLOR] Manzana MALOGRADA por color RGB: {dominant_color}")
            print(f"  - Análisis de color:")
            r, g, b = dominant_color
            print(f"    • R={r}, G={g}, B={b}")
            print(f"    • Verde dominante: {g == max(r,g,b) and g > 120}")
            print(f"    • Rango verde: {140 <= r <= 220 and 160 <= g <= 240 and 80 <= b <= 150}")
            print(f"    • Verde vs otros: {g > r and g > b and g > 130}")
            print(f"    • Colores claros: {r > 150 and g > 150 and b > 100}")
    else:
        print(f"[VERDE] Manzana VERDE detectada RGB: {dominant_color}")
        print(f"  - Análisis exitoso")
    
    return result


def get_apple_quality_details(crop_img: Optional[np.ndarray]) -> dict[str, any]:
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


def analyze_apple_quality_custom(crop_img: Optional[np.ndarray], color_ranges: dict[str, any]) -> str:
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

"""
Análisis RGB para clasificación de calidad de manzanas
Sistema de detección de estado de madurez y daño basado en análisis de color RGB
"""

import cv2
import numpy as np
import json
import os
from typing import Optional, List


def load_interface_config() -> dict:
    """Cargar configuración desde interface_config.json"""
    try:
        # Obtener la ruta del archivo de configuración
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(current_dir, "interface_config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # print(f"✅ Configuración cargada desde: {config_path}")  # Log reducido
            return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error cargando interface_config.json: {e}")
        print("🔄 Usando configuración por defecto")
        return {
            "detection_classes": ["apple", "orange"],
            "colors": {
                "sana": "#00FF00",
                "contaminada": "#FF0000",
                "detectada": "#FFFF00"
            }
        }


def get_detection_classes() -> List[str]:
    """Obtener las clases de detección desde la configuración"""
    config = load_interface_config()
    classes = config.get("detection_classes", ["apple", "orange"])
    # print(f"🎯 Clases de detección disponibles: {classes}")  # Log reducido
    return classes


def analyze_object_quality(crop_img: Optional[np.ndarray], object_class: str = "apple") -> str:
    """
    Analizar calidad de objeto detectado usando análisis RGB + detección de textura + huecos
    
    Args:
        crop_img: Imagen recortada del objeto detectado
        object_class: Tipo de objeto ('apple', 'orange', 'sports ball', etc.)
        
    Returns:
        str: 'sana', 'contaminada' o 'indeterminada'
    """
    if crop_img is None or crop_img.size == 0:
        return 'indeterminada'
    
    # PRIORIDAD 1: Detectar huecos/agujeros (daño severo)
    hole_analysis = detect_holes_and_cavities(crop_img)
    if hole_analysis['has_holes'] and hole_analysis['hole_confidence'] > 0.9:
        return 'contaminada'  # Si tiene huecos significativos, es contaminada
    
    # PRIORIDAD 2: Detectar contornos irregulares (daño estructural) - DESHABILITADO PERMANENTEMENTE
    # El algoritmo falla completamente y causa falsos positivos masivos
    # contour_analysis = detect_irregular_contours(crop_img)
    # if contour_analysis['is_irregular'] and contour_analysis['irregularity_score'] > 0.5:
    #     return 'contaminada'  # Si tiene contornos muy irregulares, es contaminada
    
    # PRIORIDAD 3: Detectar textura dañada/arrugada
    texture_analysis = detect_wrinkled_texture(crop_img)
    if texture_analysis['is_wrinkled']:
        return 'contaminada'  # Si está arrugada/dañada, es contaminada
    
    # PRIORIDAD 4: Detectar manchas oscuras/podredumbre
    if detect_dark_spots(crop_img):
        return 'contaminada'  # Si tiene manchas oscuras significativas, es contaminada
    
    # PRIORIDAD 4.5: Detectar rayas verticales de podredumbre
    if detect_vertical_rot_streaks(crop_img):
        return 'contaminada'  # Si tiene rayas de podredumbre, es contaminada
    
    # PRIORIDAD 5: Análisis RGB específico por tipo de objeto
    dominant_color = get_dominant_rgb_color(crop_img)
    color_result = classify_object_by_color_and_type(dominant_color, object_class)
    
    # Debug reducido: Solo mostrar análisis si hay contaminación detectada
    if color_result in ['contaminada', 'detectada']:
        print(f"🔍 {object_class.upper()} {color_result.upper()}: RGB {dominant_color}")
        if hole_analysis['has_holes']:
            print(f"  - Huecos detectados (confianza: {hole_analysis['hole_confidence']:.2f})")
        if texture_analysis['is_wrinkled']:
            print(f"  - Textura arrugada (confianza: {texture_analysis['wrinkle_confidence']:.2f})")
        if detect_dark_spots(crop_img):
            print(f"  - Manchas oscuras detectadas")
        if detect_vertical_rot_streaks(crop_img):
            print(f"  - Rayas de podredumbre detectadas")
    
    return color_result


def analyze_apple_quality(crop_img: Optional[np.ndarray]) -> str:
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
    elif result == 'contaminada':
        return 'malograda'
    else:
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
        
        # Buscar contornos internos (huecos)
        holes = []
        for contour in contours[1:]:  # Excluir el contorno principal
            area = cv2.contourArea(contour)
        # Solo considerar contornos significativos (más del 40% del área principal)
        # Aumentado aún más para reducir falsos positivos
        if area > main_area * 0.40:
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
            # Si el hueco es más del 20% del área total, es muy probable que sea daño
            # Aumentado de 10% a 20% para reducir falsos positivos
            hole_confidence = min(1.0, relative_hole_area * 5)
        
        return {
            'has_holes': has_holes,
            'hole_count': hole_count,
            'largest_hole_area': largest_hole_area,
            'hole_confidence': hole_confidence,
            'relative_hole_area': largest_hole_area / main_area if main_area > 0 else 0
        }
        
    except Exception as e:
        print(f"Error detectando huecos: {e}")
        return {
            'has_holes': False,
            'hole_count': 0,
            'largest_hole_area': 0,
            'hole_confidence': 0.0
        }


def detect_dark_spots(crop_img: np.ndarray) -> bool:
    """
    Detectar manchas oscuras/podredumbre en objetos
    
    Args:
        crop_img: Imagen recortada del objeto
        
    Returns:
        bool: True si tiene manchas oscuras significativas
    """
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral para detectar píxeles oscuros
        # Píxeles con intensidad menor a 120 se consideran oscuros (más sensible)
        dark_threshold = 120
        dark_pixels = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Contar píxeles oscuros
        dark_pixel_count = cv2.countNonZero(dark_pixels)
        total_pixels = crop_img.shape[0] * crop_img.shape[1]
        dark_pixel_ratio = dark_pixel_count / total_pixels
        
        # Si más del 35% de la imagen son píxeles oscuros, es contaminada (más permisivo)
        has_dark_spots = dark_pixel_ratio > 0.35
        
        return has_dark_spots
        
    except Exception as e:
        print(f"Error detectando manchas oscuras: {e}")
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
        # Convertir a escala de grises
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral para detectar píxeles oscuros (aún más sensible)
        dark_threshold = 110
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
            if h > w * 1.5 and area > 25:  # Raya vertical significativa (más sensible)
                # Verificar si ocupa una buena parte de la altura del objeto
                height_ratio = h / crop_img.shape[0]
                if height_ratio > 0.20:  # Al menos 20% de la altura (más sensible)
                    # Comprobar que la franja sea realmente OSCURA (promedio < 90)
                    stripe_roi = gray[y:y+h, x:x+w]
                    if stripe_roi.size == 0:
                        continue
                    mean_intensity = float(np.mean(stripe_roi))
                    if mean_intensity < 90:  # oscuridad suficiente
                        # print(f"🔍 RAYA VERTICAL de podredumbre detectada: {height_ratio:.1%} de altura, intensidad media {mean_intensity:.1f}")  # Log reducido
                        return True
                    else:
                        # Raya clara: ignorar (reduce falsos positivos)
                        continue
        
        return False
        
    except Exception as e:
        print(f"Error detectando rayas de podredumbre: {e}")
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
        print(f"Error detectando contornos irregulares: {e}")
        return {
            'is_irregular': False,
            'irregularity_score': 0.0,
            'contour_solidity': 1.0
        }


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


def classify_object_by_color_and_type(rgb_color: tuple[int, int, int], object_class: str) -> str:
    """
    Clasificar objeto según su color RGB dominante y tipo
    
    Args:
        rgb_color: Color RGB (R, G, B)
        object_class: Tipo de objeto ('apple', 'orange', 'sports ball', etc.)
        
    Returns:
        str: 'sana' o 'contaminada'
    """
    r, g, b = rgb_color
    
    # Clasificación específica por tipo de objeto
    if object_class.lower() == "apple":
        return classify_apple_by_color(rgb_color)
    elif object_class.lower() == "orange":
        return classify_orange_by_color(rgb_color)
    
    else:
        # Para otros objetos, usar análisis genérico
        return classify_generic_object_by_color(rgb_color)


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
    elif total_brightness < 150:  # Brillo bajo
        return 'contaminada'
    else:
        return 'sana'  # Por defecto, asumir sano


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
    elif result == 'contaminada':
        return 'malograda'
    else:
        return 'indeterminada'


def analyze_object_quality_with_logging(crop_img: Optional[np.ndarray], object_class: str = "apple") -> str:
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
        if hole_analysis['has_holes'] and hole_analysis['hole_confidence'] > 0.8:
            # print(f"[HUECOS] {object_class.upper()} CON HUECOS detectado (confianza: {hole_analysis['hole_confidence']:.2f})")  # Log reducido
            print(f"  - RGB: {dominant_color}")
            print(f"  - Número de huecos: {hole_analysis['hole_count']}")
            print(f"  - Área del hueco más grande: {hole_analysis['largest_hole_area']:.0f} píxeles")
            print(f"  - Área relativa del hueco: {hole_analysis['relative_hole_area']:.1%}")
        elif contour_analysis['is_irregular'] and contour_analysis['irregularity_score'] > 0.5:
            print(f"[CONTORNOS] {object_class.upper()} CON CONTORNOS IRREGULARES detectado (puntuación: {contour_analysis['irregularity_score']:.2f})")
            print(f"  - RGB: {dominant_color}")
            print(f"  - Solidez del contorno: {contour_analysis['contour_solidity']:.3f} (límite: 0.30)")
            print(f"  - Puntuación de irregularidad: {contour_analysis['irregularity_score']:.3f}")
        elif texture_analysis['is_wrinkled']:
            print(f"[ARRUGAS] {object_class.upper()} ARRUGADO detectado (confianza: {texture_analysis['wrinkle_confidence']:.2f})")
            print(f"  - RGB: {dominant_color}")
            print(f"  - Densidad bordes: {texture_analysis['edge_density']:.3f} (límite: 0.08)")
            print(f"  - Variación intensidad: {texture_analysis['intensity_std']:.1f} (límite: 80/100)")
            print(f"  - Gradiente promedio: {texture_analysis['avg_gradient']:.1f} (límite: 30)")
        else:
            print(f"[COLOR] {object_class.upper()} CONTAMINADO por color RGB: {dominant_color}")
            print(f"  - Análisis de color:")
            r, g, b = dominant_color
            print(f"    • R={r}, G={g}, B={b}")
            print(f"    • Tipo: {object_class}")
    else:
        print(f"[SANA] {object_class.upper()} SANA detectada RGB: {dominant_color}")
        print(f"  - Análisis exitoso")
        if hole_analysis['has_holes']:
            print(f"  - Nota: Se detectaron {hole_analysis['hole_count']} huecos menores (confianza: {hole_analysis['hole_confidence']:.2f})")
        if contour_analysis['is_irregular']:
            print(f"  - Nota: Contornos ligeramente irregulares (solidez: {contour_analysis['contour_solidity']:.3f})")
    
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

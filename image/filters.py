"""
Sistema de filtros de imagen optimizado para detección de contaminación en castañas
Incluye soporte GPU/CPU automático y filtros especializados
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Callable, Optional, Tuple

# Imports para filtros técnicos avanzados
try:
    from skimage import filters as sk_filters
    from skimage.feature import local_binary_pattern
    from skimage.filters import gabor
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARNING] scikit-image no disponible. Algunos filtros técnicos no funcionarán.")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy no disponible. Algunas funciones avanzadas no funcionarán.")


class FilterManager:
    """Gestor centralizado de filtros de imagen con soporte GPU/CPU"""
    
    def __init__(self):
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.filter_functions = self._initialize_filters()
        self.filter_config = self._load_filter_config()
        self.default_filter = self._get_default_filter()
    
    @staticmethod
    def _safe_multiply_channel(img: np.ndarray, channel: int, factor: float) -> np.ndarray:
        """Multiplicar un canal de imagen de forma segura"""
        result = img.copy()
        channel_data = result[:, :, channel].astype(np.float32)
        channel_data = np.clip(channel_data * factor, 0, 255)
        result[:, :, channel] = channel_data.astype(np.uint8)
        return result
        
    def _initialize_filters(self) -> Dict[str, Callable]:
        """Inicializar diccionario de filtros disponibles"""
        return {
            # Filtros básicos para grietas y hongos
            "crack": self._simulate_crack_detection_filter,
            "mold": FilterManager._simulate_mold_detection_filter,
            "fungal": FilterManager._simulate_fungal_detection_filter,
            "mold_texture": self._simulate_mold_texture_detection_filter,
            "spore_detection": self._simulate_spore_detection_filter,
            "uv_fluorescence": self._simulate_uv_fluorescence_filter,
            
            # Filtros técnicos avanzados
            "clahe_enhanced": self._clahe_enhanced_filter,
            "gabor_texture": self._gabor_texture_filter,
            "lbp_microtexture": self._lbp_microtexture_filter,
            "color_segmentation": self._color_segmentation_filter,
            "entropy_variance": self._entropy_variance_filter,
        }
    
    def apply_filter(self, img: np.ndarray, filter_type: str) -> Tuple[np.ndarray, str, str]:
        """
        Aplicar filtro específico a la imagen
        
        Args:
            img: Imagen de entrada
            filter_type: Tipo de filtro a aplicar
            
        Returns:
            Tuple[imagen_procesada, nombre_filtro, descripción]
        """
        if img is None:
            return img, "N/A", "Error: Imagen nula"
            
        filter_func = self.filter_functions.get(filter_type)
        if not filter_func:
            # Fallback a filtro de hongos si el filtro solicitado no existe
            filter_func = self.filter_functions.get("fungal")
            filter_type = "fungal"
        
        try:
            processed_img = filter_func(img)
            # Usar información desde configuración JSON si está disponible
            filter_info = self.get_filter_info_from_config(filter_type)
            return processed_img, filter_info["name"], filter_info["description"]
        except Exception as e:
            print(f"[WARNING] Error aplicando filtro {filter_type}: {e}")
            # Fallback a imagen original
            return img, "ERROR", f"Error: {str(e)}"
    
    def _get_filter_info(self, filter_type: str) -> Dict[str, str]:
        """Obtener información del filtro"""
        filter_info = {
            "crack": {"name": "GRIETAS", "description": "(Detección de fracturas con polarización)"},
            "mold": {"name": "MOHO", "description": "(Detección de moho)"},
            "fungal": {"name": "HONGOS", "description": "(Esporas y micelio)"},
            "mold_texture": {"name": "TEXTURA MOHO", "description": "(Texturas específicas de moho)"},
            "spore_detection": {"name": "ESPORAS", "description": "(Detección de esporas fúngicas)"},
            "uv_fluorescence": {"name": "UV-A FLUORESCENCIA", "description": "(Simulación UV-A ~365nm para fluorescencia)"},
            "clahe_enhanced": {"name": "CLAHE AVANZADO", "description": "(Contrast Limited Adaptive Histogram Equalization)"},
            "gabor_texture": {"name": "GABOR TEXTURAS", "description": "(Filtros Gabor para texturas fúngicas)"},
            "lbp_microtexture": {"name": "LBP MICROTEXTURAS", "description": "(Local Binary Patterns para microtexturas)"},
            "color_segmentation": {"name": "SEGMENTACIÓN COLOR", "description": "(Detección de manchas por diferencias HSV/LAB)"},
            "entropy_variance": {"name": "ENTROPÍA/VARIANZA", "description": "(Filtros de entropía y varianza local)"}
        }
        return filter_info.get(filter_type, {"name": "DESCONOCIDO", "description": ""})
    
    def get_available_filters(self) -> list:
        """Obtener lista de filtros disponibles"""
        return list(self.filter_functions.keys())
    
    def _load_filter_config(self) -> Dict:
        """Cargar configuración de filtros desde JSON"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, "filter_config.json")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print("[WARNING] Archivo filter_config.json no encontrado. Usando configuración por defecto.")
            return self._get_default_config()
        except json.JSONDecodeError:
            print("[ERROR] Error al leer filter_config.json. Usando configuración por defecto.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuración por defecto si no se encuentra el archivo"""
        return {
            "filters": {
                "nir": {"name": "NIR Avanzado", "is_favorite": True, "category": "básico"},
                "uv": {"name": "UV Simulado", "is_favorite": False, "category": "básico"}
            },
            "settings": {
                "default_favorite": "nir",
                "show_categories": True
            }
        }
    
    def _get_default_filter(self) -> str:
        """Obtener filtro por defecto desde configuración"""
        if self.filter_config and "settings" in self.filter_config:
            default = self.filter_config["settings"].get("default_favorite", "nir")
            # Verificar que el filtro por defecto existe
            if default in self.filter_functions:
                return default
        return "nir"  # Fallback
    
    def get_filter_info_from_config(self, filter_type: str) -> Dict[str, str]:
        """Obtener información del filtro desde configuración JSON"""
        if self.filter_config and "filters" in self.filter_config:
            filter_info = self.filter_config["filters"].get(filter_type, {})
            if filter_info:
                return {
                    "name": filter_info.get("name", filter_type.upper()),
                    "description": f"({filter_info.get('description', 'Sin descripción')})",
                    "category": filter_info.get("category", "desconocido"),
                    "recommended_for": filter_info.get("recommended_for", "")
                }
        
        # Fallback a método original
        return self._get_filter_info(filter_type)
    
    def get_filters_by_category(self) -> Dict[str, list]:
        """Obtener filtros agrupados por categoría"""
        categories = {}
        
        if self.filter_config and "filters" in self.filter_config:
            for filter_type, filter_info in self.filter_config["filters"].items():
                category = filter_info.get("category", "otros")
                if category not in categories:
                    categories[category] = []
                categories[category].append({
                    "type": filter_type,
                    "name": filter_info.get("name", filter_type),
                    "is_favorite": filter_info.get("is_favorite", False),
                    "description": filter_info.get("description", "")
                })
        
        return categories
    
    def get_favorite_filters(self) -> list:
        """Obtener lista de filtros marcados como favoritos"""
        favorites = []
        
        if self.filter_config and "filters" in self.filter_config:
            for filter_type, filter_info in self.filter_config["filters"].items():
                if filter_info.get("is_favorite", False):
                    favorites.append(filter_type)
        
        return favorites
    
    def get_quick_key_mapping(self) -> Dict[str, str]:
        """Obtener mapeo de teclas rápidas (tecla -> filtro)"""
        quick_keys = {}
        
        if self.filter_config and "filters" in self.filter_config:
            for filter_type, filter_info in self.filter_config["filters"].items():
                quick_key = filter_info.get("quick_key")
                if quick_key:
                    quick_keys[quick_key] = filter_type
        
        return quick_keys
    
    def get_filter_by_quick_key(self, key: str) -> Optional[str]:
        """Obtener tipo de filtro por tecla rápida"""
        quick_keys = self.get_quick_key_mapping()
        return quick_keys.get(key)
    
    def get_quick_keys_help(self) -> str:
        """Generar texto de ayuda para teclas rápidas"""
        help_text = "Teclas rápidas de filtros:\n"
        
        if self.filter_config and "filters" in self.filter_config:
            for filter_type, filter_info in self.filter_config["filters"].items():
                quick_key = filter_info.get("quick_key")
                if quick_key:
                    name = filter_info.get("name", filter_type)
                    help_text += f"   - '{quick_key}': {name}\n"
        
        return help_text.strip()
    
    def _simulate_uv_effect(self, img: np.ndarray) -> np.ndarray:
        """Simular efecto de luz UV para detección de contaminación"""
        if img is None:
            return img
        
        # 1. Aumentar saturación para simular fluorescencia
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)  # Doblar saturación
        
        # 2. Aplicar tinte azul/púrpura característico de UV
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Aplicar tinte azul/púrpura
        uv_img = enhanced_img.astype(np.float32)
        uv_img = self._safe_multiply_channel(uv_img, 0, 1.5)  # Más azul
        uv_img = self._safe_multiply_channel(uv_img, 1, 0.7)  # Menos verde
        uv_img = self._safe_multiply_channel(uv_img, 2, 0.8)  # Menos rojo
        uv_img = np.clip(uv_img, 0, 255).astype(np.uint8)
        
        # 4. Realzar bordes para simular fluorescencia
        gray = cv2.cvtColor(uv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 5. Combinar imagen UV con bordes realzados
        uv_img = cv2.addWeighted(uv_img, 0.8, edges_colored, 0.2, 0)
        
        return uv_img
    
    def _simulate_advanced_nir_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro NIR optimizado usando OpenCV GPU cuando está disponible"""
        if img is None:
            return img
        
        if self.cuda_available:
            try:
                # 1. Subir imagen a GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                
                # 2. Convertir a escala de grises en GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                
                # 3. Aplicar filtro Sobel en GPU
                gpu_sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0)
                gpu_sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1)
                
                sobel_x = cv2.cuda_GpuMat()
                sobel_y = cv2.cuda_GpuMat()
                gpu_sobel_x.apply(gpu_gray, sobel_x)
                gpu_sobel_y.apply(gpu_gray, sobel_y)
                
                # 4. Calcular magnitud del gradiente en GPU
                gpu_magnitude = cv2.cuda_GpuMat()
                cv2.cuda.magnitude(sobel_x, sobel_y, gpu_magnitude)
                
                # 5. Aplicar filtro Gaussiano en GPU
                gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
                gpu_blurred = cv2.cuda_GpuMat()
                gpu_blur.apply(gpu_gray, gpu_blurred)
                
                # 6. Detectar bordes con Canny en GPU
                gpu_canny = cv2.cuda.createCannyEdgeDetector(50, 150)
                gpu_edges = cv2.cuda_GpuMat()
                gpu_canny.detect(gpu_gray, gpu_edges)
                
                # 7. Bajar resultados a CPU para procesamiento adicional
                magnitude = gpu_magnitude.download()
                edges = gpu_edges.download()
                
                # 8. Aplicar tinte NIR (en CPU, pero vectorizado)
                nir_img = img.astype(np.float32)
                nir_img[:, :, 1] *= 1.4  # Más verde
                nir_img[:, :, 0] *= 0.6  # Menos azul
                nir_img[:, :, 2] *= 0.7  # Menos rojo
                nir_img = np.clip(nir_img, 0, 255).astype(np.uint8)
                
                # 9. Crear máscara de textura (vectorizado)
                texture_mask = (magnitude > 10) | (edges > 0)
                contaminated_areas = np.where(texture_mask[..., np.newaxis], nir_img, 0)
                nir_img = cv2.addWeighted(nir_img, 0.7, contaminated_areas, 0.3, 0)
                
                return nir_img
                
            except Exception as e:
                print(f"[WARNING] Error en GPU, usando CPU: {e}")
                # Fallback a CPU
                pass
        
        # Versión CPU optimizada (sin mahotas ni skimage pesados)
        # 1. Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Detección de bordes optimizada (OpenCV puro)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 3. Detectar bordes con Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # 4. Operaciones morfológicas simples
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 5. Aplicar tinte NIR (vectorizado)
        nir_img = img.astype(np.float32)
        nir_img[:, :, 1] *= 1.4  # Más verde
        nir_img[:, :, 0] *= 0.6  # Menos azul
        nir_img[:, :, 2] *= 0.7  # Menos rojo
        nir_img = np.clip(nir_img, 0, 255).astype(np.uint8)
        
        # 6. Crear máscara de textura (vectorizado)
        texture_mask = (magnitude > 10) | (edges_processed > 0)
        contaminated_areas = np.where(texture_mask[..., np.newaxis], nir_img, 0)
        nir_img = cv2.addWeighted(nir_img, 0.7, contaminated_areas, 0.3, 0)
        
        return nir_img
    
    def _simulate_advanced_spectral_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro espectral optimizado usando OpenCV GPU cuando está disponible"""
        if img is None:
            return img
        
        if self.cuda_available:
            try:
                # 1. Subir imagen a GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                
                # 2. Separar canales en GPU
                gpu_b, gpu_g, gpu_r = cv2.cuda.split(gpu_img)
                
                # 3. Calcular NDVI en GPU (vectorizado)
                gpu_g_float = cv2.cuda_GpuMat()
                gpu_r_float = cv2.cuda_GpuMat()
                gpu_g.convertTo(gpu_g_float, cv2.CV_32F)
                gpu_r.convertTo(gpu_r_float, cv2.CV_32F)
                
                # Calcular NDVI: (g - r) / (g + r + epsilon)
                gpu_sum = cv2.cuda_GpuMat()
                gpu_diff = cv2.cuda_GpuMat()
                cv2.cuda.add(gpu_g_float, gpu_r_float, gpu_sum)
                cv2.cuda.subtract(gpu_g_float, gpu_r_float, gpu_diff)
                
                # Evitar división por cero
                gpu_epsilon = cv2.cuda_GpuMat(gpu_sum.size(), cv2.CV_32F, 1.0)
                cv2.cuda.add(gpu_sum, gpu_epsilon, gpu_sum)
                
                gpu_ndvi = cv2.cuda_GpuMat()
                cv2.cuda.divide(gpu_diff, gpu_sum, gpu_ndvi)
                
                # 4. Calcular diferencia espectral en GPU
                gpu_spectral_diff = cv2.cuda_GpuMat()
                cv2.cuda.absdiff(gpu_g, gpu_r, gpu_spectral_diff)
                
                # 5. Bajar a CPU para operaciones complejas
                ndvi = gpu_ndvi.download()
                spectral_diff = gpu_spectral_diff.download()
                
                # 6. Aplicar tinte espectral (vectorizado)
                spectral_img = img.astype(np.float32)
                spectral_img[:, :, 0] *= 1.3  # Más azul
                spectral_img[:, :, 1] *= 1.2  # Más verde
                spectral_img[:, :, 2] *= 0.5  # Menos rojo
                spectral_img = np.clip(spectral_img, 0, 255).astype(np.uint8)
                
                # 7. Detectar anomalías espectrales (vectorizado)
                threshold = np.percentile(spectral_diff, 80)  # Más rápido que Otsu
                anomaly_mask = spectral_diff > threshold
                
                # 8. Realzar anomalías
                contaminated_areas = np.where(anomaly_mask[..., np.newaxis], spectral_img, 0)
                spectral_img = cv2.addWeighted(spectral_img, 0.6, contaminated_areas, 0.4, 0)
                
                return spectral_img
                
            except Exception as e:
                print(f"[WARNING] Error en GPU, usando CPU: {e}")
                # Fallback a CPU
                pass
        
        # Versión CPU optimizada (sin skimage)
        # 1. Análisis de canales de color (vectorizado)
        b, g, r = cv2.split(img)
        
        # 2. Calcular NDVI (vectorizado)
        g_float = g.astype(np.float32)
        r_float = r.astype(np.float32)
        ndvi = np.where((g_float + r_float) > 0, (g_float - r_float) / (g_float + r_float), 0)
        
        # 3. Detección de anomalías espectrales (vectorizado)
        spectral_diff = cv2.absdiff(g, r)
        threshold = np.percentile(spectral_diff, 80)  # Más rápido que Otsu
        anomaly_mask = spectral_diff > threshold
        
        # 4. Aplicar tinte espectral (vectorizado)
        spectral_img = img.astype(np.float32)
        spectral_img[:, :, 0] *= 1.3  # Más azul
        spectral_img[:, :, 1] *= 1.2  # Más verde
        spectral_img[:, :, 2] *= 0.5  # Menos rojo
        spectral_img = np.clip(spectral_img, 0, 255).astype(np.uint8)
        
        # 5. Realzar anomalías espectrales (vectorizado)
        contaminated_areas = np.where(anomaly_mask[..., np.newaxis], spectral_img, 0)
        spectral_img = cv2.addWeighted(spectral_img, 0.6, contaminated_areas, 0.4, 0)
        
        return spectral_img
    
    def _simulate_advanced_texture_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro de textura optimizado usando OpenCV GPU cuando está disponible"""
        if img is None:
            return img
        
        if self.cuda_available:
            try:
                # 1. Subir imagen a GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                
                # 2. Convertir a escala de grises en GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                
                # 3. Análisis de gradientes en GPU
                gpu_sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0)
                gpu_sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1)
                
                sobel_x = cv2.cuda_GpuMat()
                sobel_y = cv2.cuda_GpuMat()
                gpu_sobel_x.apply(gpu_gray, sobel_x)
                gpu_sobel_y.apply(gpu_gray, sobel_y)
                
                # 4. Calcular magnitud del gradiente en GPU
                gpu_magnitude = cv2.cuda_GpuMat()
                cv2.cuda.magnitude(sobel_x, sobel_y, gpu_magnitude)
                
                # 5. Aplicar filtro Gaussiano en GPU (simula texture_irregularity)
                gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (5, 5), 2.0)
                gpu_smoothed = cv2.cuda_GpuMat()
                gpu_gaussian.apply(gpu_magnitude, gpu_smoothed)
                
                # 6. Aplicar CLAHE en GPU
                gpu_clahe = cv2.cuda.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                gpu_enhanced = cv2.cuda_GpuMat()
                gpu_clahe.apply(gpu_gray, gpu_enhanced)
                
                # 7. Bajar resultados a CPU para operaciones complejas
                magnitude = gpu_magnitude.download()
                smoothed = gpu_smoothed.download()
                
                # 8. Aplicar tinte especial (vectorizado)
                texture_img = img.astype(np.float32)
                texture_img[:, :, 0] *= 0.8  # Menos azul
                texture_img[:, :, 1] *= 1.3  # Más verde
                texture_img[:, :, 2] *= 1.1  # Más rojo
                texture_img = np.clip(texture_img, 0, 255).astype(np.uint8)
                
                # 9. Detectar textura irregular (vectorizado)
                threshold = np.percentile(smoothed, 80)
                irregular_mask = smoothed > threshold
                
                # 10. Realzar áreas con textura irregular (vectorizado)
                contaminated_areas = np.where(irregular_mask[..., np.newaxis], texture_img, 0)
                texture_img = cv2.addWeighted(texture_img, 0.7, contaminated_areas, 0.3, 0)
                
                return texture_img
                
            except Exception as e:
                print(f"[WARNING] Error en GPU, usando CPU: {e}")
                # Fallback a CPU
                pass
        
        # Versión CPU optimizada (sin skimage)
        # 1. Análisis de textura simplificado
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Análisis de gradientes usando OpenCV (vectorizado)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. Aplicar filtro Gaussiano (vectorizado)
        texture_irregularity = cv2.GaussianBlur(gradient_magnitude.astype(np.uint8), (5, 5), 2.0)
        
        # 4. Aplicar filtro de contraste adaptativo
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # 5. Crear imagen de salida con tinte especial (vectorizado)
        texture_img = img.astype(np.float32)
        texture_img[:, :, 0] *= 0.8  # Menos azul
        texture_img[:, :, 1] *= 1.3  # Más verde
        texture_img[:, :, 2] *= 1.1  # Más rojo
        texture_img = np.clip(texture_img, 0, 255).astype(np.uint8)
        
        # 6. Realzar áreas con textura irregular (vectorizado)
        threshold = np.percentile(texture_irregularity, 80)
        irregular_mask = texture_irregularity > threshold
        contaminated_areas = np.where(irregular_mask[..., np.newaxis], texture_img, 0)
        texture_img = cv2.addWeighted(texture_img, 0.7, contaminated_areas, 0.3, 0)
        
        return texture_img
    
    def _simulate_contrast_enhanced_filter(self, img: np.ndarray) -> np.ndarray:
        """Simular filtro de contraste mejorado para detección visual"""
        if img is None:
            return img
        
        # 1. Convertir a LAB para mejor control de luminosidad
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 3. Recombinar canales
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 4. Aplicar filtro de realce de bordes
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        
        # 5. Ajustar saturación para realzar diferencias de color
        hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced_img
    
    @staticmethod
    def _simulate_mold_detection_filter(img: np.ndarray) -> np.ndarray:
        """Filtro especializado para detección de moho en castañas"""
        if img is None:
            return img
        
        # 1. Detección de manchas oscuras (moho típico)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Detectar áreas muy oscuras (moho negro)
        _, dark_mold = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Detectar áreas verdes/azules (moho coloreado)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Rango para moho verde/azul
        lower_mold = np.array([40, 50, 50])
        upper_mold = np.array([120, 255, 200])
        colored_mold = cv2.inRange(hsv, lower_mold, upper_mold)
        
        # 4. Combinar ambas detecciones
        mold_mask = cv2.bitwise_or(dark_mold, colored_mold)
        
        # 5. Operaciones morfológicas para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_CLOSE, kernel)
        mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_OPEN, kernel)
        
        # 6. Aplicar tinte rojizo para resaltar moho
        mold_img = img.copy()
        mold_img = FilterManager._safe_multiply_channel(mold_img, 2, 1.5)  # Más rojo
        mold_img = FilterManager._safe_multiply_channel(mold_img, 1, 0.7)  # Menos verde
        mold_img = FilterManager._safe_multiply_channel(mold_img, 0, 0.7)  # Menos azul
        
        # 7. Realzar áreas con moho detectado
        mold_areas = cv2.bitwise_and(mold_img, mold_img, mask=mold_mask)
        mold_img = cv2.addWeighted(mold_img, 0.6, mold_areas, 0.4, 0)
        
        return mold_img

    def _simulate_crack_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filtro especializado para detección de grietas/fracturas en la cáscara
        Basado en técnicas de polarización óptica y análisis de sombras
        """
        if img is None:
            return img
            
        # 1) Simulación de filtro polarizador - reduce reflejos y mejora contraste de grietas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2) Análisis de gradientes direccionales (simula detección de bordes con polarización)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # 3) Detección de líneas direccionales (grietas tienden a ser lineales)
        # Usar transformada de Hough para detectar líneas
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        # 4) Crear máscara de grietas basada en líneas detectadas
        crack_mask = np.zeros(gray.shape, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(crack_mask, (x1, y1), (x2, y2), 255, 3)
        
        # 5) Análisis de sombras (grietas crean sombras características)
        # Usar filtro Laplaciano para detectar cambios bruscos de intensidad
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 6) Combinar detección de líneas con análisis de sombras
        shadow_mask = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)[1]
        combined_mask = cv2.bitwise_or(crack_mask, shadow_mask)
        
        # 7) Morfología para conectar segmentos de grietas
        kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 1))
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_line)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_connect)
        
        # 8) Aplicar tinte azulado (simula contraste mejorado con polarización)
        crack_img = img.copy()
        crack_img = FilterManager._safe_multiply_channel(crack_img, 0, 1.4)  # Más azul
        crack_img = FilterManager._safe_multiply_channel(crack_img, 1, 0.8)  # Menos verde
        crack_img = FilterManager._safe_multiply_channel(crack_img, 2, 0.7)  # Menos rojo
        
        # 9) Realce localizado en las grietas detectadas
        crack_areas = cv2.bitwise_and(crack_img, crack_img, mask=combined_mask)
        crack_img = cv2.addWeighted(crack_img, 0.7, crack_areas, 0.3, 0)
        
        return crack_img
    
    def _simulate_uv_fluorescence_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filtro UV-A simulado (~365nm) para detección de fluorescencia de moho
        Basado en principios de fluorescencia de metabolitos fúngicos
        """
        if img is None:
            return img
            
        # 1) Simulación de iluminación UV-A (~365nm)
        # Los metabolitos de hongos suelen fluorescer en azul-verde bajo UV-A
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 2) Realzar saturación para simular fluorescencia
        s_enhanced = np.clip(s.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
        
        # 3) Aplicar filtro de paso de banda simulado (BP470-505nm para fluorescencia azul-verde)
        # Simular que solo pasa la fluorescencia emitida
        enhanced_hsv = cv2.merge([h, s_enhanced, v])
        base_fluorescence = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # 4) Detectar áreas con fluorescencia característica
        # Las áreas con moho suelen tener fluorescencia azul-verde
        lab = cv2.cvtColor(base_fluorescence, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 5) Detectar fluorescencia azul-verde (canal b* alto, canal a* bajo)
        blue_green_fluorescence = cv2.inRange(lab, (0, 100, 120), (255, 255, 255))
        
        # 6) Detectar fluorescencia verde-amarilla (alternativa)
        green_yellow_fluorescence = cv2.inRange(lab, (0, 80, 100), (255, 255, 255))
        
        # 7) Combinar detecciones de fluorescencia
        fluorescence_mask = cv2.bitwise_or(blue_green_fluorescence, green_yellow_fluorescence)
        
        # 8) Aplicar tinte característico de fluorescencia UV
        uv_img = base_fluorescence.copy().astype(np.float32)
        # Realzar canales azul y verde (fluorescencia típica)
        uv_img[:, :, 0] *= 1.6  # Más azul
        uv_img[:, :, 1] *= 1.4  # Más verde
        uv_img[:, :, 2] *= 0.3  # Menos rojo (típico de fluorescencia)
        uv_img = np.clip(uv_img, 0, 255).astype(np.uint8)
        
        # 9) Realzar áreas con fluorescencia detectada
        fluorescent_areas = cv2.bitwise_and(uv_img, uv_img, mask=fluorescence_mask)
        uv_img = cv2.addWeighted(uv_img, 0.5, fluorescent_areas, 0.5, 0)
        
        # 10) Simular ambiente oscuro necesario para UV (reducir brillo general)
        uv_img = cv2.addWeighted(uv_img, 0.8, np.zeros_like(uv_img), 0.2, 0)
        
        return uv_img
    
    
    @staticmethod
    def _simulate_rot_detection_filter(img: np.ndarray) -> np.ndarray:
        """Filtro para detección de podredumbre y vencimiento"""
        if img is None:
            return img
        
        # 1. Análisis de textura para detectar suavización
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Detectar áreas con baja textura (podredumbre suaviza la superficie)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, low_texture = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Detectar cambios de color (amarronamiento)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 4. Detectar áreas amarronadas (alto valor en canal 'b')
        _, brown_mask = cv2.threshold(b, 140, 255, cv2.THRESH_BINARY)
        
        # 5. Combinar indicadores de podredumbre
        rot_mask = cv2.bitwise_or(low_texture, brown_mask)
        
        # 6. Aplicar tinte amarronado
        rot_img = img.copy()
        rot_img = FilterManager._safe_multiply_channel(rot_img, 2, 1.3)  # Más rojo
        rot_img = FilterManager._safe_multiply_channel(rot_img, 1, 1.2)  # Más verde
        rot_img = FilterManager._safe_multiply_channel(rot_img, 0, 0.8)  # Menos azul
        
        # 7. Realzar áreas podridas
        rot_areas = cv2.bitwise_and(rot_img, rot_img, mask=rot_mask)
        rot_img = cv2.addWeighted(rot_img, 0.7, rot_areas, 0.3, 0)
        
        return rot_img
    
    @staticmethod
    def _simulate_fungal_detection_filter(img: np.ndarray) -> np.ndarray:
        """Filtro para detección de hongos y esporas"""
        if img is None:
            return img
        
        # 1. Detección de patrones circulares (esporas)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Detectar círculos usando HoughCircles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )
        
        # 3. Crear máscara para círculos detectados
        fungal_mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(fungal_mask, (x, y), r, 255, -1)
        
        # 4. Detectar patrones de red (micelio)
        edges = cv2.Canny(gray, 50, 150)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 15))
        mycelium_pattern = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
        
        # 5. Combinar detecciones
        fungal_mask = cv2.bitwise_or(fungal_mask, mycelium_pattern)
        
        # 6. Aplicar tinte púrpura para hongos
        fungal_img = img.copy()
        fungal_img = FilterManager._safe_multiply_channel(fungal_img, 0, 1.3)  # Más azul
        fungal_img = FilterManager._safe_multiply_channel(fungal_img, 2, 1.4)  # Más rojo
        fungal_img = FilterManager._safe_multiply_channel(fungal_img, 1, 0.6)  # Menos verde
        
        # 7. Realzar áreas con hongos
        fungal_areas = cv2.bitwise_and(fungal_img, fungal_img, mask=fungal_mask)
        fungal_img = cv2.addWeighted(fungal_img, 0.7, fungal_areas, 0.3, 0)
        
        return fungal_img
    
    def _simulate_mycotoxin_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro especializado para detección de micotoxinas en castañas brasileñas"""
        if img is None:
            return img
        
        # 1. Análisis de patrones de decoloración asociados con micotoxinas
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Detectar áreas con cambios de color específicos de micotoxinas
        # Las micotoxinas suelen causar decoloraciones amarronadas/verdosas
        brown_mask = cv2.inRange(lab, (0, 100, 100), (255, 255, 255))
        green_mask = cv2.inRange(lab, (0, 0, 100), (255, 255, 255))
        
        # 3. Detectar texturas granulares características de micotoxinas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture_analysis = cv2.Laplacian(gray, cv2.CV_64F)
        texture_analysis = np.uint8(np.absolute(texture_analysis))
        
        # 4. Combinar detecciones
        mycotoxin_mask = cv2.bitwise_or(brown_mask, green_mask)
        texture_mask = cv2.threshold(texture_analysis, 30, 255, cv2.THRESH_BINARY)[1]
        combined_mask = cv2.bitwise_or(mycotoxin_mask, texture_mask)
        
        # 5. Aplicar tinte rojizo para resaltar micotoxinas
        mycotoxin_img = img.copy()
        mycotoxin_img = self._safe_multiply_channel(mycotoxin_img, 2, 1.8)  # Más rojo
        mycotoxin_img = self._safe_multiply_channel(mycotoxin_img, 1, 0.5)  # Menos verde
        mycotoxin_img = self._safe_multiply_channel(mycotoxin_img, 0, 0.3)  # Menos azul
        
        # 6. Realzar áreas con micotoxinas detectadas
        mycotoxin_areas = cv2.bitwise_and(mycotoxin_img, mycotoxin_img, mask=combined_mask)
        mycotoxin_img = cv2.addWeighted(mycotoxin_img, 0.5, mycotoxin_areas, 0.5, 0)
        
        return mycotoxin_img
    
    def _simulate_aflatoxin_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro especializado para detección de aflatoxinas (micotoxinas específicas)"""
        if img is None:
            return img
        
        # 1. Las aflatoxinas producen fluorescencia específica bajo UV
        # Simular esta fluorescencia con análisis de color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 2. Detectar colores fluorescentes (azul-verde bajo UV)
        # Aflatoxinas B1 y B2: fluorescencia azul
        # Aflatoxinas G1 y G2: fluorescencia verde-amarilla
        blue_fluorescent = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        green_fluorescent = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
        
        # 3. Detectar patrones de crecimiento en red (característico de Aspergillus)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Crear kernel para detectar patrones de red
        kernel_network = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        network_pattern = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_network)
        
        # 4. Combinar detecciones
        aflatoxin_mask = cv2.bitwise_or(blue_fluorescent, green_fluorescent)
        aflatoxin_mask = cv2.bitwise_or(aflatoxin_mask, network_pattern)
        
        # 5. Aplicar tinte azul-verde fluorescente
        aflatoxin_img = img.copy()
        aflatoxin_img = self._safe_multiply_channel(aflatoxin_img, 0, 1.5)  # Más azul
        aflatoxin_img = self._safe_multiply_channel(aflatoxin_img, 1, 1.3)  # Más verde
        aflatoxin_img = self._safe_multiply_channel(aflatoxin_img, 2, 0.2)  # Menos rojo
        
        # 6. Realzar fluorescencia
        aflatoxin_areas = cv2.bitwise_and(aflatoxin_img, aflatoxin_img, mask=aflatoxin_mask)
        aflatoxin_img = cv2.addWeighted(aflatoxin_img, 0.4, aflatoxin_areas, 0.6, 0)
        
        return aflatoxin_img
    
    def _simulate_discoloration_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro para detectar patrones de decoloración fúngica específicos"""
        if img is None:
            return img
        
        # Verificación de debug
        print(f"[DEBUG] Discoloration filter - Image shape: {img.shape}, dtype: {img.dtype}")
        
        # 1. Análisis multi-espectral de decoloraciones
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Detectar patrones de decoloración específicos
        # Decoloración por hongos: cambios en canales a* y b*
        print("[DEBUG] Calculando discoloration_a...")
        discoloration_a = cv2.threshold(np.abs(a - np.mean(a)), 20, 255, cv2.THRESH_BINARY)[1]
        print("[DEBUG] Calculando discoloration_b...")
        discoloration_b = cv2.threshold(np.abs(b - np.mean(b)), 25, 255, cv2.THRESH_BINARY)[1]
        
        # 3. Detectar patrones de difusión (hongos se extienden gradualmente)
        print("[DEBUG] Convirtiendo a escala de grises...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("[DEBUG] Calculando gradiente Sobel...")
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        print("[DEBUG] Convirtiendo gradiente a uint8...")
        gradient = np.uint8(np.absolute(gradient))
        
        # 4. Crear máscara de difusión
        print("[DEBUG] Creando máscara de difusión...")
        diffusion_mask = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)[1]
        
        # 5. Combinar detecciones
        print(f"[DEBUG] discoloration_a shape: {discoloration_a.shape}, dtype: {discoloration_a.dtype}")
        print(f"[DEBUG] discoloration_b shape: {discoloration_b.shape}, dtype: {discoloration_b.dtype}")
        print(f"[DEBUG] diffusion_mask shape: {diffusion_mask.shape}, dtype: {diffusion_mask.dtype}")
        
        print("[DEBUG] Combinando discoloration_a y discoloration_b...")
        discoloration_mask = cv2.bitwise_or(discoloration_a, discoloration_b)
        print("[DEBUG] Combinando con diffusion_mask...")
        discoloration_mask = cv2.bitwise_or(discoloration_mask, diffusion_mask)
        print("[DEBUG] Máscaras combinadas exitosamente")
        
        # 6. Aplicar tinte amarillo-naranja para resaltar decoloraciones
        print("[DEBUG] Aplicando tinte a imagen...")
        discoloration_img = img.copy()
        print(f"[DEBUG] Imagen copiada - shape: {discoloration_img.shape}, dtype: {discoloration_img.dtype}")
        
        discoloration_img = FilterManager._safe_multiply_channel(discoloration_img, 2, 1.4)  # Más rojo
        print("[DEBUG] Canal rojo procesado")
        discoloration_img = FilterManager._safe_multiply_channel(discoloration_img, 1, 1.2)  # Más verde
        print("[DEBUG] Canal verde procesado")
        discoloration_img = FilterManager._safe_multiply_channel(discoloration_img, 0, 0.8)  # Menos azul
        print("[DEBUG] Canal azul procesado")
        
        # 7. Realzar áreas decoloradas
        discoloration_areas = cv2.bitwise_and(discoloration_img, discoloration_img, mask=discoloration_mask)
        discoloration_img = cv2.addWeighted(discoloration_img, 0.6, discoloration_areas, 0.4, 0)
        
        return discoloration_img
    
    def _simulate_mold_texture_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro para detectar texturas específicas de diferentes tipos de moho"""
        if img is None:
            return img
        
        # 1. Análisis de textura avanzado
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Detectar texturas granulares (Penicillium)
        kernel_granular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        granular_texture = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_granular)
        
        # 3. Detectar texturas fibrosas (Aspergillus)
        kernel_fibrous = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 7))
        fibrous_texture = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_fibrous)
        
        # 4. Detectar texturas algodonosas (Fusarium)
        kernel_cottony = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cottony_texture = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_cottony)
        
        # 5. Combinar análisis de texturas
        texture_combined = cv2.addWeighted(granular_texture, 0.4, fibrous_texture, 0.3, 0)
        texture_combined = cv2.addWeighted(texture_combined, 1.0, cottony_texture, 0.3, 0)
        
        # 6. Crear máscara de textura de moho
        mold_texture_mask = cv2.threshold(texture_combined, 30, 255, cv2.THRESH_BINARY)[1]
        
        # 7. Aplicar tinte púrpura para diferentes tipos de moho
        mold_texture_img = img.copy()
        mold_texture_img = self._safe_multiply_channel(mold_texture_img, 0, 1.4)  # Más azul
        mold_texture_img = self._safe_multiply_channel(mold_texture_img, 2, 1.3)  # Más rojo
        mold_texture_img = self._safe_multiply_channel(mold_texture_img, 1, 0.7)  # Menos verde
        
        # 8. Realzar texturas de moho
        mold_texture_areas = cv2.bitwise_and(mold_texture_img, mold_texture_img, mask=mold_texture_mask)
        mold_texture_img = cv2.addWeighted(mold_texture_img, 0.7, mold_texture_areas, 0.3, 0)
        
        return mold_texture_img
    
    def _simulate_spore_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro para detectar esporas fúngicas específicas"""
        if img is None:
            return img
        
        # 1. Detección de esporas usando análisis de forma
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Detectar círculos pequeños (esporas)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
            param1=30, param2=20, minRadius=2, maxRadius=15
        )
        
        # 3. Crear máscara de esporas
        spore_mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(spore_mask, (x, y), r, 255, -1)
        
        # 4. Detectar patrones de agrupación de esporas
        contours, _ = cv2.findContours(spore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filtrar por área (esporas pequeñas)
        spore_clusters = np.zeros(gray.shape, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 200:  # Tamaño típico de esporas
                cv2.fillPoly(spore_clusters, [contour], 255)
        
        # 6. Combinar detecciones
        final_spore_mask = cv2.bitwise_or(spore_mask, spore_clusters)
        
        # 7. Aplicar tinte verde fluorescente para esporas
        spore_img = img.copy()
        spore_img = self._safe_multiply_channel(spore_img, 1, 1.6)  # Más verde
        spore_img = self._safe_multiply_channel(spore_img, 0, 0.8)  # Menos azul
        spore_img = self._safe_multiply_channel(spore_img, 2, 0.4)  # Menos rojo
        
        # 8. Realzar esporas detectadas
        spore_areas = cv2.bitwise_and(spore_img, spore_img, mask=final_spore_mask)
        spore_img = cv2.addWeighted(spore_img, 0.5, spore_areas, 0.5, 0)
        
        return spore_img
    
    def _simulate_brazil_chestnut_specialized_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro especializado para castañas brasileñas (Bertholletia excelsa)"""
        if img is None:
            return img
        
        # 1. Análisis específico para características de castañas brasileñas
        # Las castañas brasileñas tienen características específicas de color y textura
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 2. Detectar color base típico de castañas brasileñas sanas
        # Color marrón-dorado característico
        healthy_color = cv2.inRange(hsv, (10, 30, 80), (30, 255, 255))
        
        # 3. Detectar desviaciones del color saludable (contaminación)
        deviation_mask = cv2.bitwise_not(healthy_color)
        
        # 4. Análisis de textura específica de la cáscara de castaña brasileña
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 5. Detectar patrones de grietas o irregularidades en la cáscara
        kernel_crack = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 10))
        crack_pattern = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_crack)
        
        # 6. Detectar manchas o decoloraciones
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        spot_mask = cv2.threshold(cv2.absdiff(l, cv2.GaussianBlur(l, (15, 15), 0)), 30, 255, cv2.THRESH_BINARY)[1]
        
        # 7. Combinar indicadores de contaminación
        contamination_mask = cv2.bitwise_or(deviation_mask, crack_pattern)
        contamination_mask = cv2.bitwise_or(contamination_mask, spot_mask)
        
        # 8. Aplicar tinte específico para castañas brasileñas
        brazil_img = img.copy()
        
        # Mantener colores naturales para áreas sanas
        healthy_areas = cv2.bitwise_and(brazil_img, brazil_img, mask=healthy_color)
        
        # Aplicar tinte rojizo para áreas contaminadas
        contaminated_areas = brazil_img.copy()
        contaminated_areas = self._safe_multiply_channel(contaminated_areas, 2, 1.5)  # Más rojo
        contaminated_areas = self._safe_multiply_channel(contaminated_areas, 1, 0.8)  # Menos verde
        contaminated_areas = self._safe_multiply_channel(contaminated_areas, 0, 0.7)  # Menos azul
        
        # 9. Combinar imagen final
        brazil_img = cv2.addWeighted(healthy_areas, 0.7, contaminated_areas, 0.3, 0)
        
        return brazil_img
    
    def _clahe_enhanced_filter(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) avanzado
        Realza contraste en zonas con moho o suciedad
        """
        if img is None:
            return img
        
        try:
            # Convertir a LAB para mejor control de luminosidad
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Aplicar CLAHE con parámetros optimizados para detección de moho
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Aplicar CLAHE adicional en canal A (verde-rojo) para detectar cambios de color
            a_enhanced = clahe.apply(a)
            
            # Recombinar canales
            enhanced_lab = cv2.merge([l_enhanced, a_enhanced, b])
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Detectar áreas con alto contraste (posible moho)
            gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # Crear máscara de áreas de alto contraste
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Aplicar tinte rojizo para resaltar áreas con moho
            clahe_img = enhanced_img.copy()
            clahe_img = self._safe_multiply_channel(clahe_img, 2, 1.3)  # Más rojo
            clahe_img = self._safe_multiply_channel(clahe_img, 1, 0.8)  # Menos verde
            clahe_img = self._safe_multiply_channel(clahe_img, 0, 0.8)  # Menos azul
            
            # Realzar áreas con alto contraste
            contrast_areas = cv2.bitwise_and(clahe_img, clahe_img, mask=edges_processed)
            clahe_img = cv2.addWeighted(clahe_img, 0.7, contrast_areas, 0.3, 0)
            
            return clahe_img
            
        except Exception as e:
            print(f"[ERROR] Error en filtro CLAHE: {e}")
            return img
    
    def _gabor_texture_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filtros Gabor para detectar texturas típicas de hongos y rugosidades
        """
        if img is None:
            return img
        
        if not SKIMAGE_AVAILABLE:
            print("[WARNING] scikit-image no disponible para filtros Gabor")
            return img
        
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar múltiples filtros Gabor con diferentes orientaciones y frecuencias
            # Orientaciones típicas para detectar texturas fúngicas
            orientations = [0, 30, 60, 90, 120, 150]  # grados
            frequencies = [0.1, 0.2, 0.3]  # frecuencias espaciales
            
            gabor_responses = []
            
            for freq in frequencies:
                for orientation in orientations:
                    # Convertir orientación a radianes
                    theta = np.radians(orientation)
                    
                    # Aplicar filtro Gabor
                    filtered_real, _ = gabor(gray, frequency=freq, theta=theta)
                    
                    # Usar solo la respuesta real (magnitud)
                    gabor_responses.append(np.abs(filtered_real))
            
            # Combinar respuestas de múltiples filtros Gabor
            if gabor_responses:
                # Promediar todas las respuestas
                combined_gabor = np.mean(gabor_responses, axis=0)
                
                # Normalizar a rango 0-255
                combined_gabor = ((combined_gabor - combined_gabor.min()) / 
                                (combined_gabor.max() - combined_gabor.min()) * 255).astype(np.uint8)
                
                # Crear imagen en color con tinte púrpura para texturas fúngicas
                gabor_img = img.copy()
                gabor_img = self._safe_multiply_channel(gabor_img, 0, 1.4)  # Más azul
                gabor_img = self._safe_multiply_channel(gabor_img, 2, 1.3)  # Más rojo
                gabor_img = self._safe_multiply_channel(gabor_img, 1, 0.7)  # Menos verde
                
                # Aplicar máscara de texturas detectadas
                texture_mask = cv2.threshold(combined_gabor, 50, 255, cv2.THRESH_BINARY)[1]
                texture_areas = cv2.bitwise_and(gabor_img, gabor_img, mask=texture_mask)
                gabor_img = cv2.addWeighted(gabor_img, 0.6, texture_areas, 0.4, 0)
                
                return gabor_img
            else:
                return img
            
        except Exception as e:
            print(f"[ERROR] Error en filtros Gabor: {e}")
            return img
    
    def _lbp_microtexture_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Local Binary Patterns para analizar microtexturas del cascarón
        Detecta cambios en microtextura causados por moho
        """
        if img is None:
            return img
        
        if not SKIMAGE_AVAILABLE:
            print("[WARNING] scikit-image no disponible para LBP")
            return img
        
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Parámetros LBP optimizados para microtexturas de castañas
            radius = 2
            n_points = 8 * radius
            
            # Calcular LBP
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calcular histograma de LBP para detectar patrones anómalos
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalizar
            
            # Detectar áreas con patrones LBP inusuales (posible moho)
            # Las áreas con moho suelen tener patrones LBP más uniformes
            lbp_variance = np.var(lbp)
            lbp_std = np.std(lbp)
            
            # Crear máscara de áreas con microtextura anómala
            threshold = np.mean(lbp) + lbp_std
            anomaly_mask = cv2.threshold(lbp.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Aplicar tinte verde para resaltar microtexturas
            lbp_img = img.copy()
            lbp_img = self._safe_multiply_channel(lbp_img, 1, 1.4)  # Más verde
            lbp_img = self._safe_multiply_channel(lbp_img, 0, 0.8)  # Menos azul
            lbp_img = self._safe_multiply_channel(lbp_img, 2, 0.8)  # Menos rojo
            
            # Realzar áreas con microtextura anómala
            texture_areas = cv2.bitwise_and(lbp_img, lbp_img, mask=anomaly_mask)
            lbp_img = cv2.addWeighted(lbp_img, 0.7, texture_areas, 0.3, 0)
            
            return lbp_img
                
        except Exception as e:
            print(f"[ERROR] Error en filtro LBP: {e}")
            return img
    
    def _color_segmentation_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Segmentación por color para detectar manchas por diferencias en tono HSV/LAB
        """
        if img is None:
            return img
        
        try:
            # Análisis en espacio HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Análisis en espacio LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Detectar manchas oscuras (moho típico)
            dark_mask_hsv = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
            dark_mask_lab = cv2.inRange(lab, (0, 0, 0), (80, 255, 255))
            dark_mask = cv2.bitwise_or(dark_mask_hsv, dark_mask_lab)
            
            # Detectar manchas verdes/azules (moho coloreado)
            green_mask_hsv = cv2.inRange(hsv, (40, 50, 50), (80, 255, 200))
            green_mask_lab = cv2.inRange(lab, (0, 120, 80), (255, 255, 255))
            green_mask = cv2.bitwise_or(green_mask_hsv, green_mask_lab)
            
            # Detectar manchas marrones/amarillas (decaimiento)
            brown_mask_hsv = cv2.inRange(hsv, (10, 50, 50), (30, 255, 200))
            brown_mask_lab = cv2.inRange(lab, (0, 100, 120), (255, 255, 255))
            brown_mask = cv2.bitwise_or(brown_mask_hsv, brown_mask_lab)
            
            # Combinar todas las detecciones de manchas
            combined_mask = cv2.bitwise_or(dark_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
            
            # Limpiar ruido con operaciones morfológicas
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Aplicar tinte rojizo para resaltar manchas detectadas
            color_img = img.copy()
            color_img = self._safe_multiply_channel(color_img, 2, 1.5)  # Más rojo
            color_img = self._safe_multiply_channel(color_img, 1, 0.7)  # Menos verde
            color_img = self._safe_multiply_channel(color_img, 0, 0.7)  # Menos azul
            
            # Realzar áreas con manchas detectadas
            spot_areas = cv2.bitwise_and(color_img, color_img, mask=combined_mask)
            color_img = cv2.addWeighted(color_img, 0.6, spot_areas, 0.4, 0)
            
            return color_img
            
        except Exception as e:
            print(f"[ERROR] Error en segmentación por color: {e}")
            return img
    
    def _entropy_variance_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filtros de entropía y varianza para detectar áreas contaminadas
        Las áreas contaminadas suelen tener entropía local más alta
        """
        if img is None:
            return img
        
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calcular varianza local usando OpenCV
            kernel_size = 9
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            variance = sqr_mean - mean**2
            
            # Normalizar varianza
            variance_normalized = ((variance - variance.min()) / 
                                 (variance.max() - variance.min()) * 255).astype(np.uint8)
            
            # Calcular entropía local si scikit-image está disponible
            if SKIMAGE_AVAILABLE:
                try:
                    # Calcular entropía local
                    entropy_img = entropy(gray, disk(5))
                    entropy_normalized = ((entropy_img - entropy_img.min()) / 
                                        (entropy_img.max() - entropy_img.min()) * 255).astype(np.uint8)
                    
                    # Combinar varianza y entropía
                    combined_variance_entropy = cv2.addWeighted(variance_normalized, 0.6, entropy_normalized, 0.4, 0)
                except:
                    combined_variance_entropy = variance_normalized
            else:
                combined_variance_entropy = variance_normalized
            
            # Detectar áreas con alta variabilidad local
            high_variance_threshold = np.percentile(combined_variance_entropy, 85)
            high_variance_mask = cv2.threshold(combined_variance_entropy, high_variance_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Aplicar tinte naranja para resaltar áreas de alta variabilidad
            entropy_img = img.copy()
            entropy_img = self._safe_multiply_channel(entropy_img, 2, 1.4)  # Más rojo
            entropy_img = self._safe_multiply_channel(entropy_img, 1, 1.2)  # Más verde
            entropy_img = self._safe_multiply_channel(entropy_img, 0, 0.6)  # Menos azul
            
            # Realzar áreas con alta variabilidad local
            variance_areas = cv2.bitwise_and(entropy_img, entropy_img, mask=high_variance_mask)
            entropy_img = cv2.addWeighted(entropy_img, 0.7, variance_areas, 0.3, 0)
            
            return entropy_img
            
        except Exception as e:
            print(f"[ERROR] Error en filtro entropía/varianza: {e}")
            return img
    
    

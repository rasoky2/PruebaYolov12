"""
Sistema de filtros de imagen optimizado para detección de contaminación en castañas
Incluye soporte GPU/CPU automático y filtros especializados
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Callable, Optional, Tuple


class FilterManager:
    """Gestor centralizado de filtros de imagen con soporte GPU/CPU"""
    
    def __init__(self):
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.filter_functions = self._initialize_filters()
        self.filter_config = self._load_filter_config()
        self.default_filter = self._get_default_filter()
        
    def _initialize_filters(self) -> Dict[str, Callable]:
        """Inicializar diccionario de filtros disponibles"""
        return {
            "uv": self._simulate_uv_effect,
            "nir": self._simulate_advanced_nir_filter,
            "spectral": self._simulate_advanced_spectral_filter,
            "texture": self._simulate_advanced_texture_filter,
            "contrast": self._simulate_contrast_enhanced_filter,
            "mold": FilterManager._simulate_mold_detection_filter,
            "rot": FilterManager._simulate_rot_detection_filter,
            "fungal": FilterManager._simulate_fungal_detection_filter,
            # Nuevos filtros avanzados para castañas brasileñas
            "mycotoxin": self._simulate_mycotoxin_detection_filter,
            "aflatoxin": self._simulate_aflatoxin_detection_filter,
            "discoloration": self._simulate_discoloration_detection_filter,
            "mold_texture": self._simulate_mold_texture_detection_filter,
            "spore_detection": self._simulate_spore_detection_filter,
            "brazil_chestnut": self._simulate_brazil_chestnut_specialized_filter
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
            # Fallback a UV si el filtro no existe
            filter_func = self.filter_functions["uv"]
            filter_type = "uv"
        
        try:
            processed_img = filter_func(img)
            # Usar información desde configuración JSON si está disponible
            filter_info = self.get_filter_info_from_config(filter_type)
            return processed_img, filter_info["name"], filter_info["description"]
        except Exception as e:
            print(f"⚠️ Error aplicando filtro {filter_type}: {e}")
            # Fallback a imagen original
            return img, "ERROR", f"Error: {str(e)}"
    
    def _get_filter_info(self, filter_type: str) -> Dict[str, str]:
        """Obtener información del filtro"""
        filter_info = {
            "uv": {"name": "UV SIMULADO", "description": "(Fluorescencia)"},
            "nir": {"name": "NIR AVANZADO", "description": "(Infrarrojo cercano)"},
            "spectral": {"name": "ESPECTRAL", "description": "(Multibanda)"},
            "texture": {"name": "TEXTURA", "description": "(Análisis de patrones)"},
            "contrast": {"name": "CONTRASTE", "description": "(Realce visual)"},
            "mold": {"name": "MOHO", "description": "(Detección de moho)"},
            "rot": {"name": "PODREDUMBRE", "description": "(Detección de vencimiento)"},
            "fungal": {"name": "HONGOS", "description": "(Esporas y micelio)"},
            # Información de filtros avanzados para castañas brasileñas
            "mycotoxin": {"name": "MICOTOXINAS", "description": "(Detección de toxinas fúngicas)"},
            "aflatoxin": {"name": "AFLATOXINAS", "description": "(Detección de aflatoxinas)"},
            "discoloration": {"name": "DECOLORACIÓN", "description": "(Patrones de decoloración fúngica)"},
            "mold_texture": {"name": "TEXTURA MOHO", "description": "(Texturas específicas de moho)"},
            "spore_detection": {"name": "ESPORAS", "description": "(Detección de esporas fúngicas)"},
            "brazil_chestnut": {"name": "CASTAÑA BRASILEÑA", "description": "(Análisis especializado)"}
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
    
    def _simulate_uv_effect(self, img: np.ndarray) -> np.ndarray:
        """Simular efecto de luz UV para detección de contaminación"""
        if img is None:
            return img
        
        # 1. Aumentar saturación para simular fluorescencia
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 2.0)  # Doblar saturación
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # 2. Aplicar tinte azul/púrpura característico de UV
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Aplicar tinte azul/púrpura
        uv_img = enhanced_img.astype(np.float32)
        uv_img[:, :, 0] = cv2.multiply(uv_img[:, :, 0], 1.5)  # Más azul
        uv_img[:, :, 1] = cv2.multiply(uv_img[:, :, 1], 0.7)  # Menos verde
        uv_img[:, :, 2] = cv2.multiply(uv_img[:, :, 2], 0.8)  # Menos rojo
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
                print(f"⚠️ Error en GPU, usando CPU: {e}")
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
                print(f"⚠️ Error en GPU, usando CPU: {e}")
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
                print(f"⚠️ Error en GPU, usando CPU: {e}")
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
        s = cv2.multiply(s, 1.5)
        s = np.clip(s, 0, 255).astype(np.uint8)
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
        mold_img[:, :, 2] = cv2.multiply(mold_img[:, :, 2], 1.5)  # Más rojo
        mold_img[:, :, 1] = cv2.multiply(mold_img[:, :, 1], 0.7)  # Menos verde
        mold_img[:, :, 0] = cv2.multiply(mold_img[:, :, 0], 0.7)  # Menos azul
        
        # 7. Realzar áreas con moho detectado
        mold_areas = cv2.bitwise_and(mold_img, mold_img, mask=mold_mask)
        mold_img = cv2.addWeighted(mold_img, 0.6, mold_areas, 0.4, 0)
        
        return mold_img
    
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
        rot_img[:, :, 2] = cv2.multiply(rot_img[:, :, 2], 1.3)  # Más rojo
        rot_img[:, :, 1] = cv2.multiply(rot_img[:, :, 1], 1.2)  # Más verde
        rot_img[:, :, 0] = cv2.multiply(rot_img[:, :, 0], 0.8)  # Menos azul
        
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
        fungal_img[:, :, 0] = cv2.multiply(fungal_img[:, :, 0], 1.3)  # Más azul
        fungal_img[:, :, 2] = cv2.multiply(fungal_img[:, :, 2], 1.4)  # Más rojo
        fungal_img[:, :, 1] = cv2.multiply(fungal_img[:, :, 1], 0.6)  # Menos verde
        
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
        mycotoxin_img[:, :, 2] = cv2.multiply(mycotoxin_img[:, :, 2], 1.8)  # Más rojo
        mycotoxin_img[:, :, 1] = cv2.multiply(mycotoxin_img[:, :, 1], 0.5)  # Menos verde
        mycotoxin_img[:, :, 0] = cv2.multiply(mycotoxin_img[:, :, 0], 0.3)  # Menos azul
        
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
        aflatoxin_img[:, :, 0] = cv2.multiply(aflatoxin_img[:, :, 0], 1.5)  # Más azul
        aflatoxin_img[:, :, 1] = cv2.multiply(aflatoxin_img[:, :, 1], 1.3)  # Más verde
        aflatoxin_img[:, :, 2] = cv2.multiply(aflatoxin_img[:, :, 2], 0.2)  # Menos rojo
        
        # 6. Realzar fluorescencia
        aflatoxin_areas = cv2.bitwise_and(aflatoxin_img, aflatoxin_img, mask=aflatoxin_mask)
        aflatoxin_img = cv2.addWeighted(aflatoxin_img, 0.4, aflatoxin_areas, 0.6, 0)
        
        return aflatoxin_img
    
    def _simulate_discoloration_detection_filter(self, img: np.ndarray) -> np.ndarray:
        """Filtro para detectar patrones de decoloración fúngica específicos"""
        if img is None:
            return img
        
        # 1. Análisis multi-espectral de decoloraciones
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Detectar patrones de decoloración específicos
        # Decoloración por hongos: cambios en canales a* y b*
        discoloration_a = cv2.threshold(np.abs(a - np.mean(a)), 20, 255, cv2.THRESH_BINARY)[1]
        discoloration_b = cv2.threshold(np.abs(b - np.mean(b)), 25, 255, cv2.THRESH_BINARY)[1]
        
        # 3. Detectar patrones de difusión (hongos se extienden gradualmente)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        gradient = np.uint8(np.absolute(gradient))
        
        # 4. Crear máscara de difusión
        diffusion_mask = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)[1]
        
        # 5. Combinar detecciones
        discoloration_mask = cv2.bitwise_or(discoloration_a, discoloration_b)
        discoloration_mask = cv2.bitwise_or(discoloration_mask, diffusion_mask)
        
        # 6. Aplicar tinte amarillo-naranja para resaltar decoloraciones
        discoloration_img = img.copy()
        discoloration_img[:, :, 2] = cv2.multiply(discoloration_img[:, :, 2], 1.4)  # Más rojo
        discoloration_img[:, :, 1] = cv2.multiply(discoloration_img[:, :, 1], 1.2)  # Más verde
        discoloration_img[:, :, 0] = cv2.multiply(discoloration_img[:, :, 0], 0.8)  # Menos azul
        
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
        mold_texture_img[:, :, 0] = cv2.multiply(mold_texture_img[:, :, 0], 1.4)  # Más azul
        mold_texture_img[:, :, 2] = cv2.multiply(mold_texture_img[:, :, 2], 1.3)  # Más rojo
        mold_texture_img[:, :, 1] = cv2.multiply(mold_texture_img[:, :, 1], 0.7)  # Menos verde
        
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
        spore_img[:, :, 1] = cv2.multiply(spore_img[:, :, 1], 1.6)  # Más verde
        spore_img[:, :, 0] = cv2.multiply(spore_img[:, :, 0], 0.8)  # Menos azul
        spore_img[:, :, 2] = cv2.multiply(spore_img[:, :, 2], 0.4)  # Menos rojo
        
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
        contaminated_areas[:, :, 2] = cv2.multiply(contaminated_areas[:, :, 2], 1.5)  # Más rojo
        contaminated_areas[:, :, 1] = cv2.multiply(contaminated_areas[:, :, 1], 0.8)  # Menos verde
        contaminated_areas[:, :, 0] = cv2.multiply(contaminated_areas[:, :, 0], 0.7)  # Menos azul
        
        # 9. Combinar imagen final
        brazil_img = cv2.addWeighted(healthy_areas, 0.7, contaminated_areas, 0.3, 0)
        
        return brazil_img
    
    def apply_fungal_contamination_pipeline(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], str]:
        """
        Pipeline automatizado para detección de hongos y contaminación en castañas brasileñas
        
        Pipeline optimizado:
        1. brazil_chestnut → normalización inicial
        2. discoloration → detección de zonas sospechosas  
        3. mold_texture → confirmación de textura fúngica
        4. fungal + spore_detection → evidencia estructural
        5. mycotoxin + aflatoxin → detección de toxinas residuales
        
        Args:
            img: Imagen de entrada
            
        Returns:
            Tuple[imagen_resultado, scores_contaminacion, descripcion_detalles]
        """
        if img is None:
            return img, {}, "Error: imagen nula"
        
        print("[PIPELINE] Iniciando análisis de contaminación fúngica...")
        
        contamination_scores = {}
        pipeline_steps = []
        
        try:
            # PASO 1: Normalización inicial con brazil_chestnut
            print("[PIPELINE] Paso 1/6: Normalización inicial...")
            normalized_img, _, _ = self.apply_filter(img, "brazil_chestnut")
            pipeline_steps.append("Normalización inicial")
            
            # PASO 2: Detección de decoloración
            print("[PIPELINE] Paso 2/6: Detección de decoloración...")
            discoloration_img, _, _ = self.apply_filter(img, "discoloration")
            discoloration_score = self._calculate_contamination_score(discoloration_img, "discoloration")
            contamination_scores["decoloracion"] = discoloration_score
            pipeline_steps.append(f"Decoloración detectada: {discoloration_score:.2f}")
            
            # PASO 3: Análisis de textura de moho
            print("[PIPELINE] Paso 3/6: Análisis de textura de moho...")
            mold_texture_img, _, _ = self.apply_filter(img, "mold_texture")
            mold_texture_score = self._calculate_contamination_score(mold_texture_img, "mold_texture")
            contamination_scores["textura_moho"] = mold_texture_score
            pipeline_steps.append(f"Textura de moho: {mold_texture_score:.2f}")
            
            # PASO 4: Detección estructural de hongos
            print("[PIPELINE] Paso 4/6: Detección estructural de hongos...")
            fungal_img, _, _ = self.apply_filter(img, "fungal")
            spore_img, _, _ = self.apply_filter(img, "spore_detection")
            
            fungal_score = self._calculate_contamination_score(fungal_img, "fungal")
            spore_score = self._calculate_contamination_score(spore_img, "spore_detection")
            
            contamination_scores["hongos_estructurales"] = fungal_score
            contamination_scores["esporas"] = spore_score
            pipeline_steps.append(f"Hongos estructurales: {fungal_score:.2f}, Esporas: {spore_score:.2f}")
            
            # PASO 5: Detección de toxinas
            print("[PIPELINE] Paso 5/6: Detección de toxinas...")
            mycotoxin_img, _, _ = self.apply_filter(img, "mycotoxin")
            aflatoxin_img, _, _ = self.apply_filter(img, "aflatoxin")
            
            mycotoxin_score = self._calculate_contamination_score(mycotoxin_img, "mycotoxin")
            aflatoxin_score = self._calculate_contamination_score(aflatoxin_img, "aflatoxin")
            
            contamination_scores["micotoxinas"] = mycotoxin_score
            contamination_scores["aflatoxinas"] = aflatoxin_score
            pipeline_steps.append(f"Micotoxinas: {mycotoxin_score:.2f}, Aflatoxinas: {aflatoxin_score:.2f}")
            
            # PASO 6: Combinar resultados visualmente
            print("[PIPELINE] Paso 6/6: Combinando resultados...")
            final_img = self._combine_pipeline_results(
                normalized_img, discoloration_img, mold_texture_img, 
                fungal_img, spore_img, mycotoxin_img, aflatoxin_img
            )
            
            # Calcular score total de contaminación
            total_score = self._calculate_total_contamination_score(contamination_scores)
            contamination_scores["total"] = total_score
            
            # Generar descripción detallada
            description = self._generate_contamination_report(contamination_scores, pipeline_steps)
            
            print(f"[PIPELINE] Análisis completado. Score total: {total_score:.2f}")
            return final_img, contamination_scores, description
            
        except Exception as e:
            print(f"[ERROR] Error en pipeline de contaminación: {e}")
            return img, {"error": 1.0}, f"Error en pipeline: {str(e)}"
    
    def _calculate_contamination_score(self, img: np.ndarray, filter_type: str) -> float:
        """Calcular score de contaminación basado en análisis de imagen"""
        if img is None:
            return 0.0
        
        try:
            # Convertir a escala de grises para análisis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Análisis específico por tipo de filtro
            if filter_type in ["discoloration", "mycotoxin", "aflatoxin"]:
                # Detectar áreas con cambios de color significativos
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                return min(edge_density * 10, 1.0)  # Normalizar a 0-1
            
            elif filter_type in ["mold_texture", "fungal", "spore_detection"]:
                # Detectar texturas complejas (hongos)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                texture_variance = np.var(laplacian)
                return min(texture_variance / 10000, 1.0)  # Normalizar
            
            else:
                # Análisis general
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                contrast = np.std(hist)
                return min(contrast / 1000, 1.0)
                
        except Exception as e:
            print(f"[ERROR] Error calculando score para {filter_type}: {e}")
            return 0.0
    
    def _combine_pipeline_results(self, *images) -> np.ndarray:
        """Combinar resultados de múltiples filtros en una imagen final"""
        if not images or images[0] is None:
            return images[0] if images else None
        
        try:
            # Usar la primera imagen como base
            result = images[0].copy().astype(np.float32)
            
            # Combinar con pesos específicos para cada tipo de detección
            weights = [0.15, 0.15, 0.20, 0.15, 0.15, 0.10, 0.10]  # Suma = 1.0
            
            for i, img in enumerate(images[1:], 1):
                if img is not None:
                    img_float = img.astype(np.float32)
                    weight = weights[i] if i < len(weights) else 0.1
                    result = cv2.addWeighted(result, 1 - weight, img_float, weight, 0)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"[ERROR] Error combinando resultados: {e}")
            return images[0] if images else None
    
    def _calculate_total_contamination_score(self, scores: Dict[str, float]) -> float:
        """Calcular score total de contaminación con pesos específicos"""
        if not scores:
            return 0.0
        
        # Pesos específicos para cada tipo de contaminación
        weights = {
            "decoloracion": 0.15,
            "textura_moho": 0.25,
            "hongos_estructurales": 0.20,
            "esporas": 0.15,
            "micotoxinas": 0.15,
            "aflatoxinas": 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for key, score in scores.items():
            if key in weights and key != "total":
                weight = weights[key]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_contamination_report(self, scores: Dict[str, float], steps: list) -> str:
        """Generar reporte detallado de contaminación"""
        total = scores.get("total", 0.0)
        
        # Determinar nivel de contaminación
        if total < 0.2:
            level = "BAJA"
            color = "🟢"
        elif total < 0.5:
            level = "MEDIA"
            color = "🟡"
        else:
            level = "ALTA"
            color = "🔴"
        
        report = f"{color} CONTAMINACIÓN {level} (Score: {total:.2f})\n"
        report += "=" * 40 + "\n"
        
        # Detalles por tipo
        for key, score in scores.items():
            if key != "total" and key != "error":
                percentage = score * 100
                status = "✅ DETECTADO" if score > 0.3 else "⚪ NORMAL"
                report += f"• {key.replace('_', ' ').title()}: {percentage:.1f}% {status}\n"
        
        # Recomendaciones
        report += "\n📋 RECOMENDACIONES:\n"
        if total > 0.5:
            report += "🚫 DESCARTA esta castaña - Contaminación alta\n"
        elif total > 0.3:
            report += "⚠️ INSPECCIÓN MANUAL requerida\n"
        else:
            report += "✅ APROBADA - Contaminación baja\n"
        
        return report

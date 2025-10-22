"""
Cache de imágenes optimizado para reutilizar conversiones BGR→RGB
y redimensionamientos frecuentes con soporte para VRAM
"""

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageCache:
    """Cache optimizado para operaciones de imagen frecuentes con soporte VRAM"""

    def __init__(self, max_size: int = 100, use_vram: bool = True):
        self.max_size = max_size
        self.use_vram = use_vram
        self._rgb_cache = {}  # Cache para imágenes RGB
        self._resized_cache = {}  # Cache para imágenes redimensionadas
        self._stats_cache = {}  # Cache para estadísticas calculadas
        self._vram_cache = {}  # Cache en VRAM para GPU
        self._last_cleanup = time.time()
        self._cleanup_interval = 30  # Limpiar cada 30 segundos

    def get_rgb_image(self, bgr_img: np.ndarray, cache_key: str = None) -> np.ndarray:
        """
        Obtener imagen RGB desde cache o convertir BGR→RGB
        
        Args:
            bgr_img: Imagen en formato BGR
            cache_key: Clave única para el cache (opcional)
            
        Returns:
            Imagen en formato RGB
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(bgr_img)

        if cache_key in self._rgb_cache:
            return self._rgb_cache[cache_key]

        # Convertir BGR→RGB y cachear
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        self._cache_image(self._rgb_cache, cache_key, rgb_img)
        return rgb_img

    def get_resized_image(self, img: np.ndarray, size: tuple[int, int],
                         cache_key: str = None) -> np.ndarray:
        """
        Obtener imagen redimensionada desde cache o redimensionar
        
        Args:
            img: Imagen original
            size: Tamaño objetivo (width, height)
            cache_key: Clave única para el cache
            
        Returns:
            Imagen redimensionada
        """
        if cache_key is None:
            cache_key = f"{self._generate_cache_key(img)}_{size[0]}x{size[1]}"

        if cache_key in self._resized_cache:
            return self._resized_cache[cache_key]

        # Redimensionar y cachear
        resized_img = cv2.resize(img, size)
        self._cache_image(self._resized_cache, cache_key, resized_img)
        return resized_img

    def get_color_stats(self, rgb_img: np.ndarray, cache_key: str = None) -> dict[str, float]:
        """
        Obtener estadísticas de color desde cache o calcular
        
        Args:
            rgb_img: Imagen en formato RGB
            cache_key: Clave única para el cache
            
        Returns:
            Diccionario con estadísticas de color
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(rgb_img)

        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # Calcular estadísticas y cachear
        stats = self._calculate_color_stats(rgb_img)
        self._stats_cache[cache_key] = stats
        return stats

    def _generate_cache_key(self, img: np.ndarray) -> str:
        """Generar clave única para el cache basada en la imagen"""
        return f"{img.shape}_{img.dtype}_{hash(img.tobytes())}"

    def _cache_image(self, cache_dict: dict, key: str, img: np.ndarray):
        """Agregar imagen al cache con límite de tamaño"""
        if len(cache_dict) >= self.max_size:
            # Remover el elemento más antiguo (FIFO)
            oldest_key = next(iter(cache_dict))
            del cache_dict[oldest_key]

        cache_dict[key] = img.copy()

    def _calculate_color_stats(self, rgb_img: np.ndarray) -> dict[str, float]:
        """Calcular estadísticas de color de forma optimizada"""
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

    def clear_cache(self):
        """Limpiar todos los caches"""
        self._rgb_cache.clear()
        self._resized_cache.clear()
        self._stats_cache.clear()
        logger.info("Cache de imágenes limpiado")

    def get_cache_stats(self) -> dict[str, int]:
        """Obtener estadísticas del cache"""
        return {
            'rgb_cache_size': len(self._rgb_cache),
            'resized_cache_size': len(self._resized_cache),
            'stats_cache_size': len(self._stats_cache),
            'vram_cache_size': len(self._vram_cache),
            'total_cached': len(self._rgb_cache) + len(self._resized_cache) + len(self._stats_cache) + len(self._vram_cache)
        }

    def get_vram_tensor(self, image: np.ndarray, cache_key: str = None):
        """
        Obtener tensor en VRAM para procesamiento GPU
        
        Args:
            image: Imagen como numpy array
            cache_key: Clave única para el cache
            
        Returns:
            Tensor en VRAM (si está disponible)
        """
        if not self.use_vram:
            return None
            
        try:
            import torch
            
            if cache_key and cache_key in self._vram_cache:
                return self._vram_cache[cache_key]
            
            # Convertir a tensor y mover a GPU
            tensor = torch.from_numpy(image).float()
            if torch.cuda.is_available():
                tensor = tensor.cuda()
                
                # Guardar en cache VRAM
                if cache_key:
                    self._vram_cache[cache_key] = tensor
                    self._cleanup_vram_cache()
                
                return tensor
        except Exception as e:
            logger.debug(f"Error creando tensor VRAM: {e}")
            
        return None

    def _cleanup_vram_cache(self):
        """Limpiar cache VRAM si es necesario"""
        if len(self._vram_cache) > self.max_size:
            # Eliminar entradas más antiguas
            keys_to_remove = list(self._vram_cache.keys())[:len(self._vram_cache) - self.max_size]
            for key in keys_to_remove:
                del self._vram_cache[key]


# Instancia global del cache
image_cache = ImageCache(max_size=50)


def get_image_cache() -> ImageCache:
    """Obtener instancia global del cache de imágenes"""
    return image_cache


def clear_image_cache():
    """Limpiar cache de imágenes"""
    image_cache.clear_cache()

"""
Gestor de configuración principal del sistema CastañaSerial
Maneja todas las configuraciones del modelo YOLO, cámara, procesamiento y más
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Gestor centralizado de configuración del sistema"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._change_callbacks = []
    
    def _get_default_config_path(self) -> str:
        """Obtener ruta por defecto del archivo de configuración"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(current_dir, "config", "configuration.json")
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde archivo JSON"""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"Configuración principal cargada desde: {self.config_path}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error cargando configuración: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto si falla la carga"""
        return {
            "yolo_model": {
                "selected_model": "yolo12n.pt",
                "confidence_threshold": 0.75,
                "iou_threshold": 0.45,
                "image_size": 640,
                "save_predictions": False,
                "verbose": False
            },
            "camera_settings": {
                "selected_camera_id": 0,
                "resolution": "1280x720",
                "fps": 30,
                "auto_exposure": True,
                "brightness": 0,
                "contrast": 0,
                "saturation": 0
            },
            "processing_settings": {
                "render_every_n_frames": 1,
                "line_detection_radius_px": 50,
                "memory_cleanup_interval": 100,
                "performance_monitoring": True
            }
        }
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """Obtener configuración del modelo YOLO"""
        return self.config.get("yolo_model", {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Obtener configuración de cámara"""
        return self.config.get("camera_settings", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Obtener configuración de procesamiento"""
        return self.config.get("processing_settings", {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Obtener configuración de detección"""
        return self.config.get("detection_settings", {})
    

    
    def get_ui_config(self) -> Dict[str, Any]:
        """Obtener configuración de interfaz"""
        return self.config.get("ui_settings", {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Obtener configuración de análisis"""
        return self.config.get("analysis_settings", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Obtener configuración de rendimiento"""
        return self.config.get("performance_settings", {})
    
    def get_backup_config(self) -> Dict[str, Any]:
        """Obtener configuración de respaldos"""
        return self.config.get("backup_settings", {})
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Obtener configuración específica"""
        section_config = self.config.get(section, {})
        return section_config.get(key, default)
    
    def set_setting(self, section: str, key: str, value: Any) -> bool:
        """Establecer configuración específica"""
        try:
            if section not in self.config:
                self.config[section] = {}
            
            old_value = self.config[section].get(key)
            self.config[section][key] = value
            
            # Notificar cambio
            self._notify_change("setting_updated", {
                "section": section,
                "key": key,
                "old_value": old_value,
                "new_value": value
            })
            
            logger.info(f"Configuración actualizada: {section}.{key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")
            return False
    
    def update_section(self, section: str, values: Dict[str, Any]) -> bool:
        """Actualizar sección completa de configuración"""
        try:
            if section not in self.config:
                self.config[section] = {}
            
            old_values = self.config[section].copy()
            self.config[section].update(values)
            
            # Notificar cambio
            self._notify_change("section_updated", {
                "section": section,
                "old_values": old_values,
                "new_values": values
            })
            
            logger.info(f"Sección actualizada: {section}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando sección: {e}")
            return False
    
    def save_config(self, create_backup: bool = True) -> bool:
        """Guardar configuración al archivo JSON"""
        try:
            if create_backup:
                self._create_backup()
            
            # Actualizar timestamp
            self.config["last_updated"] = datetime.now().isoformat()
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Guardar con formato legible
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self._notify_change("config_saved", {"path": self.config_path})
            logger.info(f"Configuración guardada en: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            return False
    
    def reload_config(self) -> bool:
        """Recargar configuración desde archivo"""
        try:
            self.config = self._load_config()
            self._notify_change("config_reloaded", {"path": self.config_path})
            logger.info("Configuración recargada")
            return True
        except Exception as e:
            logger.error(f"Error recargando configuración: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Restablecer configuración a valores por defecto"""
        try:
            self.config = self._get_default_config()
            self._notify_change("config_reset", {"path": self.config_path})
            logger.info("Configuración restablecida a valores por defecto")
            return True
        except Exception as e:
            logger.error(f"Error restableciendo configuración: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Obtener resumen de la configuración actual"""
        return {
            "config_path": self.config_path,
            "last_updated": self.config.get("last_updated"),
            "version": self.config.get("version"),
            "sections": list(self.config.keys()),
            "yolo_model": self.get_yolo_config().get("selected_model"),
            "camera_id": self.get_camera_config().get("selected_camera_id"),
            "analysis_enabled": self.get_analysis_config().get("rgb_analysis_enabled")
        }
    
    def _create_backup(self) -> str:
        """Crear backup del archivo de configuración actual"""
        try:
            backup_dir = os.path.join(os.path.dirname(self.config_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"config_backup_{timestamp}.json")
            
            if os.path.exists(self.config_path):
                import shutil
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Backup creado: {backup_path}")
                return backup_path
            return ""
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            return ""
    
    def _notify_change(self, change_type: str, details: Dict[str, Any]):
        """Notificar cambios a los callbacks registrados"""
        for callback in self._change_callbacks:
            try:
                callback(change_type, details)
            except Exception as e:
                logger.warning(f"Error en callback de cambio: {e}")
    
    def add_change_callback(self, callback):
        """Agregar callback para notificar cambios"""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback):
        """Remover callback de notificaciones"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)


# Instancia global del gestor de configuración
config_manager = ConfigurationManager()


def get_config_manager() -> ConfigurationManager:
    """Obtener instancia global del gestor de configuración"""
    return config_manager


def get_yolo_config() -> Dict[str, Any]:
    """Obtener configuración del modelo YOLO"""
    return config_manager.get_yolo_config()


def get_camera_config() -> Dict[str, Any]:
    """Obtener configuración de cámara"""
    return config_manager.get_camera_config()


def get_processing_config() -> Dict[str, Any]:
    """Obtener configuración de procesamiento"""
    return config_manager.get_processing_config()


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """Obtener configuración específica"""
    return config_manager.get_setting(section, key, default)


def set_setting(section: str, key: str, value: Any) -> bool:
    """Establecer configuración específica"""
    return config_manager.set_setting(section, key, value)


def save_config() -> bool:
    """Guardar configuración"""
    return config_manager.save_config()


def reload_config() -> bool:
    """Recargar configuración"""
    return config_manager.reload_config()

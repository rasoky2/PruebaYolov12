"""
Módulo para sincronización automática entre código y configuración JSON
Detecta cambios en funciones de clasificación y los sincroniza con el archivo de configuración
"""

import ast
import inspect
import logging
from typing import Any

from .color_config import get_color_config, sync_config_from_code


logger = logging.getLogger(__name__)


class CodeSyncManager:
    """Gestor de sincronización automática entre código y configuración"""

    def __init__(self):
        self.config = get_color_config()
        self._last_sync_time = 0
        self._monitored_functions = [
            "classify_apple_by_color",
            "classify_orange_by_color",
            "classify_generic_object_by_color",
        ]

    def extract_color_ranges_from_function(self, func) -> dict[str, Any]:
        """
        Extraer rangos de color de una función de clasificación
        
        Args:
            func: Función a analizar
            
        Returns:
            dict: Rangos de color extraídos
        """
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            ranges = {"sana": [], "contaminada": []}
            priority_counter = 1
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    condition = self._extract_condition(node.test)
                    if condition:
                        # Determinar si es condición sana o contaminada
                        quality_type = self._determine_quality_type(condition, node)
                        if quality_type:
                            range_config = self._create_range_config(condition, priority_counter)
                            ranges[quality_type].append(range_config)
                            priority_counter += 1
            
            return ranges
            
        except Exception as e:
            logger.error(f"Error extrayendo rangos de función {func.__name__}: {e}")
            return {"sana": [], "contaminada": []}

    def _extract_condition(self, test_node: ast.AST) -> str | None:
        """Extraer condición de un nodo de prueba"""
        try:
            if isinstance(test_node, ast.Compare):
                return self._format_compare_condition(test_node)
            if isinstance(test_node, ast.BoolOp):
                return self._format_boolop_condition(test_node)
            return None
        except Exception:
            return None

    def _format_compare_condition(self, node: ast.Compare) -> str:
        """Formatear condición de comparación"""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return ""
        
        left = self._format_ast_node(node.left)
        op = self._format_operator(node.ops[0])
        right = self._format_ast_node(node.comparators[0])
        
        return f"{left} {op} {right}"

    def _format_boolop_condition(self, node: ast.BoolOp) -> str:
        """Formatear condición booleana"""
        op = "and" if isinstance(node.op, ast.And) else "or"
        conditions = []
        
        for value in node.values:
            condition = self._extract_condition(value)
            if condition:
                conditions.append(condition)
        
        return f" {op} ".join(conditions)

    def _format_ast_node(self, node: ast.AST) -> str:
        """Formatear nodo AST a string"""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.BinOp):
            left = self._format_ast_node(node.left)
            op = self._format_operator(node.op)
            right = self._format_ast_node(node.right)
            return f"{left} {op} {right}"
        return ""

    def _format_operator(self, op: ast.AST) -> str:
        """Formatear operador AST"""
        op_map = {
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }
        return op_map.get(type(op), str(op))

    def _determine_quality_type(self, condition: str, node: ast.If) -> str | None:
        """Determinar si la condición es para fruta sana o contaminada"""
        # Analizar el cuerpo del if para determinar el tipo
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                return_value = stmt.value.value
                if return_value == "sana":
                    return "sana"
                if return_value == "contaminada":
                    return "contaminada"
        return None

    def _create_range_config(self, condition: str, priority: int) -> dict[str, Any]:
        """Crear configuración de rango desde condición"""
        # Simplificar condición a rangos RGB básicos
        range_config = {
            "name": f"condicion_automatica_{priority}",
            "condition": condition,
            "priority": priority,
        }
        
        # Intentar extraer rangos RGB si es posible
        rgb_ranges = self._extract_rgb_ranges(condition)
        range_config.update(rgb_ranges)
        
        return range_config

    def _extract_rgb_ranges(self, condition: str) -> dict[str, Any]:
        """Extraer rangos RGB de una condición"""
        ranges = {}
        
        # Buscar patrones como "r < 100", "g > 140", etc.
        import re
        
        # Patrón para r_min
        r_min_match = re.search(r"r\s*>\s*(\d+)", condition)
        if r_min_match:
            ranges["r_min"] = int(r_min_match.group(1))
        
        # Patrón para r_max
        r_max_match = re.search(r"r\s*<\s*(\d+)", condition)
        if r_max_match:
            ranges["r_max"] = int(r_max_match.group(1))
        
        # Patrón para g_min
        g_min_match = re.search(r"g\s*>\s*(\d+)", condition)
        if g_min_match:
            ranges["g_min"] = int(g_min_match.group(1))
        
        # Patrón para g_max
        g_max_match = re.search(r"g\s*<\s*(\d+)", condition)
        if g_max_match:
            ranges["g_max"] = int(g_max_match.group(1))
        
        # Patrón para b_min
        b_min_match = re.search(r"b\s*>\s*(\d+)", condition)
        if b_min_match:
            ranges["b_min"] = int(b_min_match.group(1))
        
        # Patrón para b_max
        b_max_match = re.search(r"b\s*<\s*(\d+)", condition)
        if b_max_match:
            ranges["b_max"] = int(b_max_match.group(1))
        
        return ranges

    def sync_functions_to_config(self) -> bool:
        """
        Sincronizar todas las funciones monitoreadas con la configuración
        
        Returns:
            bool: True si se sincronizó exitosamente
        """
        try:
            from . import analysys
            
            all_ranges = {}
            
            for func_name in self._monitored_functions:
                if hasattr(analysys, func_name):
                    func = getattr(analysys, func_name)
                    ranges = self.extract_color_ranges_from_function(func)
                    
                    if ranges["sana"] or ranges["contaminada"]:
                        # Mapear nombre de función a tipo de objeto
                        object_class = self._map_function_to_object_class(func_name)
                        all_ranges[object_class] = ranges
            
            if all_ranges:
                success = sync_config_from_code(all_ranges)
                if success:
                    logger.info("Funciones sincronizadas con configuración")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error sincronizando funciones: {e}")
            return False

    def _map_function_to_object_class(self, func_name: str) -> str:
        """Mapear nombre de función a clase de objeto"""
        mapping = {
            "classify_apple_by_color": "frutas",
            "classify_orange_by_color": "frutas",
            "classify_generic_object_by_color": "generic",
        }
        return mapping.get(func_name, "generic")

    def auto_sync_on_change(self, file_path: str) -> bool:
        """
        Sincronización automática cuando se detecta cambio en archivo
        
        Args:
            file_path: Ruta del archivo modificado
            
        Returns:
            bool: True si se sincronizó
        """
        try:
            # Verificar si el archivo modificado contiene funciones de clasificación
            if "analysys.py" in file_path or "color_config.py" in file_path:
                import time
                current_time = time.time()
                
                # Evitar sincronizaciones muy frecuentes (mínimo 5 segundos)
                if current_time - self._last_sync_time > 5:
                    success = self.sync_functions_to_config()
                    if success:
                        self._last_sync_time = current_time
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error en sincronización automática: {e}")
            return False

    def get_sync_status(self) -> dict[str, Any]:
        """Obtener estado de sincronización"""
        return {
            "last_sync_time": self._last_sync_time,
            "monitored_functions": self._monitored_functions,
            "config_summary": self.config.get_config_summary(),
        }


# Instancia global del gestor de sincronización
sync_manager = CodeSyncManager()


def sync_functions_to_config() -> bool:
    """Sincronizar funciones con configuración"""
    return sync_manager.sync_functions_to_config()


def auto_sync_on_change(file_path: str) -> bool:
    """Sincronización automática en cambios de archivo"""
    return sync_manager.auto_sync_on_change(file_path)


def get_sync_status() -> dict[str, Any]:
    """Obtener estado de sincronización"""
    return sync_manager.get_sync_status()

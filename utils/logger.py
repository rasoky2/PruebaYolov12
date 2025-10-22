"""
Sistema de logging con colores para CastañaSerial
Usa colorama para colores simples y efectivos en terminal
"""

from datetime import datetime

from colorama import Fore, Style, init

# Inicializar colorama para Windows
init(autoreset=True)


class CastanaLogger:
    """Logger simple con colores para el proyecto CastañaSerial"""

    def __init__(self):
        self.enabled = True
        self.show_timestamp = True
        self.show_component = True

    def _format_message(self, message: str, level: str, component: str = None) -> str:
        """Formatear mensaje con timestamp y componente"""
        parts = []

        if self.show_timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")
            parts.append(f"{Fore.CYAN}[{timestamp}]{Style.RESET_ALL}")

        if level:
            parts.append(level)

        if self.show_component and component:
            parts.append(f"{Fore.BLUE}[{component.upper()}]{Style.RESET_ALL}")

        parts.append(message)
        return " ".join(parts)

    def info(self, message: str, component: str = None):
        """Mensaje informativo (blanco)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.WHITE}[INFO]{Style.RESET_ALL}", component)
            print(formatted)

    def success(self, message: str, component: str = None):
        """Mensaje de éxito (verde)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.GREEN}[OK]{Style.RESET_ALL}", component)
            print(formatted)

    def warning(self, message: str, component: str = None):
        """Mensaje de advertencia (amarillo)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}", component)
            print(formatted)

    def error(self, message: str, component: str = None):
        """Mensaje de error (rojo)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.RED}[ERROR]{Style.RESET_ALL}", component)
            print(formatted)

    def debug(self, message: str, component: str = None):
        """Mensaje de debug (magenta)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL}", component)
            print(formatted)

    def critical(self, message: str, component: str = None):
        """Mensaje crítico (rojo brillante)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.RED}{Style.BRIGHT}[CRITICAL]{Style.RESET_ALL}", component)
            print(formatted)

    def chestnut(self, message: str, component: str = None):
        """Mensaje específico para castañas (naranja)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.LIGHTRED_EX}[CHESTNUT]{Style.RESET_ALL}", component)
            print(formatted)

    def arduino(self, message: str):
        """Mensaje específico para Arduino (cian)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.CYAN}[ARDUINO]{Style.RESET_ALL}")
            print(formatted)

    def filter_info(self, message: str):
        """Mensaje específico para filtros (verde claro)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.LIGHTGREEN_EX}[FILTER]{Style.RESET_ALL}")
            print(formatted)

    def camera(self, message: str):
        """Mensaje específico para cámaras (azul claro)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.LIGHTBLUE_EX}[CAMERA]{Style.RESET_ALL}")
            print(formatted)

    def performance(self, message: str):
        """Mensaje específico para rendimiento (amarillo brillante)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.LIGHTYELLOW_EX}[PERFORMANCE]{Style.RESET_ALL}")
            print(formatted)

    def yolo(self, message: str):
        """Mensaje específico para YOLO (magenta brillante)"""
        if self.enabled:
            formatted = self._format_message(message, f"{Fore.LIGHTMAGENTA_EX}[YOLO]{Style.RESET_ALL}")
            print(formatted)


# Instancia global del logger
castana_logger = CastanaLogger()

# Exportar métodos convenientes
info = castana_logger.info
success = castana_logger.success
warning = castana_logger.warning
error = castana_logger.error
debug = castana_logger.debug
critical = castana_logger.critical
chestnut = castana_logger.chestnut
arduino = castana_logger.arduino
filter_info = castana_logger.filter_info
camera = castana_logger.camera
performance = castana_logger.performance
yolo = castana_logger.yolo


# Funciones helper para componentes específicos
def arduino_info(message: str):
    """Información de Arduino"""
    arduino(f"{message}")


def arduino_ok(message: str):
    """Éxito de Arduino"""
    arduino(f"{message}")


def arduino_error(message: str):
    """Error de Arduino"""
    arduino(f"{message}")


def arduino_warning(message: str):
    """Advertencia de Arduino"""
    arduino(f"{message}")


def detection_info(message: str):
    """Información de detección"""
    chestnut(f"{message}")


def detection_success(message: str):
    """Éxito de detección"""
    chestnut(f"{message}")


def detection_error(message: str):
    """Error de detección"""
    chestnut(f"{message}")


def camera_log(message: str):
    """Información de cámara"""
    camera(f"{message}")


def camera_ok(message: str):
    """Éxito de cámara"""
    camera(f"{message}")


def camera_error(message: str):
    """Error de cámara"""
    camera(f"{message}")


def filter_info_detailed(message: str):
    """Información detallada de filtros"""
    filter_info(f"{message}")


def performance_info(message: str):
    """Información de rendimiento"""
    performance(f"{message}")


def yolo_info(message: str):
    """Información de YOLO"""
    yolo(f"{message}")


# Función para mostrar separadores con colores
def separator(char: str = "=", length: int = 70, color: str = "CYAN"):
    """Mostrar separador con color"""
    color_code = getattr(Fore, color.upper(), Fore.CYAN)
    separator_line = char * length
    print(f"{color_code}{separator_line}{Style.RESET_ALL}")


# Función para mostrar títulos con colores
def title(text: str, char: str = "=", color: str = "GREEN"):
    """Mostrar título con colores"""
    color_code = getattr(Fore, color.upper(), Fore.GREEN)
    separator(char, len(text) + 4, color)
    print(f"{color_code}  {text}  {Style.RESET_ALL}")
    separator(char, len(text) + 4, color)

"""
Gestor de conexión y comunicación con Arduino
Maneja detección automática de puertos, conexión y envío de señales
"""

import time
from typing import Optional, List
import serial.tools.list_ports

# Importar PySerial para comunicación con Arduino
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARNING] Librería 'serial' no disponible. Instala con: pip install pyserial")


class ArduinoManager:
    """Gestor centralizado de comunicación con Arduino"""
    
    def __init__(self):
        self.connection = None
        self.enabled = False
        self.last_signal = None
        self.signal_time = 0
        self.frames_sin_deteccion = 0
        self.port = None
        self.baudrate = 9600
    
    def is_serial_available(self) -> bool:
        """Verificar si PySerial está disponible"""
        return SERIAL_AVAILABLE
    
    def list_available_ports(self) -> List:
        """Listar puertos seriales disponibles"""
        if not SERIAL_AVAILABLE:
            print("[ERROR] Librería 'serial' no disponible")
            return []
        
        print("[SEARCH] Puertos seriales disponibles:")
        try:
            ports = serial.tools.list_ports.comports()
            if ports:
                for i, port in enumerate(ports):
                    # Detectar si es Arduino
                    is_arduino = "Arduino" in port.description or "USB" in port.description
                    arduino_marker = " (ARDUINO RECOMENDADO)" if is_arduino else ""
                    print(f"   {i+1}. {port.device} - {port.description}{arduino_marker}")
            else:
                print("   No se encontraron puertos seriales")
            return ports
        except Exception as e:
            print(f"[ERROR] Error listando puertos: {e}")
            return []
    
    def auto_detect_port(self) -> Optional[str]:
        """Detectar automáticamente el puerto Arduino"""
        if not SERIAL_AVAILABLE:
            return None
        
        try:
            ports = serial.tools.list_ports.comports()
            
            # Buscar puerto con "Arduino" en la descripción
            for port in ports:
                if "Arduino" in port.description:
                    return port.device
            
            # Si no encuentra Arduino específico, buscar USB
            for port in ports:
                if "USB" in port.description:
                    return port.device
                    
            return None
        except Exception as e:
            print(f"[ERROR] Error detectando puerto Arduino: {e}")
            return None
    
    def connect(self, port: str = None, baudrate: int = 9600) -> bool:
        """Conectar con Arduino"""
        if not SERIAL_AVAILABLE:
            print("[ERROR] Librería 'serial' no disponible. Instala con: pip install pyserial")
            self.enabled = False
            return False
        
        # Si no se especifica puerto, intentar detección automática
        if port is None:
            port = self.auto_detect_port()
            if port is None:
                print("[ERROR] No se pudo detectar puerto Arduino automáticamente")
                self.enabled = False
                return False
        
        self.port = port
        self.baudrate = baudrate
        
        try:
            self.connection = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Esperar conexión estable
            
            # Verificar que Arduino responda
            self.connection.write(b"TEST\n")
            self.connection.flush()
            time.sleep(0.5)
            
            print(f"[OK] Arduino conectado en puerto {port}")
            self.enabled = True
            return True
        except serial.SerialException as e:
            print(f"[ERROR] Error conectando Arduino: {e}")
            print(f"[TIP] Asegúrate de que el Arduino esté conectado y el puerto {port} esté disponible")
            self.enabled = False
            return False
        except Exception as e:
            print(f"[ERROR] Error inesperado: {e}")
            self.enabled = False
            return False
    
    def disconnect(self) -> None:
        """Desconectar Arduino"""
        if self.connection and self.enabled:
            try:
                self.connection.close()
                print("[ARDUINO] Conexión Arduino cerrada")
            except:
                pass
        
        self.enabled = False
        self.connection = None
        self.last_signal = None
        self.signal_time = 0
        self.frames_sin_deteccion = 0
    
    def send_signal(self, signal_type: str, delay_prevention: float = 1.0) -> bool:
        """Enviar señal al Arduino con prevención de spam"""
        if not self.enabled or self.connection is None:
            return False
        
        current_time = time.time()
        
        # Prevenir envío de señales duplicadas muy rápido
        if (self.last_signal == signal_type and 
            current_time - self.signal_time < delay_prevention):
            return False
        
        try:
            # Enviar comando al Arduino
            command = f"{signal_type}\n"
            self.connection.write(command.encode())
            self.connection.flush()
            
            # Actualizar estado
            self.last_signal = signal_type
            self.signal_time = current_time
            
            print(f"[ARDUINO] Señal enviada al Arduino: {signal_type}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error enviando señal al Arduino: {e}")
            return False
    
    def send_detection_signals(self, total_castanas: int, contaminadas: int) -> None:
        """Enviar señales basadas en el estado de detección"""
        if not self.enabled:
            return
        
        if contaminadas > 0:
            # Si hay al menos una castaña contaminada, enviar señal CONTAMINADO
            self.send_signal('CONTAMINADO', delay_prevention=0.3)  # Rápido para contaminados
            self.frames_sin_deteccion = 0  # Resetear contador
        elif total_castanas > 0:
            # Si hay castañas pero todas son sanas, enviar señal SANO
            self.send_signal('SANO', delay_prevention=0.1)  # Súper rápido para sanas
            self.frames_sin_deteccion = 0  # Resetear contador
        else:
            # Si no hay castañas detectadas, incrementar contador
            self.frames_sin_deteccion += 1
            
            # Si han pasado pocos frames sin detección, enviar SANO súper rápido
            if self.frames_sin_deteccion >= 5:  # 5 frames = ~0.17 segundos a 30fps
                self.send_signal('SANO', delay_prevention=0.05)  # Súper rápido
                self.frames_sin_deteccion = 0  # Resetear contador
    
    def get_status_info(self) -> dict:
        """Obtener información del estado actual de Arduino"""
        status_info = {
            'enabled': self.enabled,
            'connected': self.enabled and self.connection is not None,
            'port': self.port,
            'last_signal': self.last_signal,
            'frames_sin_deteccion': self.frames_sin_deteccion
        }
        
        if self.enabled and self.last_signal:
            # Determinar color de señal
            if self.last_signal == "SANO":
                status_info['signal_color'] = (0, 255, 0)  # Verde
                status_info['behavior_text'] = "[OK] Enviará CONTAMINADO si detecta contaminación"
            else:  # CONTAMINADO
                status_info['signal_color'] = (0, 0, 255)  # Rojo
                status_info['behavior_text'] = "🔄 Cambiará a SANO si no detecta contaminación"
        else:
            status_info['signal_color'] = (255, 255, 0)  # Amarillo
            status_info['behavior_text'] = "🔄 Cambiará a SANO rápidamente si no detecta"
        
        return status_info
    
    def interactive_setup(self) -> bool:
        """Configuración interactiva de Arduino"""
        if not SERIAL_AVAILABLE:
            print("[ERROR] Librería 'serial' no disponible. Instala con: pip install pyserial")
            return False
        
        print("\n[ARDUINO] Inicializando conexión Arduino...")
        print("[TIP] Para conectar Arduino:")
        print("   1. Conecta el Arduino por USB")
        print("   2. Verifica el puerto COM (Windows) o /dev/ttyUSB* (Linux)")
        print("   3. Asegúrate de que el Arduino esté ejecutando el código de recepción")
        
        # Listar puertos disponibles
        ports = self.list_available_ports()
        
        # Detectar puerto Arduino automáticamente
        arduino_port = self.auto_detect_port()
        
        selected_port = None
        
        if arduino_port:
            print(f"\n[OK] Arduino detectado automáticamente: {arduino_port}")
            use_auto = input("¿Usar este puerto? (s/n): ").strip().lower()
            if use_auto in ['s', 'si', 'sí', 'y', 'yes', '']:
                selected_port = arduino_port
            else:
                selected_port = None
        else:
            selected_port = None
        
        if not selected_port and ports:
            # Pedir puerto manualmente
            print("\n[TIP] Opciones:")
            print("   1. Ingresa número de la lista (ej: 3)")
            print("   2. Ingresa puerto completo (ej: COM6)")
            print("   3. Presiona Enter para omitir Arduino")
            
            user_input = input("\nSelección: ").strip()
            
            if user_input.isdigit():
                # Usuario ingresó número
                index = int(user_input) - 1
                if 0 <= index < len(ports):
                    selected_port = ports[index].device
                    print(f"[OK] Seleccionado: {selected_port}")
                else:
                    print(f"[ERROR] Número inválido. Omitiendo Arduino")
                    selected_port = None
            elif user_input:
                # Usuario ingresó texto
                if user_input.startswith('COM') or user_input.startswith('/dev/'):
                    selected_port = user_input
                else:
                    print(f"[ERROR] Formato inválido. Omitiendo Arduino")
                    selected_port = None
            else:
                # Usuario presionó Enter
                selected_port = None
                print("[INFO] Modo solo detección - sin control Arduino")
        
        # Intentar conectar Arduino
        if selected_port:
            print(f"\n[ARDUINO] Conectando con puerto: {selected_port}")
            if self.connect(selected_port):
                print("🎉 Arduino listo para recibir señales de detección!")
                return True
            else:
                print("[WARNING] Continuando sin Arduino - solo detección visual")
                return False
        else:
            print("[INFO] Modo solo detección - sin control Arduino")
            return False

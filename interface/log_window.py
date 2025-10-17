"""
Ventana separada para mostrar logs del sistema
Mejora el rendimiento al no escribir constantemente a la terminal
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
from collections import deque
import time


class LogWindow:
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.root = None  # Se establecerá con set_root_reference
        self.log_window = None
        self.log_text = None
        self.log_buffer = deque(maxlen=1000)  # Buffer circular de 1000 líneas
        self.log_lock = threading.Lock()
        self.auto_scroll = True
        
        # Crear botón para abrir logs
        self.create_log_button()
    
    def create_log_button(self):
        """Crear botón para abrir ventana de logs"""
        self.log_button = ttk.Button(
            self.parent, 
            text="📋 Abrir Logs", 
            command=self.toggle_log_window,
            width=12
        )
        self.log_button.pack(side=tk.RIGHT, padx=(5, 0))
    
    def toggle_log_window(self):
        """Abrir o cerrar ventana de logs"""
        if self.log_window is None or not self.log_window.winfo_exists():
            self.open_log_window()
        else:
            self.close_log_window()
    
    def open_log_window(self):
        """Abrir ventana de logs"""
        self.log_window = tk.Toplevel(self.parent)
        self.log_window.title("Logs del Sistema - CastañaSerial")
        self.log_window.geometry("800x500")
        self.log_window.configure(bg='#2C3E50')
        
        # Configurar cierre de ventana
        self.log_window.protocol("WM_DELETE_WINDOW", self.close_log_window)
        
        # Frame principal
        main_frame = ttk.Frame(self.log_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Botones de control
        ttk.Button(controls_frame, text="🗑️ Limpiar", command=self.clear_logs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="💾 Guardar", command=self.save_logs).pack(side=tk.LEFT, padx=(0, 5))
        
        # Checkbox auto-scroll
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Contador de líneas
        self.line_count_label = ttk.Label(controls_frame, text="Líneas: 0")
        self.line_count_label.pack(side=tk.RIGHT)
        
        # Área de texto para logs
        self.log_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#1E1E1E',
            fg='#FFFFFF',
            insertbackground='white',
            selectbackground='#404040',
            height=25
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configurar colores para diferentes tipos de logs
        self.setup_text_tags()
        
        # Cargar logs existentes del buffer
        self.load_existing_logs()
        
        # Cambiar texto del botón
        self.log_button.config(text="📋 Cerrar Logs")
        
        print("📋 Ventana de logs abierta")
    
    def setup_text_tags(self):
        """Configurar etiquetas de color para diferentes tipos de logs"""
        self.log_text.tag_configure("INFO", foreground="#00FF00")      # Verde
        self.log_text.tag_configure("WARNING", foreground="#FFA500")   # Naranja
        self.log_text.tag_configure("ERROR", foreground="#FF0000")     # Rojo
        self.log_text.tag_configure("DEBUG", foreground="#00FFFF")     # Cian
        self.log_text.tag_configure("YOLO", foreground="#FFFF00")      # Amarillo
        self.log_text.tag_configure("ARDUINO", foreground="#FF69B4")   # Rosa
        self.log_text.tag_configure("CAMERA", foreground="#87CEEB")    # Azul claro
        self.log_text.tag_configure("DEFAULT", foreground="#FFFFFF")   # Blanco
    
    def close_log_window(self):
        """Cerrar ventana de logs"""
        if self.log_window:
            self.log_window.destroy()
            self.log_window = None
            self.log_text = None
            self.log_button.config(text="📋 Abrir Logs")
            print("📋 Ventana de logs cerrada")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Agregar mensaje al log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Agregar al buffer
        with self.log_lock:
            self.log_buffer.append((log_entry, level))
        
        # Actualizar ventana si está abierta
        if self.log_window and self.log_text and hasattr(self, 'root'):
            self.root.after(0, self._update_log_display, log_entry, level)
    
    def _update_log_display(self, log_entry: str, level: str) -> None:
        """Actualizar display de logs en el hilo principal"""
        if not self.log_text:
            return
        
        # Determinar tag de color
        tag = LogWindow.get_log_tag(level)
        
        # Insertar texto
        self.log_text.insert(tk.END, log_entry + "\n", tag)
        
        # Auto-scroll si está habilitado
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
        
        # Actualizar contador
        line_count = int(self.log_text.index('end-1c').split('.')[0])
        self.line_count_label.config(text=f"Líneas: {line_count}")
        
        # Limitar líneas si es necesario
        if line_count > 2000:
            self.log_text.delete("1.0", "1000.0")
    
    @staticmethod
    def get_log_tag(level: str) -> str:
        """Obtener tag de color basado en el nivel"""
        level_upper = level.upper()
        if "ERROR" in level_upper or "❌" in level:
            return "ERROR"
        elif "WARNING" in level_upper or "⚠️" in level:
            return "WARNING"
        elif "YOLO" in level_upper or "0:" in level:
            return "YOLO"
        elif "ARDUINO" in level_upper or "CONTAMINADO" in level_upper or "SANO" in level_upper:
            return "ARDUINO"
        elif "CAMERA" in level_upper or "🔄" in level:
            return "CAMERA"
        elif "DEBUG" in level_upper or "🔍" in level:
            return "DEBUG"
        elif "INFO" in level_upper or "✅" in level:
            return "INFO"
        else:
            return "DEFAULT"
    
    def load_existing_logs(self):
        """Cargar logs existentes del buffer"""
        if not self.log_text:
            return
        
        with self.log_lock:
            for log_entry, level in self.log_buffer:
                tag = LogWindow.get_log_tag(level)
                self.log_text.insert(tk.END, log_entry + "\n", tag)
        
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
    
    def clear_logs(self):
        """Limpiar todos los logs"""
        if self.log_text:
            self.log_text.delete("1.0", tk.END)
        
        with self.log_lock:
            self.log_buffer.clear()
        
        self.line_count_label.config(text="Líneas: 0")
        self.log("Logs limpiados", "INFO")
    
    def save_logs(self):
        """Guardar logs a archivo"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Guardar logs"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    with self.log_lock:
                        for log_entry, _ in self.log_buffer:
                            f.write(log_entry + "\n")
                
                self.log(f"Logs guardados en: {filename}", "INFO")
        except Exception as e:
            self.log(f"Error guardando logs: {e}", "ERROR")
    
    def set_root_reference(self, root):
        """Establecer referencia al root para after()"""
        self.root = root


# Instancia global para capturar prints
_log_window_instance = None


def setup_global_logging(log_window):
    """Configurar logging global para capturar prints"""
    global _log_window_instance
    _log_window_instance = log_window


def log_print(*args, **kwargs):
    """Función de reemplazo para print() que también envía a la ventana de logs"""
    # Llamar al print original
    print(*args, **kwargs)
    
    # Enviar también a la ventana de logs si está disponible
    if _log_window_instance and _log_window_instance.log_window:
        message = " ".join(str(arg) for arg in args)
        level = "INFO"
        
        # Determinar nivel basado en el contenido
        if "❌" in message or "ERROR" in message.upper():
            level = "ERROR"
        elif "⚠️" in message or "WARNING" in message.upper():
            level = "WARNING"
        elif "🔍" in message or "DEBUG" in message.upper():
            level = "DEBUG"
        elif "YOLO" in message.upper() or "0:" in message:
            level = "YOLO"
        elif "ARDUINO" in message.upper() or "CONTAMINADO" in message.upper() or "SANO" in message.upper():
            level = "ARDUINO"
        elif "🔄" in message or "CAMERA" in message.upper():
            level = "CAMERA"
        
        _log_window_instance.log(message, level)

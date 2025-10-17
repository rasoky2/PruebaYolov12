"""
Widget para controles de modelo, performance y línea de activación
"""

import tkinter as tk
from tkinter import ttk, messagebox


class ControlsPanel:
    def __init__(self, parent):
        self.parent = parent
        
        # Variables
        self.model_var = tk.StringVar(value="YOLO12x")
        self.render_every_var = tk.IntVar(value=1)
        self.arduino_debounce_var = tk.DoubleVar(value=2.0)
        self.arduino_cooldown_var = tk.DoubleVar(value=0.3)
        self.line_detection_radius_var = tk.IntVar(value=50)
        
        # Variables de línea
        self.line_draw_mode = False
        self.line_start_norm = None
        self.line_end_norm = None
        self.temp_line_end_norm = None  # Para mostrar línea temporal mientras arrastra
        self.is_drawing_line = False    # True cuando está arrastrando
        
        # UI Elements
        self.model_combo = None
        self.render_every_spin = None
        self.arduino_debounce_spin = None
        self.arduino_cooldown_spin = None
        
        self.create_ui()
    
    def create_ui(self):
        """Crear interfaz del panel de controles"""
        # Modelo YOLO
        model_frame = ttk.LabelFrame(self.parent, text="Modelo YOLO", padding=5)
        model_frame.pack(fill=tk.X, pady=5)
        
        rowm = ttk.Frame(model_frame)
        rowm.pack(fill=tk.X, pady=2)
        ttk.Label(rowm, text="Modelo:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(rowm, textvariable=self.model_var, state="readonly", width=15)
        self.model_combo['values'] = ['YOLO12n', 'YOLO12l', 'YOLO12x']
        self.model_combo.pack(side=tk.RIGHT)
        
        # Render/Performance
        perf_frame = ttk.LabelFrame(self.parent, text="Performance UI", padding=5)
        perf_frame.pack(fill=tk.X, pady=5)
        
        # Render cada N frames
        rowp = ttk.Frame(perf_frame)
        rowp.pack(fill=tk.X, pady=2)
        ttk.Label(rowp, text="Render cada N frames:").pack(side=tk.LEFT)
        self.render_every_spin = ttk.Spinbox(rowp, from_=1, to=10, textvariable=self.render_every_var, width=5, state='readonly')
        self.render_every_spin.pack(side=tk.RIGHT)

        # Debounce Arduino
        rowd = ttk.Frame(perf_frame)
        rowd.pack(fill=tk.X, pady=2)
        ttk.Label(rowd, text="Estabilidad Arduino (s):").pack(side=tk.LEFT)
        self.arduino_debounce_spin = ttk.Spinbox(rowd, from_=0.0, to=10.0, increment=0.5, textvariable=self.arduino_debounce_var, width=5, state='readonly')
        self.arduino_debounce_spin.pack(side=tk.RIGHT)

        # Cooldown Arduino
        rowc = ttk.Frame(perf_frame)
        rowc.pack(fill=tk.X, pady=2)
        ttk.Label(rowc, text="Cooldown Arduino (s):").pack(side=tk.LEFT)
        self.arduino_cooldown_spin = ttk.Spinbox(rowc, from_=0.0, to=5.0, increment=0.1, textvariable=self.arduino_cooldown_var, width=5, state='readonly')
        self.arduino_cooldown_spin.pack(side=tk.RIGHT)
        
        # Radio de detección de línea
        rowr = ttk.Frame(perf_frame)
        rowr.pack(fill=tk.X, pady=2)
        ttk.Label(rowr, text="Radio detección línea (px):").pack(side=tk.LEFT)
        self.line_detection_radius_spin = ttk.Spinbox(rowr, from_=10, to=200, increment=5, textvariable=self.line_detection_radius_var, width=5, state='readonly')
        self.line_detection_radius_spin.pack(side=tk.RIGHT)

        # Línea de activación
        line_frame = ttk.LabelFrame(self.parent, text="Línea de activación", padding=5)
        line_frame.pack(fill=tk.X, pady=5)
        ttk.Button(line_frame, text="Dibujar línea", command=self.enable_line_draw).pack(fill=tk.X, pady=2)
        ttk.Button(line_frame, text="Borrar línea", command=self.clear_activation_line).pack(fill=tk.X, pady=2)
    
    def enable_line_draw(self):
        """Habilitar modo de dibujo de línea"""
        self.line_draw_mode = True
        messagebox.showinfo("Línea", "Haz clic y arrastra para dibujar la línea de activación.\nPrimer clic: inicio | Segundo clic: fin")
    
    def clear_activation_line(self):
        """Borrar línea de activación"""
        self.line_start_norm = None
        self.line_end_norm = None
        self.temp_line_end_norm = None
        self.line_draw_mode = False
        self.is_drawing_line = False
    
    def on_video_click(self, event):
        """Capturar clic inicial para comenzar a dibujar línea"""
        if not self.line_draw_mode:
            return
        try:
            # Calcular normalización respecto al tamaño actual del label
            label_width = event.widget.winfo_width()
            label_height = event.widget.winfo_height()
            if label_width <= 1 or label_height <= 1:
                return
            
            nx = max(0.0, min(1.0, event.x / float(label_width)))
            ny = max(0.0, min(1.0, event.y / float(label_height)))
            
            # Primer clic: establecer punto inicial
            if not self.is_drawing_line:
                self.line_start_norm = (nx, ny)
                self.temp_line_end_norm = (nx, ny)  # Inicializar punto temporal
                self.is_drawing_line = True
                print(f"🎨 Iniciando dibujo de línea en: ({nx:.3f}, {ny:.3f})")
            else:
                # Segundo clic: finalizar línea
                self.line_end_norm = (nx, ny)
                self.temp_line_end_norm = None
                self.is_drawing_line = False
                self.line_draw_mode = False
                print(f"✅ Línea completada: ({self.line_start_norm[0]:.3f}, {self.line_start_norm[1]:.3f}) → ({nx:.3f}, {ny:.3f})")
                messagebox.showinfo("Línea", "Línea de activación configurada")
        except Exception as e:
            print(f"Error en clic: {e}")
    
    def on_video_motion(self, event):
        """Capturar movimiento del mouse para mostrar línea temporal"""
        if not self.line_draw_mode or not self.is_drawing_line:
            return
        try:
            # Calcular normalización respecto al tamaño actual del label
            label_width = event.widget.winfo_width()
            label_height = event.widget.winfo_height()
            if label_width <= 1 or label_height <= 1:
                return
            
            nx = max(0.0, min(1.0, event.x / float(label_width)))
            ny = max(0.0, min(1.0, event.y / float(label_height)))
            
            # Actualizar punto temporal para mostrar línea mientras arrastra
            self.temp_line_end_norm = (nx, ny)
        except Exception:
            pass
    
    def get_model(self):
        """Obtener modelo seleccionado"""
        return self.model_var.get()
    
    def get_render_every(self):
        """Obtener cada cuántos frames renderizar"""
        return self.render_every_var.get()
    
    def get_arduino_debounce(self):
        """Obtener tiempo de estabilidad de Arduino"""
        return self.arduino_debounce_var.get()
    
    def get_arduino_cooldown(self):
        """Obtener tiempo de cooldown de Arduino"""
        return self.arduino_cooldown_var.get()
    
    def get_line_detection_radius(self):
        """Obtener radio de detección de línea"""
        return self.line_detection_radius_var.get()
    
    def get_line_coordinates(self):
        """Obtener coordenadas de la línea"""
        return self.line_start_norm, self.line_end_norm
    
    def get_temp_line_coordinates(self):
        """Obtener coordenadas de la línea temporal (mientras se dibuja)"""
        return self.line_start_norm, self.temp_line_end_norm
    
    def has_line(self):
        """Verificar si hay línea configurada"""
        return self.line_start_norm is not None and self.line_end_norm is not None
    
    def is_line_draw_mode(self):
        """Verificar si está en modo dibujo de línea"""
        return self.line_draw_mode
    
    def is_drawing_line(self):
        """Verificar si está actualmente dibujando una línea"""
        return self.is_drawing_line

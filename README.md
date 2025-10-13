# 🥜 Detector de Castañas con YOLO12n

Sistema de detección y clasificación de castañas brasileñas usando inteligencia artificial.

## 📋 Características

- **Detección en tiempo real** con YOLO12n
- **Interfaz Gráfica Completa** con tkinter para visualización y control
- **Análisis RGB directo** para clasificación de calidad (SANA vs CONTAMINADA)
- **Control de Arduino** para automatización
- **Interfaz de cámara múltiple** con configuración personalizable
- **Demo Interactivo** sin necesidad de cámara real

## 🚀 Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecuta el sistema:
```bash
# Demo interactivo (sin cámara)
python demo_interface.py

# Interfaz gráfica completa
python interface.py

# Sistema principal (consola)
python main.py

# Menú de demostración
python show_demo.py
```

## 📊 Funcionalidades

### Interfaz Gráfica
- **Visualización en tiempo real** de detecciones
- **Configuración de colores** personalizable
- **Selección de cámara** con información detallada
- **Estadísticas en vivo** de detecciones y calidad
- **Panel de detecciones** con historial

### Detección de Castañas
- Clasificación automática: **Sanas** vs **Contaminadas**
- Análisis RGB directo: brillo, variación, densidad de bordes
- Detección de clases: sports ball, apple, orange, donut, bowl, carrot, banana


### Control de Hardware
- **Arduino Integration**: Control de servos para separación automática
- **Múltiples Cámaras**: Soporte para hasta 5 dispositivos
- **Configuración JSON**: Cámaras favoritas personalizables

## 🎯 Controles

- **ESC**: Salir
- **'c'**: Cambiar confianza de detección
- **'s'**: Guardar captura
- **'m'**: Cambiar cámara
- **'o'**: Optimización manual de memoria

## 📁 Estructura del Proyecto

```
├── main.py                    # Sistema principal (consola)
├── interface.py               # Interfaz gráfica completa
├── demo_interface.py          # Demo interactivo (sin cámara)
├── show_demo.py              # Menú de demostración
├── run_with_interface.py     # Ejecutor unificado
├── requirements.txt           # Dependencias Python
├── camera_config.json        # Configuración de cámaras
├── interface_config.json     # Configuración de interfaz
├── functions/
│   ├── __init__.py
│   └── analysys.py          # Análisis RGB y clasificación
├── core/
│   └── yolo12n.pt           # Modelo YOLO12n
└── arduino/
    ├── __init__.py
    └── arduino_manager.py    # Control de Arduino
```

## 🔧 Configuración

### Cámaras
Edita `camera_config.json` para personalizar tus dispositivos:
```json
{
    "cameras": {
        "0": {
            "name": "Cámara Principal",
            "description": "Webcam integrada",
            "is_favorite": true
        },
        "1": {
            "name": "Cámara Externa",
            "description": "Cámara USB externa",
            "is_favorite": false
        }
    }
}
```

### Interfaz Gráfica
Configura colores y parámetros en `interface_config.json`:
```json
{
    "colors": {
        "sana": "#00FF00",
        "contaminada": "#FF0000",
        "no_confirmada": "#FFFF00"
    },
    "rgb_thresholds": {
        "brightness_threshold_low": 70,
        "brightness_threshold_medium": 100,
        "variation_threshold": 35,
        "edge_density_threshold": 0.15,
        "contamination_threshold": 1
    }
}
```

# 🥜 Detector de Castañas con YOLO12n

Sistema de detección y clasificación de castañas brasileñas usando inteligencia artificial.

## 📋 Características

- **Detección en tiempo real** con YOLO12n
- **Análisis dual RGB + UV simulado** para máxima precisión
- **Filtros especializados** para detección de contaminación fúngica
- **Control de Arduino** para automatización
- **Interfaz de cámara múltiple** con configuración personalizable

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/rasoky2/PruebaYolov12.git
cd PruebaYolov12
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta el detector:
```bash
python main.py
```

## 📊 Funcionalidades

### Detección de Castañas
- Clasificación automática: **Sanas** vs **Contaminadas**
- Análisis HSV + textura para detección de moho
- Simulación UV para fluorescencia de contaminantes

### Filtros Especializados
- **NIR Avanzado**: Análisis de textura con Haralick
- **Spectral**: PCA y anomalías espectrales
- **Pipeline Completo**: Análisis exhaustivo automatizado
- **Filtros para Hongos**: Detección específica de contaminación fúngica

### Control de Hardware
- **Arduino Integration**: Control de servos para separación automática
- **Múltiples Cámaras**: Soporte para hasta 5 dispositivos
- **Configuración JSON**: Cámaras favoritas y filtros personalizables

## 🎯 Controles

- **ESC**: Salir
- **'c'**: Cambiar confianza de detección
- **'s'**: Guardar captura
- **'m'**: Cambiar cámara
- **'f'**: Cambiar filtro
- **'o'**: Optimización manual de memoria

## 📁 Estructura del Proyecto

```
├── main.py                 # Aplicación principal
├── requirements.txt        # Dependencias Python
├── camera_config.json     # Configuración de cámaras
├── filter_config.json     # Configuración de filtros
├── core/
│   └── yolo12n.pt        # Modelo YOLO12n
├── image/
│   ├── __init__.py
│   └── filters.py        # Sistema de filtros
└── arduino/
    ├── __init__.py
    └── arduino_manager.py # Control de Arduino
```

## 🔧 Configuración

### Cámaras
Edita `camera_config.json` para personalizar tus dispositivos:
```json
{
    "cameras": {
        "0": {
            "name": "Cámara Principal",
            "is_favorite": true
        }
    }
}
```

### Filtros
Configura filtros favoritos en `filter_config.json`:
```json
{
    "settings": {
        "default_favorite": "mold_texture"
    }
}
```


**Desarrollado con ❤️ para la industria de castañas brasileñas**

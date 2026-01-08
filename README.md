# 🍎 Sistema de Inspección de Calidad de Frutas

Sistema de detección y clasificación de calidad para **Plátanos, Manzanas y Naranjas** utilizando Inteligencia Artificial (YOLO) y análisis de imagen.

## 📋 Características

- **Sistemas Expertos por Fruta**: Modelos especializados para detectar estados de madurez y enfermedades específicas.
- **Detección en tiempo real**: Procesamiento fluido con YOLOv12.
- **Interfaz Premium (Shadcn Light)**: Diseño moderno, limpio y minimalista basado en el estándar Shadcn.
- **Traducción Automática**: Todas las clases detectadas se muestran en español gracias a un sistema de mapeo JSON.
- **Selección Inteligente**: Carga automática del modelo y configuración según la fruta seleccionada.
- **Detección de Enfermedades**: Identifica Cancrosis, Mancha Negra, HLB, Moho, Podredumbre, entre otros.

## 🚀 Instalación

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la aplicación principal:

```bash
python main.py
```

## 📊 Funcionalidades

### Interfaz de Control

- **Selector de Fruta**: Cambia entre Plátano, Manzana y Naranja instantáneamente.
- **Visualización en Vivo**: Video en tiempo real con etiquetas traducidas y estados de calidad.
- **Gestión de Cámara**: Selección de dispositivos con información de resolución y FPS.
- **Sistema de Logs**: Registro interno de eventos y errores (ahora oculto para una interfaz más limpia).

### Clasificación de Calidad

- **Sana**: Fruta fresca, madura o apta para consumo.
- **Contaminada**: Detecta enfermedades, daños físicos o estados de descomposición.
- **Traducciones incluidas**:
  - _Plátano_: Maduro, Verde, Podrido, Pasado.
  - _Manzana_: Fresca, Podrida.
  - _Naranja_: HLB (Greening), Cancrosis, Mancha Negra.

## 📁 Estructura del Proyecto

```
├── main.py                    # Punto de entrada de la aplicación
├── extract_classes.py         # Herramienta para generar JSON de clases y traducciones
├── core/                      # Modelos de IA
│   ├── platano/               # Modelo y clases para plátanos
│   ├── manzana/               # Modelo y clases para manzanas
│   └── naranja/               # Modelo y clases para naranjas
├── interface/                 # Componentes de la Interfaz Gráfica (PyQt6)
│   ├── main_window.py         # Ventana principal y lógica de UI
│   ├── controls_panel.py      # Panel de selección de fruta
│   └── styles.py              # Definición de estética Shadcn Light (Inter font)
└── utils/                     # Utilidades de logging y sistema
```

## 🔧 Herramientas Útiles

### Generador de Clases (`extract_classes.py`)

Puedes usar este script para extraer las clases de cualquier nuevo modelo `.pt` y generar automáticamente su archivo de traducciones:

```bash
python extract_classes.py ruta/al/modelo.pt
```

## 🎨 Estética

La interfaz utiliza la fuente **Inter** y una paleta de colores basada en **Shadcn Light** (Zinc), proporcionando una experiencia de usuario de nivel profesional y alto contraste.

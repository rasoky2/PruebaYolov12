import json
import os
import sys
from ultralytics import YOLO

# Mapeo de traducciones al español
TRANSLATIONS = {
    # Madurez
    "overripe": "Muy Maduro",
    "raw": "Crudo",
    "ripe": "Maduro",
    "unripe": "Verde / Inmaduro",
    "fresh": "Fresco",
    "stale": "Pasado",
    
    # Estados de Calidad
    "rotten": "Podrido / Malogrado",
    "damaged": "Dañado",
    "bruised": "Machucado",
    
    # Frutas y combinaciones comunes
    "apple": "Manzana",
    "fresh_apple": "Manzana Fresca",
    "rotten_apple": "Manzana Podrida",
    
    "orange": "Naranja",
    "fresh_orange": "Naranja Fresca",
    "rotten_orange": "Naranja Podrida",
    
    "banana": "Plátano",
    "fresh_banana": "Plátano Fresco",
    "rotten_banana": "Plátano Podrido",
    
    # Otros / Enfermedades específicas
    "wrinkled": "Arrugado",
    "mold": "Moho",
    "blackspot": "Mancha Negra",
    "canker": "Cancrosis (Cancro)",
    "greening": "Greening (HLB)"
}

def extract_classes(model_path):
    # Verificar si el modelo existe
    if not os.path.exists(model_path):
        print(f"Error: El modelo no se encuentra en {model_path}")
        return

    print(f"Cargando modelo desde: {model_path}")
    try:
        model = YOLO(model_path)
        
        # Obtener nombres de clases
        classes_dict = model.names
        print(f"Clases detectadas: {classes_dict}")

        # Crear estructura enriquecida con español y clasificación de calidad
        enhanced_classes = {}
        
        # Palabras clave que indican contaminación/daño
        bad_keywords = ["rotten", "bad", "canker", "blackspot", "greening", 
                       "mold", "stale", "damaged", "bruised", "overripe"]
        
        for class_id, class_name in classes_dict.items():
            # Determinar calidad basada en palabras clave
            is_contaminated = any(keyword in class_name.lower() for keyword in bad_keywords)
            quality = "contaminada" if is_contaminated else "sana"
            
            enhanced_classes[class_id] = {
                "name": class_name,
                "es": TRANSLATIONS.get(class_name, class_name),
                "quality": quality
            }
        
        # Ruta de salida: mismo directorio que el modelo
        output_dir = os.path.dirname(model_path)
        output_path = os.path.join(output_dir, "classes.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_classes, f, indent=4, ensure_ascii=False)
            
        print(f"Archivo JSON actualizado exitosamente en: {output_path}")
        
    except Exception as e:
        print(f"Error al procesar el modelo: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo manual: procesar un modelo específico
        target_model = sys.argv[1]
        extract_classes(target_model)
    else:
        # Modo automático: procesar todos los modelos en las carpetas de frutas
        base_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.join(base_dir, "core")
        
        fruit_folders = ["platano", "manzana", "naranja"]
        
        print("=" * 60)
        print("Generando classes.json para todos los modelos...")
        print("=" * 60)
        
        for fruit in fruit_folders:
            fruit_path = os.path.join(core_dir, fruit)
            if not os.path.exists(fruit_path):
                print(f"⚠️  Carpeta {fruit} no encontrada, saltando...")
                continue
            
            # Buscar archivos .pt en la carpeta
            pt_files = [f for f in os.listdir(fruit_path) if f.endswith('.pt')]
            
            if not pt_files:
                print(f"⚠️  No se encontraron modelos .pt en {fruit}/")
                continue
            
            # Procesar el primer .pt encontrado (o best.pt si existe)
            best_pt = [f for f in pt_files if 'best' in f.lower()]
            model_file = best_pt[0] if best_pt else pt_files[0]
            model_path = os.path.join(fruit_path, model_file)
            
            print(f"\n📦 Procesando {fruit.upper()}: {model_file}")
            extract_classes(model_path)
        
        print("\n" + "=" * 60)
        print("✅ Proceso completado")
        print("=" * 60)


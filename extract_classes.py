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

        # Crear estructura enriquecida con español y calidad
        enhanced_classes = {}
        for class_id, class_name in classes_dict.items():
            cn_lower = class_name.lower()
            
            # Determinar calidad sugerida
            bad_keywords = ["bad", "rot", "canker", "blackspot", "greening", "mold", "stale", "damaged", "bruised", "wrinkled", "overripe"]
            quality = "contaminada" if any(kw in cn_lower for kw in bad_keywords) else "sana"
            
            enhanced_classes[class_id] = {
                "name": class_name,
                "es": TRANSLATIONS.get(cn_lower, class_name), # Usar cn_lower para mayor coincidencia
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
        target_model = sys.argv[1]
    else:
        # Default fallback
        target_model = r"d:\CastañaSerial\core\platano\best.pt"
    
    extract_classes(target_model)

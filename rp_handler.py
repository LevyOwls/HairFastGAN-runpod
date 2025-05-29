import runpod
import base64
import io
from PIL import Image
import numpy as np
from hair_swap import HairFast, get_parser
import torch

# Inicializar HairFast
hair_fast = HairFast(get_parser().parse_args([]))

def base64_to_pil(base64_string):
    """Convierte una imagen base64 a PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def pil_to_base64(pil_image):
    """Convierte una PIL Image a base64."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def handler(event):
    """
    Handler para RunPod que procesa las imágenes y retorna el resultado.
    
    Input:
    {
        "input": {
            "face_image": "base64_string",
            "shape_image": "base64_string",
            "color_image": "base64_string"
        }
    }
    """
    try:
        # Obtener las imágenes del input
        face_b64 = event["input"]["face_image"]
        shape_b64 = event["input"]["shape_image"]
        color_b64 = event["input"]["color_image"]

        # Convertir base64 a PIL Images
        face_img = base64_to_pil(face_b64)
        shape_img = base64_to_pil(shape_b64)
        color_img = base64_to_pil(color_b64)

        # Procesar las imágenes con HairFast
        result = hair_fast(face_img, shape_img, color_img)

        # Convertir el resultado a base64
        result_b64 = pil_to_base64(result)

        return {
            "status": "success",
            "output": {
                "result_image": result_b64
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Iniciar el servidor RunPod
runpod.serverless.start({"handler": handler}) 
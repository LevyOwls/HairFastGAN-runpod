import runpod
import base64
import io
from PIL import Image
from hair_swap import HairFast, get_parser

# Inicializar HairFast con los parámetros por defecto según el README
hair_fast = HairFast(get_parser().parse_args([]))

def handler(event):
    """
    Handler para RunPod basado en el README oficial de HairFastGAN.
    Se espera un payload JSON con:
    {
      "input": {
        "face_path":  "/ruta/a/face.png",
        "shape_path": "/ruta/a/shape.png",
        "color_path": "/ruta/a/color.png"
      }
    }
    Devuelve:
    {
      "result_image": "<PNG base64>"
    }
    """
    data = event.get('input', {})
    face_path  = data.get('face_path')
    shape_path = data.get('shape_path')
    color_path = data.get('color_path')
    if not all([face_path, shape_path, color_path]):
        raise ValueError('Se requieren face_path, shape_path y color_path en el payload')

    # Cargar imágenes desde disco
    face_img  = Image.open(face_path).convert('RGB')
    shape_img = Image.open(shape_path).convert('RGB')
    color_img = Image.open(color_path).convert('RGB')

    # Inferencia (transferencia de peinado)
    result_img = hair_fast(face_img, shape_img, color_img)

    # Convertir resultado a base64 PNG
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    result_b64 = base64.b64encode(buf.getvalue()).decode()

    return {'result_image': result_b64}

if __name__ == '__main__':
    runpod.serverless.start(handler)

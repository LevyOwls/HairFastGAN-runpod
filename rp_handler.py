import runpod
import base64
import io
from PIL import Image
from hair_swap import HairFast, get_parser

# Configurar el tamaño del modelo según entrenamiento (512)
IMAGE_SIZE = 512

# Inicializar HairFast con parámetros por defecto
hair_fast = HairFast(get_parser().parse_args(["--size", str(IMAGE_SIZE)]))

def base64_to_pil(b64: str) -> Image.Image:
    """Convierte un string base64 a PIL Image en RGB."""
    if ',' in b64:
        b64 = b64.split(',')[1]
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert('RGB')


def preprocess(img: Image.Image) -> Image.Image:
    """Recorta centrado al cuadrado y redimensiona a IMAGE_SIZE."""
    w, h = img.size
    m = min(w, h)
    left, top = (w - m)//2, (h - m)//2
    cropped = img.crop((left, top, left + m, top + m))
    return cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def pil_to_base64(img: Image.Image) -> str:
    """Convierte PIL Image a string base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def handler(event):
    """
    Handler para RunPod que procesa base64 de 3 imágenes y devuelve el resultado.
    Espera payload:
    {
      "input": {
        "face_image": "<base64>",
        "shape_image": "<base64>",
        "color_image": "<base64>"
      }
    }
    """
    data = event.get('input', {})
    face_b64  = data.get('face_image')
    shape_b64 = data.get('shape_image')
    color_b64 = data.get('color_image')
    if not all([face_b64, shape_b64, color_b64]):
        raise ValueError('Se requieren base64: face_image, shape_image y color_image en el payload')

    # Decodificar y preprocesar
    face  = preprocess(base64_to_pil(face_b64))
    shape = preprocess(base64_to_pil(shape_b64))
    color = preprocess(base64_to_pil(color_b64))

    # Inferencia
    result_img = hair_fast(face, shape, color)

    # Convertir a base64 y retornar JSON
    return {
        'status': 'success',
        'output': {'result_image': pil_to_base64(result_img)}
    }

if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})

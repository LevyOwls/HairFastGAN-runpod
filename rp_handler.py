import runpod
import base64
import io
from PIL import Image
from hair_swap import HairFast, get_parser

# Definimos el tamaño que el modelo va a esperar
IMAGE_SIZE = 512

# Inicializamos HairFast PASANDO --size para que el modelo esté en 512×512
def init_hairfast():
    parser = get_parser()
    args = parser.parse_args(["--size", str(IMAGE_SIZE)])
    return HairFast(args)

hair_fast = init_hairfast()

def base64_to_pil(b64: str) -> Image.Image:
    """Convierte base64 a PIL RGB."""
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
    """Convierte PIL a base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# Handler serverless

def handler(event):
    data = event.get('input', {})
    face_b64  = data.get('face_image')
    shape_b64 = data.get('shape_image')
    color_b64 = data.get('color_image')
    if not all([face_b64, shape_b64, color_b64]):
        raise ValueError('Faltan face_image, shape_image o color_image')

    # Convertir y preprocesar
    face  = preprocess(base64_to_pil(face_b64))
    shape = preprocess(base64_to_pil(shape_b64))
    color = preprocess(base64_to_pil(color_b64))

    # Inferencia y obtención de tensor de salida
    output_img = hair_fast(face, shape, color)

    # Convertir a base64 y devolver
    return {
        'status': 'success',
        'output': {'result_image': pil_to_base64(output_img)}
    }

if __name__ == '__main__':
    runpod.serverless.start(handler)

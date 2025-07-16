import os
import urllib.request
from dotenv import load_dotenv
try:
    from .yolocounterv1 import YoloOnnx
except:
    from yolocounterv1 import YoloOnnx

# Cargar variables de entorno
load_dotenv()

# Clases del modelo
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
 'hair drier', 'toothbrush']

# Ruta local y en la nube
local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cloud_dir = os.getenv("S3_BUCKET_URL")
filename = "yolov7_training.onnx"

# Intentar cargar local o descargar desde S3
# try:
yolopath = os.path.join(local_dir, filename)
with open(yolopath):
    print('Cargando archivo local:', filename)
# except:
#     yolopath = os.path.join('/tmp', filename)
#     urllib.request.urlretrieve(cloud_dir + filename, yolopath)
#     print('Descargado desde S3:', filename)

# Instancia del modelo
yolo = YoloOnnx(weigths_path=yolopath, class_names=class_names, cuda=False)

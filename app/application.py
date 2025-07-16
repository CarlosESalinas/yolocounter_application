import os
from flask import Flask, render_template, request, jsonify
from PIL import Image

try:
    from yolomodel import yolo  
except:
    from .yolomodel import yolo  

# Configuración de rutas
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

application = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/detect-count', methods=['POST'])
def predict():
    # 1. Verificar que se haya subido un archivo
    if 'image' not in request.files:
        print("No se proporcionó archivo en la solicitud")
        return jsonify({'error': 'No se proporcionó imagen'}), 400
    
    file = request.files['image']
    
    # 2. Verificar que el archivo tenga nombre
    if file.filename == '':
        print("Se envió un archivo sin nombre")
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    # 3. Verificar extensión del archivo
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        print(f"Extensión de archivo no permitida: {file.filename}")
        return jsonify({'error': 'Tipo de archivo no soportado'}), 400
    
    try:
        # 4. Verificar que sea una imagen válida
        file.stream.seek(0)
        image = Image.open(file.stream)
        image.verify()
        file.stream.seek(0)
        image = Image.open(file.stream)
        
        # 5. Procesar la imagen con YOLO
        try:
            _, outputs, c_classes = yolo.inference(file)
            detections = [
                (yolo.convertbox([x0, y0, x1, y1]), int(cls_id), str(prob), yolo.class_names[int(cls_id)])
                for (batch_id, x0, y0, x1, y1, cls_id, prob) in outputs
            ]
            
            return jsonify(countings=c_classes, detections=detections)
            
        except Exception as yolo_error:
            print(f"Error en el modelo YOLO: {str(yolo_error)}")
            return jsonify({
                'error': 'Error en el procesamiento del modelo',
                'details': str(yolo_error)
            }), 500
            
    except (IOError, OSError) as e:
        print(f"Error al procesar imagen: {str(e)}")
        return jsonify({'error': 'El archivo no es una imagen válida o está corrupto'}), 400
        
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500




if __name__ == "__main__":
    application.run(debug=True)

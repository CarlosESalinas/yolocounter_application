# import sys
# import os

# # Agrega el directorio donde está yolocounter al sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from yolocounter.application import application as app
# import pytest

# @pytest.fixture
# def client():
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client


import sys
import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from PIL import Image
import io

# Configuración de paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

# Importar la aplicación
try:
    from app.application import application as app
except ImportError:
    # Fallback si el import directo no funciona
    sys.path.insert(0, os.path.join(parent_dir, 'yolocounter'))
    from app.application import application as app

@pytest.fixture
def client():
    """Cliente de prueba para Flask"""
    # Configurar la aplicación para testing
    app.config['TESTING'] = True
    app.config['DEBUG'] = False
    app.config['WTF_CSRF_ENABLED'] = False  # Desactivar CSRF para pruebas
    
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def app_context():
    """Context manager para la aplicación Flask"""
    with app.app_context():
        yield app

@pytest.fixture
def sample_image():
    """Proporciona una imagen de muestra válida"""
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def large_image():
    """Proporciona una imagen grande para pruebas"""
    img = Image.new('RGB', (1000, 1000), color='blue')
    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def small_image():
    """Proporciona una imagen pequeña para pruebas"""
    img = Image.new('RGB', (10, 10), color='green')
    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def png_image():
    """Proporciona una imagen PNG"""
    img = Image.new('RGB', (100, 100), color='yellow')
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def invalid_image():
    """Proporciona datos inválidos que no son una imagen"""
    return io.BytesIO(b'This is not an image file')

@pytest.fixture
def corrupted_image():
    """Proporciona datos corruptos que parecen imagen"""
    # Comenzar con headers JPEG válidos pero datos corruptos
    corrupted_data = b'\xff\xd8\xff\xe0' + b'corrupted data' * 100
    return io.BytesIO(corrupted_data)

@pytest.fixture
def empty_file():
    """Proporciona un archivo vacío"""
    return io.BytesIO(b'')

@pytest.fixture
def temp_dir():
    """Proporciona un directorio temporal para pruebas"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_yolo_success():
    """Mock de YOLO que simula detección exitosa"""
    with patch('yolocounter.yolomodel.yolo') as mock:
        mock.inference.return_value = (
            None,  # Primera salida no usada
            [(0, 10, 10, 50, 50, 0, 0.9)],  # outputs
            {'person': 1}  # c_classes
        )
        mock.convertbox.return_value = [10, 10, 50, 50]
        mock.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus']
        yield mock

@pytest.fixture
def mock_yolo_no_detections():
    """Mock de YOLO que simula no detecciones"""
    with patch('yolocounter.yolomodel.yolo') as mock:
        mock.inference.return_value = (None, [], {})
        mock.convertbox.return_value = []
        mock.class_names = ['person', 'bicycle', 'car']
        yield mock

@pytest.fixture
def mock_yolo_multiple_detections():
    """Mock de YOLO que simula múltiples detecciones"""
    with patch('yolocounter.yolomodel.yolo') as mock:
        mock.inference.return_value = (
            None,
            [
                (0, 10, 10, 50, 50, 0, 0.9),   # persona
                (0, 60, 60, 100, 100, 2, 0.8), # carro
                (0, 20, 20, 40, 40, 0, 0.7)    # otra persona
            ],
            {'person': 2, 'car': 1}
        )
        mock.convertbox.side_effect = [
            [10, 10, 50, 50],
            [60, 60, 100, 100],
            [20, 20, 40, 40]
        ]
        mock.class_names = ['person', 'bicycle', 'car']
        yield mock

@pytest.fixture
def mock_yolo_error():
    """Mock de YOLO que simula error en la inferencia"""
    with patch('yolocounter.yolomodel.yolo') as mock:
        mock.inference.side_effect = Exception("YOLO model error")
        mock.class_names = ['person', 'bicycle', 'car']
        yield mock

@pytest.fixture
def mock_yolo_timeout():
    """Mock de YOLO que simula timeout"""
    with patch('yolocounter.yolomodel.yolo') as mock:
        import time
        def slow_inference(*args, **kwargs):
            time.sleep(10)  # Simular operación muy lenta
            return (None, [], {})
        
        mock.inference.side_effect = slow_inference
        mock.class_names = ['person', 'bicycle', 'car']
        yield mock

# Configuración de pytest
def pytest_configure(config):
    """Configuración personalizada de pytest"""
    config.addinivalue_line(
        "markers", "slow: marca las pruebas como lentas"
    )
    config.addinivalue_line(
        "markers", "integration: marca las pruebas de integración"
    )
    config.addinivalue_line(
        "markers", "unit: marca las pruebas unitarias"
    )

@pytest.fixture(scope="session")
def test_data_dir():
    """Directorio con datos de prueba"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_dir, exist_ok=True)
    
    # Crear algunas imágenes de prueba si no existen
    test_images = [
        ('test_image.jpg', 'JPEG', (100, 100), 'red'),
        ('test_image.png', 'PNG', (200, 200), 'blue'),
        ('small_image.jpg', 'JPEG', (50, 50), 'green'),
        ('large_image.jpg', 'JPEG', (500, 500), 'yellow')
    ]
    
    for filename, format_name, size, color in test_images:
        filepath = os.path.join(test_dir, filename)
        if not os.path.exists(filepath):
            img = Image.new('RGB', size, color)
            img.save(filepath, format=format_name)
    
    yield test_dir

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configuración automática para cada prueba"""
    # Configurar variables de entorno para pruebas
    os.environ['TESTING'] = 'true'
    os.environ['FLASK_ENV'] = 'testing'
    
    yield
    
    # Limpiar después de cada prueba
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'FLASK_ENV' in os.environ:
        del os.environ['FLASK_ENV']

# Helpers para pruebas
def create_test_image(size=(100, 100), color='red', format_name='JPEG'):
    """Helper para crear imágenes de prueba"""
    img = Image.new('RGB', size, color)
    img_io = io.BytesIO()
    img.save(img_io, format=format_name)
    img_io.seek(0)
    return img_io

def create_file_data(image_io, filename):
    """Helper para crear datos de archivo para peticiones POST"""
    return {
        'image': (image_io, filename)
    }

# Markers personalizados para categorizar pruebas
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
]
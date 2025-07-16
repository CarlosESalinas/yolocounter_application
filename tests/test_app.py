# def test_index_route(client):
#     response = client.get('/')
#     assert response.status_code == 200
#     assert b"Yolo Counter" in response.data

# def test_detect_route_without_image(client):
#     response = client.post('/detect-count', data={})
#     assert response.status_code == 400 or response.status_code == 500  # No image = error

import pytest
import json
import io
import os
from PIL import Image
import tempfile
from unittest.mock import patch, MagicMock

class TestIndexRoute:
    """Pruebas para la ruta principal"""
    
    def test_index_route_success(self, client):
        """Prueba que la página principal carga correctamente"""
        response = client.get('/')
        assert response.status_code == 200
        assert b"Yolo Counter" in response.data
        
    def test_index_route_content_type(self, client):
        """Prueba que la respuesta sea HTML"""
        response = client.get('/')
        assert response.content_type == 'text/html; charset=utf-8'


class TestDetectRoute:
    """Pruebas para el endpoint de detección"""
    
    def test_detect_route_without_image(self, client):
        """Prueba que falla sin imagen"""
        response = client.post('/detect-count', data={})
        assert response.status_code in [400, 500]
    
    def test_detect_route_with_invalid_file(self, client):
        """Prueba con archivo inválido"""
        data = {
            'image': (io.BytesIO(b'not an image'), 'test.txt')
        }
        response = client.post('/detect-count', data=data)
        assert response.status_code in [400, 500]
    
    def test_detect_route_with_empty_file(self, client):
        """Prueba con archivo vacío"""
        data = {
            'image': (io.BytesIO(b''), 'empty.jpg')
        }
        response = client.post('/detect-count', data=data)
        assert response.status_code in [400, 500]
    
    def test_detect_route_wrong_method(self, client):
        """Prueba que GET no está permitido en /detect-count"""
        response = client.get('/detect-count')
        assert response.status_code == 405  # Method Not Allowed
    
    def test_detect_route_wrong_field_name(self, client):
        """Prueba con nombre de campo incorrecto"""
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'wrong_field': (img_io, 'test.jpg')  # Campo incorrecto
        }
        response = client.post('/detect-count', data=data)
        assert response.status_code in [400, 500]

    # @patch('app.yolomodel.yolo')
    @patch('app.application.yolo')
    def test_detect_route_with_valid_image_mock(self, mock_yolo, client):
        """Prueba con imagen válida usando mock"""
        
        mock_yolo.inference.return_value = (
            None,  # Primera salida no usada
            [(0, 10, 10, 50, 50, 0, 0.9)],  # outputs: (batch_id, x0, y0, x1, y1, cls_id, prob)
            {'person': 1}  # c_classes
        )
        mock_yolo.convertbox.return_value = [10, 10, 50, 50]
        mock_yolo.class_names = ['person', 'bicycle', 'car']
        
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'test.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200
        
        # Verificar que la respuesta sea JSON
        assert response.content_type == 'application/json'
        
        # Verificar estructura de respuesta
        json_data = json.loads(response.data)
        assert 'countings' in json_data
        assert 'detections' in json_data
        assert json_data['countings']['person'] == 1
        assert len(json_data['detections']) == 1

    @patch('app.yolomodel.yolo')
    def test_detect_route_no_detections(self, mock_yolo, client):
        """Prueba cuando no se detectan objetos"""
        # Configurar mock sin detecciones
        mock_yolo.inference.return_value = (
            None,
            [],  # Sin detecciones
            {}   # Sin conteos
        )
        
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='blue')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'test.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200
        
        json_data = json.loads(response.data)
        assert json_data['countings'] == {}
        assert json_data['detections'] == []

    # @patch('app.yolomodel.yolo')
    @patch('app.application.yolo')
    def test_detect_route_multiple_detections(self, mock_yolo, client):
        """Prueba con múltiples detecciones"""
        # Configurar mock con múltiples detecciones
        mock_yolo.inference.return_value = (
            None,
            [
                (0, 10, 10, 50, 50, 0, 0.9),  # persona
                (0, 60, 60, 100, 100, 2, 0.8),  # carro
                (0, 20, 20, 40, 40, 0, 0.7)   # otra persona
            ],
            {'person': 2, 'car': 1}
        )
        mock_yolo.convertbox.side_effect = [
            [10, 10, 50, 50],
            [60, 60, 100, 100],
            [20, 20, 40, 40]
        ]
        mock_yolo.class_names = ['person', 'bicycle', 'car']
        
        # Crear imagen válida
        img = Image.new('RGB', (200, 200), color='green')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'test.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200
        
        json_data = json.loads(response.data)
        assert json_data['countings']['person'] == 2
        assert json_data['countings']['car'] == 1
        assert len(json_data['detections']) == 3

    @patch('app.application.yolo')
    def test_detect_route_yolo_exception(self, mock_yolo, client):
        """Prueba cuando YOLO lanza una excepción"""
        # Configurar mock para lanzar excepción
        mock_yolo.inference.side_effect = Exception("YOLO model error")
        
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'test.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 500


class TestImageFormats:
    """Pruebas con diferentes formatos de imagen"""
    
    @pytest.mark.parametrize("format_name,mime_type", [
        ('JPEG', 'image/jpeg'),
        ('PNG', 'image/png'),
        ('BMP', 'image/bmp'),
    ])
    @patch('app.yolomodel.yolo')
    def test_different_image_formats(self, mock_yolo, client, format_name, mime_type):
        """Prueba con diferentes formatos de imagen"""
        # Configurar mock
        mock_yolo.inference.return_value = (None, [], {})
        
        # Crear imagen en el formato especificado
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format=format_name)
        img_io.seek(0)
        
        data = {
            'image': (img_io, f'test.{format_name.lower()}')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200

    @patch('app.yolomodel.yolo')
    def test_large_image(self, mock_yolo, client):
        """Prueba con imagen grande"""
        # Configurar mock
        mock_yolo.inference.return_value = (None, [], {})
        
        # Crear imagen grande
        img = Image.new('RGB', (2000, 2000), color='blue')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'large_image.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200

    @patch('app.yolomodel.yolo')
    def test_small_image(self, mock_yolo, client):
        """Prueba con imagen muy pequeña"""
        # Configurar mock
        mock_yolo.inference.return_value = (None, [], {})
        
        # Crear imagen pequeña
        img = Image.new('RGB', (10, 10), color='yellow')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'small_image.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200


class TestResponseValidation:
    """Pruebas para validar la estructura de respuesta"""
    
    @patch('app.yolomodel.yolo')
    def test_response_structure(self, mock_yolo, client):
        """Prueba que la respuesta tenga la estructura correcta"""
        # Configurar mock
        mock_yolo.inference.return_value = (
            None,
            [(0, 10, 10, 50, 50, 0, 0.9)],
            {'person': 1}
        )
        mock_yolo.convertbox.return_value = [10, 10, 50, 50]
        mock_yolo.class_names = ['person', 'bicycle', 'car']
        
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'test.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        json_data = json.loads(response.data)
        
        # Verificar estructura básica
        assert isinstance(json_data, dict)
        assert 'countings' in json_data
        assert 'detections' in json_data
        
        # Verificar tipos
        assert isinstance(json_data['countings'], dict)
        assert isinstance(json_data['detections'], list)
        
        # Verificar estructura de detecciones
        if json_data['detections']:
            detection = json_data['detections'][0]
            assert isinstance(detection, list)
            assert len(detection) == 4  # bbox, cls_id, prob, class_name
            assert isinstance(detection[0], list)  # bbox
            assert isinstance(detection[1], int)   # cls_id
            assert isinstance(detection[2], str)   # prob
            assert isinstance(detection[3], str)   # class_name


class TestEdgeCases:
    """Pruebas para casos límite"""
    
    def test_nonexistent_route(self, client):
        """Prueba ruta inexistente"""
        response = client.get('/nonexistent')
        assert response.status_code == 404
    
    @patch('app.yolomodel.yolo')
    def test_corrupted_image_data(self, mock_yolo, client):
        """Prueba con datos de imagen corruptos"""
        # Crear datos corruptos que parecen imagen
        corrupted_data = b'\xff\xd8\xff\xe0' + b'corrupted data' * 100
        
        data = {
            'image': (io.BytesIO(corrupted_data), 'corrupted.jpg')
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code in [400, 500]
    
    @patch('app.yolomodel.yolo')
    def test_very_long_filename(self, mock_yolo, client):
        """Prueba con nombre de archivo muy largo"""
        # Configurar mock
        mock_yolo.inference.return_value = (None, [], {})
        
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        long_filename = 'a' * 255 + '.jpg'
        data = {
            'image': (img_io, long_filename)
        }
        
        response = client.post('/detect-count', data=data)
        assert response.status_code == 200


class TestPerformance:
    """Pruebas de rendimiento básicas"""
    
    @patch('app.yolomodel.yolo')
    def test_response_time(self, mock_yolo, client):
        """Prueba tiempo de respuesta básico"""
        import time
        
        # Configurar mock
        mock_yolo.inference.return_value = (None, [], {})
        
        # Crear imagen válida
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        
        data = {
            'image': (img_io, 'test.jpg')
        }
        
        start_time = time.time()
        response = client.post('/detect-count', data=data)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Debería responder en menos de 5 segundos


# Fixtures adicionales para pruebas específicas
@pytest.fixture
def sample_image():
    """Fixture que proporciona una imagen de muestra"""
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def mock_yolo_with_detections():
    """Fixture que proporciona un mock de YOLO con detecciones"""
    with patch('app.yolomodel.yolo') as mock:
        mock.inference.return_value = (
            None,
            [(0, 10, 10, 50, 50, 0, 0.9)],
            {'person': 1}
        )
        mock.convertbox.return_value = [10, 10, 50, 50]
        mock.class_names = ['person', 'bicycle', 'car']
        yield mock
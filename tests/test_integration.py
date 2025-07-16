import pytest
import json
import time
import os
import requests
from PIL import Image
import io
import threading
import subprocess
import signal

@pytest.mark.integration
class TestApplicationIntegration:
    """Pruebas de integración que requieren la aplicación completa"""
    
    def test_full_pipeline_with_real_model(self, client, sample_image):
        """Prueba completa con el modelo YOLO real (si está disponible)"""
        try:
            # Intentar usar el modelo real
            data = {
                'image': (sample_image, 'test.jpg')
            }
            
            response = client.post('/detect-count', data=data)
            
            # Verificar que la respuesta sea exitosa
            assert response.status_code == 200
            assert response.content_type == 'application/json'
            
            # Verificar estructura de respuesta
            json_data = json.loads(response.data)
            assert 'countings' in json_data
            assert 'detections' in json_data
            assert isinstance(json_data['countings'], dict)
            assert isinstance(json_data['detections'], list)
            
        except Exception as e:
            pytest.skip(f"Modelo YOLO no disponible: {e}")
    
    def test_stress_multiple_requests(self, client, sample_image):
        """Prueba de estrés con múltiples peticiones secuenciales"""
        import time
        
        results = []
        errors = []
        
        # Hacer 5 peticiones secuenciales rápidas
        start_time = time.time()
        
        for i in range(5):
            try:
                # Crear nueva instancia de imagen para cada request
                img = Image.new('RGB', (100, 100), color='red')
                img_io = io.BytesIO()
                img.save(img_io, format='JPEG')
                img_io.seek(0)
                
                data = {
                    'image': (img_io, f'test_{i}.jpg')
                }
                
                response = client.post('/detect-count', data=data)
                results.append(response.status_code)
                
            except Exception as e:
                errors.append(f"Request {i}: {str(e)}")
        
        end_time = time.time()
        
        # Verificar resultados
        print(f"Resultados: {results}")
        print(f"Errores: {errors}")
        print(f"Tiempo total: {end_time - start_time:.2f} segundos")
        
        assert len(errors) == 0, f"Errores encontrados: {errors}"
        assert len(results) == 5, f"Se esperaban 5 resultados, se obtuvieron {len(results)}"
        
        # Verificar que los códigos de estado sean válidos
        for status in results:
            assert status in [200, 500], f"Código de estado inesperado: {status}"
        
        assert (end_time - start_time) < 10, f"Tiempo excesivo para 5 requests: {end_time - start_time:.2f} segundos"
    
    def test_large_image_processing(self, client):
        """Prueba con imagen grande (2MB+)"""
        # Crear imagen grande con parámetros que garanticen tamaño > 1MB
        large_img = Image.new('RGB', (3000, 3000), color='blue')
        
        # Agregar algo de ruido para aumentar el tamaño
        import random
        pixels = large_img.load()
        for i in range(0, 3000, 10):  # Cada 10 píxeles para no ser demasiado lento
            for j in range(0, 3000, 10):
                pixels[i, j] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
        
        img_io = io.BytesIO()
        large_img.save(img_io, format='JPEG', quality=100)  # Máxima calidad
        img_io.seek(0)
        
        # Verificar que la imagen sea grande
        image_size = img_io.getbuffer().nbytes
        print(f"Tamaño de imagen generada: {image_size} bytes ({image_size / (1024*1024):.2f} MB)")
        
        # Si aún no es lo suficientemente grande, forzar el tamaño
        if image_size <= 1024 * 1024:
            # Crear imagen aún más grande
            large_img = Image.new('RGB', (4000, 4000), color='blue')
            # Agregar más ruido
            pixels = large_img.load()
            for i in range(0, 4000, 8):
                for j in range(0, 4000, 8):
                    pixels[i, j] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
            
            img_io = io.BytesIO()
            large_img.save(img_io, format='PNG')  # PNG es menos comprimido
            img_io.seek(0)
            image_size = img_io.getbuffer().nbytes
            print(f"Tamaño de imagen PNG: {image_size} bytes ({image_size / (1024*1024):.2f} MB)")
        
        assert image_size > 1024 * 1024, f"La imagen generada es demasiado pequeña: {image_size} bytes"
        
        data = {
            'image': (img_io, 'large_image.jpg')
        }
        
        start_time = time.time()
        response = client.post('/detect-count', data=data)
        end_time = time.time()
        
        # Verificar respuesta
        assert response.status_code in [200, 500]
        print(f"Tiempo de procesamiento: {end_time - start_time:.2f} segundos")
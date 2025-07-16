from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os
import json

def test_yolo_detection_app():
    print("Iniciando prueba de UI para YOLO Object Detection...")
    
    # Configurar opciones del navegador
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Descomenta para ejecutar sin GUI
    driver = None  
    
    try:
        # Inicializar el driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        driver.maximize_window()
        print("Driver iniciado correctamente.")
        
        # Navegar a la página
        print("Navegando a http://localhost:5000/")
        driver.get("http://localhost:5000/")
        
        # Verificar que la página se cargó correctamente
        print(f"Título de la página: {driver.title}")
        
        # Verificar que el archivo existe antes de subirlo
        file_path = "C:/Users/carlo/Downloads/jugadores.jpg"
        if not os.path.exists(file_path):
            print(f"ADVERTENCIA: El archivo {file_path} no existe.")
            print("Por favor, verifica la ruta del archivo.")
            return
        
        # Esperar a que la página cargue completamente
        wait = WebDriverWait(driver, 15)
        
        print("Buscando elementos del formulario...")
        
        # Buscar el input de archivo - puede tener diferentes selectores
        file_input = None
        possible_selectors = [
            (By.NAME, "image"),
            (By.ID, "image"),
            (By.CSS_SELECTOR, "input[type='file']"),
            (By.XPATH, "//input[@type='file']")
        ]
        
        for selector_type, selector_value in possible_selectors:
            try:
                file_input = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                print(f"Input de archivo encontrado usando: {selector_type} = '{selector_value}'")
                break
            except:
                continue
        
        if not file_input:
            print("ERROR: No se pudo encontrar el input de archivo")
            return
        
        # Buscar el botón de submit
        submit_button = None
        possible_button_selectors = [
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.CSS_SELECTOR, "input[type='submit']"),
            (By.TAG_NAME, "button"),
            (By.XPATH, "//button[contains(text(), 'Detect')]"),
            (By.XPATH, "//button[contains(text(), 'Submit')]"),
            (By.XPATH, "//input[@type='submit']")
        ]
        
        for selector_type, selector_value in possible_button_selectors:
            try:
                submit_button = wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                print(f"Botón encontrado usando: {selector_type} = '{selector_value}'")
                break
            except:
                continue
        
        if not submit_button:
            print("ERROR: No se pudo encontrar el botón de submit")
            return
        
        # Subir imagen
        print(f"Subiendo imagen: {file_path}")
        file_input.send_keys(file_path)
        
        # Hacer clic en el botón
        print("Haciendo clic en el botón de envío...")
        submit_button.click()
        
        # Esperar a que aparezcan los resultados
        print("Esperando resultados de la detección...")
        
        # Buscar elementos que puedan contener los resultados
        result_found = False
        max_wait_time = 30  # segundos
        start_time = time.time()
        
        while not result_found and (time.time() - start_time) < max_wait_time:
            try:
                # Intentar encontrar diferentes elementos que podrían contener resultados
                possible_result_selectors = [
                    (By.ID, "result"),
                    (By.ID, "results"),
                    (By.CLASS_NAME, "results"),
                    (By.CLASS_NAME, "detections"),
                    (By.CSS_SELECTOR, ".detection-results"),
                    (By.XPATH, "//*[contains(@class, 'result')]"),
                    (By.XPATH, "//*[contains(text(), 'Detected')]"),
                    (By.XPATH, "//*[contains(text(), 'Objects')]"),
                    (By.XPATH, "//*[contains(text(), 'Count')]")
                ]
                
                for selector_type, selector_value in possible_result_selectors:
                    try:
                        result_element = driver.find_element(selector_type, selector_value)
                        if result_element.is_displayed():
                            result_text = result_element.text
                            print(f"Resultado encontrado con {selector_type}='{selector_value}': {result_text}")
                            result_found = True
                            break
                    except:
                        continue
                
                if not result_found:
                    # Verificar si hay respuesta JSON en la página
                    page_source = driver.page_source
                    if "countings" in page_source or "detections" in page_source:
                        print("Respuesta JSON detectada en la página")
                        result_found = True
                        break
                    
                    # Verificar si la URL cambió (indicando que se procesó la request)
                    current_url = driver.current_url
                    if "detect-count" in current_url:
                        print("La página fue redirigida a /detect-count")
                        result_found = True
                        break
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error buscando resultados: {e}")
                time.sleep(1)
        
        if result_found:
            print("✓ Prueba exitosa: La aplicación procesó la imagen correctamente.")
            
            # Intentar extraer información específica de los resultados
            try:
                # Buscar elementos específicos que podrían mostrar conteos
                count_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'person') or contains(text(), 'car') or contains(text(), 'bicycle')]")
                if count_elements:
                    print("Objetos detectados:")
                    for element in count_elements[:5]:  # Mostrar solo los primeros 5
                        print(f"  - {element.text}")
                
                # Si la respuesta es JSON, intentar parsearlo
                if "countings" in driver.page_source:
                    print("La aplicación devolvió resultados en formato JSON")
                    
            except Exception as e:
                print(f"Error al extraer detalles: {e}")
        else:
            print("⚠ Advertencia: No se encontraron resultados después de la detección.")
            print("Esto podría indicar:")
            print("  - El procesamiento está tomando más tiempo del esperado")
            print("  - Hay un error en el modelo YOLO")
            print("  - La imagen no contiene objetos detectables")
            print("  - Problema con el endpoint /detect-count")
        
        # Información adicional de debugging
        print(f"URL actual: {driver.current_url}")
        print(f"Título actual: {driver.title}")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Cerrando el navegador...")
        if driver is not None:
            try:
                driver.quit()
            except Exception as e:
                print(f"Error al cerrar el navegador: {e}")
        else:
            print("El navegador no fue inicializado.")
        print("Prueba finalizada.")


def test_api_endpoint_directly():
    """Prueba adicional: verificar que el endpoint /detect-count funciona"""
    print("\n" + "="*50)
    print("Prueba adicional: Verificando endpoint /detect-count")
    
    try:
        import requests
        
        # Verificar que el servidor esté corriendo
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            print("✓ Servidor Flask está corriendo correctamente")
        else:
            print(f"⚠ Servidor responde con código: {response.status_code}")
            
        # Intentar hacer una petición POST al endpoint
        file_path = "C:/Users/carlo/Downloads/jugadores.jpg"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                files = {'image': f}
                response = requests.post("http://localhost:5000/detect-count", files=files)
                
            if response.status_code == 200:
                print("✓ Endpoint /detect-count responde correctamente")
                try:
                    data = response.json()
                    print(f"Conteos encontrados: {data.get('countings', 'N/A')}")
                    print(f"Detecciones: {len(data.get('detections', []))}")
                except:
                    print("Respuesta no es JSON válido")
            else:
                print(f"⚠ Endpoint responde con código: {response.status_code}")
                print(f"Respuesta: {response.text}")
                
    except ImportError:
        print("requests no está instalado. Instálalo con: pip install requests")
    except Exception as e:
        print(f"Error en prueba API: {e}")

# Ejecutar las pruebas
if __name__ == "__main__":
    test_yolo_detection_app()
    test_api_endpoint_directly()
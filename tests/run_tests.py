#!/usr/bin/env python
"""
Script para ejecutar diferentes tipos de pruebas
"""
import subprocess
import sys
import os
import argparse

def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Error: Comando falló con código {result.returncode}")
            return False
        else:
            print("✅ Completado exitosamente")
            return True
            
    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Ejecutar pruebas del proyecto YOLO')
    parser.add_argument('--type', choices=['unit', 'integration', 'all', 'fast', 'slow'], 
                       default='all', help='Tipo de pruebas a ejecutar')
    parser.add_argument('--coverage', action='store_true', help='Generar reporte de cobertura')
    parser.add_argument('--verbose', '-v', action='store_true', help='Salida detallada')
    parser.add_argument('--file', help='Ejecutar archivo específico de pruebas')
    parser.add_argument('--function', help='Ejecutar función específica')
    
    args = parser.parse_args()
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('application.py'):
        print("❌ Error: No se encontró application.py. Ejecuta desde el directorio raíz del proyecto.")
        sys.exit(1)
    
    # Comandos base
    base_cmd = "python -m pytest"
    
    if args.verbose:
        base_cmd += " -v"
    
    if args.coverage:
        base_cmd += " --cov=yolocounter --cov-report=html --cov-report=term-missing"
    
    success = True
    
    if args.file:
        # Ejecutar archivo específico
        cmd = f"{base_cmd} {args.file}"
        if args.function:
            cmd += f"::{args.function}"
        success = run_command(cmd, f"Ejecutando pruebas en {args.file}")
    
    elif args.type == 'unit':
        # Ejecutar solo pruebas unitarias
        cmd = f"{base_cmd} -m unit"
        success = run_command(cmd, "Ejecutando pruebas unitarias")
    
    elif args.type == 'integration':
        # Ejecutar solo pruebas de integración
        cmd = f"{base_cmd} -m integration"
        success = run_command(cmd, "Ejecutando pruebas de integración")
    
    elif args.type == 'fast':
        # Ejecutar pruebas rápidas (excluir lentas)
        cmd = f"{base_cmd} -m 'not slow'"
        success = run_command(cmd, "Ejecutando pruebas rápidas")
    
    elif args.type == 'slow':
        # Ejecutar solo pruebas lentas
        cmd = f"{base_cmd} -m slow"
        success = run_command(cmd, "Ejecutando pruebas lentas")
    
    elif args.type == 'all':
        # Ejecutar todas las pruebas en secuencia
        test_commands = [
            (f"{base_cmd} test_app.py", "Pruebas básicas de aplicación"),
            (f"{base_cmd} -m unit", "Pruebas unitarias"),
            (f"{base_cmd} -m integration", "Pruebas de integración"),
        ]
        
        for cmd, description in test_commands:
            if not run_command(cmd, description):
                success = False
                break
    
    # Ejecutar linting si todas las pruebas pasaron
    if success and args.type == 'all':
        print(f"\n{'='*60}")
        print("🔍 Ejecutando análisis de código...")
        print(f"{'='*60}")
        
        # Flake8 (si está disponible)
        run_command("python -m flake8 yolocounter/ --max-line-length=88 --exclude=migrations", 
                   "Verificando estilo de código con flake8")
        
        # Black (si está disponible)
        run_command("python -m black --check yolocounter/", 
                   "Verificando formato de código con black")
    
    if success:
        print(f"\n{'='*60}")
        print("🎉 ¡Todas las pruebas completadas exitosamente!")
        print(f"{'='*60}")
        
        if args.coverage:
            print("\n📊 Reporte de cobertura generado en htmlcov/index.html")
    else:
        print(f"\n{'='*60}")
        print("❌ Algunas pruebas fallaron")
        print(f"{'='*60}")
        sys.exit(1)

if __name__ == "__main__":
    main()
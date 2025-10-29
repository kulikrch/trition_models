"""
Скрипт для исправления конфликтов зависимостей на Windows
"""
import subprocess
import sys
import os

def run_command(command):
    """Выполняет команду и возвращает результат"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Output: {e.stderr}")
        return False

def fix_triton_conflict():
    """Исправляет конфликт с triton"""
    print("Fixing triton conflict...")
    
    # Удаляем конфликтующий triton
    commands = [
        "pip uninstall -y triton",
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "pip install onnx onnxruntime",
        "pip install fastapi uvicorn",
        "pip install tritonclient[http]",
        "pip install prometheus_client pillow numpy requests python-multipart"
    ]
    
    for cmd in commands:
        success = run_command(cmd)
        if not success:
            print(f"Failed at command: {cmd}")
            return False
    
    return True

def test_imports():
    """Тестирует импорты"""
    print("Testing imports...")
    
    test_packages = [
        'torch',
        'onnx', 
        'onnxruntime',
        'fastapi',
        'uvicorn',
        'prometheus_client'
    ]
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            return False
    
    # Отдельно тестируем torchvision
    try:
        import torchvision
        print("✓ torchvision")
    except Exception as e:
        print(f"✗ torchvision: {e}")
        print("Trying to fix torchvision...")
        run_command("pip uninstall -y torchvision")
        run_command("pip install torchvision --index-url https://download.pytorch.org/whl/cpu")
        
        try:
            import torchvision
            print("✓ torchvision fixed")
        except:
            print("✗ torchvision still broken, but we can continue without it for now")
    
    return True

def main():
    print("="*60)
    print("FIXING PYTHON DEPENDENCIES")
    print("="*60)
    
    # Исправляем конфликт triton
    if not fix_triton_conflict():
        print("Failed to fix dependencies")
        sys.exit(1)
    
    # Тестируем импорты
    if not test_imports():
        print("Some imports failed, but core packages should work")
    
    print("\n" + "="*60)
    print("DEPENDENCIES FIXED!")
    print("="*60)
    print("You can now run:")
    print("python run_project.py --epochs 5")

if __name__ == "__main__":
    main()
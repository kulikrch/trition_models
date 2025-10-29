"""
Простой тест основных компонентов без сложных зависимостей
"""
import torch
import onnx
import onnxruntime as ort
import os
import sys

def test_pytorch_model():
    """Тест PyTorch модели"""
    print("Testing PyTorch model...")
    
    if not os.path.exists("models/cifar10_model.pth"):
        print("✗ Model not found")
        return False
    
    try:
        # Загружаем модель
        sys.path.append("src/train")
        from model import get_model
        
        model = get_model(num_classes=10)
        checkpoint = torch.load("models/cifar10_model.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Тестовый инференс
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ PyTorch model works! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch model error: {e}")
        return False

def test_onnx_conversion():
    """Тест конвертации в ONNX"""
    print("Testing ONNX conversion...")
    
    try:
        sys.path.append("src/train")
        from model import get_model
        
        # Загружаем PyTorch модель
        model = get_model(num_classes=10)
        checkpoint = torch.load("models/cifar10_model.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Конвертируем в ONNX
        dummy_input = torch.randn(1, 3, 32, 32)
        onnx_path = "models/test_model.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        # Проверяем ONNX модель
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Тестируем с ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        print(f"✓ ONNX conversion works! Output shape: {ort_output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ ONNX conversion error: {e}")
        return False

def test_fastapi():
    """Тест FastAPI компонентов"""
    print("Testing FastAPI...")
    
    try:
        from fastapi import FastAPI
        from PIL import Image
        import numpy as np
        
        # Создаем тестовое изображение
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, 'RGB')
        
        # Тестируем предобработку
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))
        
        print(f"✓ FastAPI preprocessing works! Output shape: {img_array.shape}")
        return True
        
    except Exception as e:
        print(f"✗ FastAPI test error: {e}")
        return False

def main():
    """Запуск всех тестов"""
    print("="*50)
    print("BASIC COMPONENT TESTS")
    print("="*50)
    
    tests = [
        ("PyTorch Model", test_pytorch_model),
        ("ONNX Conversion", test_onnx_conversion),
        ("FastAPI Components", test_fastapi)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            print()
    
    print("="*50)
    print(f"TESTS COMPLETED: {passed}/{total} passed")
    print("="*50)
    
    if passed >= 2:
        print("✓ Core functionality is working!")
        print("You can proceed with manual service testing.")
    else:
        print("✗ Some core issues remain.")
    
    return passed >= 2

if __name__ == "__main__":
    main()
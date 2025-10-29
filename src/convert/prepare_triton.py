"""
Скрипт для подготовки моделей для Triton Inference Server
"""
import torch
import os
import shutil
import sys

# Добавляем путь к модели
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import get_model


class TritonModelPreparer:
    def __init__(self, model_repository_path='../../triton/model_repository'):
        self.model_repository_path = model_repository_path
        
    def prepare_pytorch_model(self, pytorch_model_path, model_name='cifar10_pytorch'):
        """Подготавливает PyTorch модель для Triton"""
        print(f"Preparing PyTorch model for Triton: {model_name}")
        
        # Путь к модели в репозитории Triton
        triton_model_path = os.path.join(self.model_repository_path, model_name, '1')
        os.makedirs(triton_model_path, exist_ok=True)
        
        # Загружаем PyTorch модель
        model = get_model(num_classes=10)
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Создаем traced модель для TorchScript
        dummy_input = torch.randn(1, 3, 32, 32)
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Сохраняем traced модель
        model_file_path = os.path.join(triton_model_path, 'model.pt')
        traced_model.save(model_file_path)
        
        print(f"PyTorch model saved to: {model_file_path}")
        return model_file_path
    
    def prepare_onnx_model(self, onnx_model_path, model_name='cifar10_onnx'):
        """Подготавливает ONNX модель для Triton"""
        print(f"Preparing ONNX model for Triton: {model_name}")
        
        # Путь к модели в репозитории Triton
        triton_model_path = os.path.join(self.model_repository_path, model_name, '1')
        os.makedirs(triton_model_path, exist_ok=True)
        
        # Копируем ONNX модель
        model_file_path = os.path.join(triton_model_path, 'model.onnx')
        shutil.copy2(onnx_model_path, model_file_path)
        
        print(f"ONNX model copied to: {model_file_path}")
        return model_file_path
    
    def verify_model_repository(self):
        """Проверяет структуру репозитория моделей"""
        print("Verifying model repository structure...")
        
        required_models = ['cifar10_pytorch', 'cifar10_onnx']
        
        for model_name in required_models:
            model_path = os.path.join(self.model_repository_path, model_name)
            config_path = os.path.join(model_path, 'config.pbtxt')
            version_path = os.path.join(model_path, '1')
            
            if not os.path.exists(model_path):
                print(f"✗ Model directory missing: {model_path}")
                continue
                
            if not os.path.exists(config_path):
                print(f"✗ Config file missing: {config_path}")
                continue
                
            if not os.path.exists(version_path):
                print(f"✗ Version directory missing: {version_path}")
                continue
            
            # Проверяем наличие файла модели
            if model_name == 'cifar10_pytorch':
                model_file = os.path.join(version_path, 'model.pt')
            else:
                model_file = os.path.join(version_path, 'model.onnx')
            
            if os.path.exists(model_file):
                print(f"✓ {model_name} ready for Triton")
            else:
                print(f"✗ Model file missing: {model_file}")
        
        print("Model repository verification completed")
    
    def prepare_all_models(self):
        """Подготавливает все модели для Triton"""
        pytorch_model_path = '../../models/cifar10_model.pth'
        onnx_model_path = '../../models/cifar10_model.onnx'
        
        print("="*60)
        print("PREPARING MODELS FOR TRITON INFERENCE SERVER")
        print("="*60)
        
        # Проверяем наличие исходных моделей
        if not os.path.exists(pytorch_model_path):
            print(f"Error: PyTorch model not found at {pytorch_model_path}")
            print("Please train the model first using src/train/train.py")
            return False
        
        if not os.path.exists(onnx_model_path):
            print(f"Error: ONNX model not found at {onnx_model_path}")
            print("Please convert the model first using src/convert/to_onnx.py")
            return False
        
        # Подготавливаем PyTorch модель
        try:
            self.prepare_pytorch_model(pytorch_model_path)
        except Exception as e:
            print(f"Error preparing PyTorch model: {e}")
            return False
        
        # Подготавливаем ONNX модель
        try:
            self.prepare_onnx_model(onnx_model_path)
        except Exception as e:
            print(f"Error preparing ONNX model: {e}")
            return False
        
        # Проверяем репозиторий
        self.verify_model_repository()
        
        print("\n" + "="*60)
        print("MODEL PREPARATION COMPLETED")
        print("="*60)
        print("Models are ready for Triton Inference Server!")
        print("You can now start Triton with:")
        print("docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \\")
        print("  -v $(pwd)/triton/model_repository:/models \\")
        print("  nvcr.io/nvidia/tritonserver:23.10-py3 \\")
        print("  tritonserver --model-repository=/models")
        
        return True


def main():
    """Основная функция"""
    preparer = TritonModelPreparer()
    success = preparer.prepare_all_models()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
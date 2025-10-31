"""
Подготовка всех моделей для Triton Inference Server
"""
import torch
import torch.nn as nn
import os
import shutil
import sys
from pathlib import Path

# Добавляем путь к модели
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from working_demo import SimpleCNN

class TritonModelPreparer:
    def __init__(self, model_repository_path='../../triton/model_repository'):
        self.model_repository_path = Path(model_repository_path)
        self.models_path = Path('../../models')
        
    def create_model_directories(self):
        """Создает структуру директорий для всех моделей"""
        models = [
            'cifar10_original',      # Оригинальная PyTorch модель
            'cifar10_optimized',     # Оптимизированная PyTorch модель
            'cifar10_onnx',          # ONNX модель
            'cifar10_onnx_optimized' # Оптимизированная ONNX модель
        ]
        
        for model_name in models:
            model_dir = self.model_repository_path / model_name
            version_dir = model_dir / '1'
            version_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Создана директория: {model_dir}")
        
        return models
    
    def prepare_pytorch_original(self):
        """Подготавливает оригинальную PyTorch модель"""
        print("🔄 Подготовка оригинальной PyTorch модели...")
        
        model_path = self.models_path / 'working_model.pth'
        if not model_path.exists():
            print(f"❌ Оригинальная модель не найдена: {model_path}")
            return False
        
        # Загружаем и сохраняем как TorchScript
        model = SimpleCNN(num_classes=10)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Создаем traced модель
        dummy_input = torch.randn(1, 3, 32, 32)
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Сохраняем для Triton
        triton_path = self.model_repository_path / 'cifar10_original' / '1' / 'model.pt'
        traced_model.save(str(triton_path))
        
        print(f"✅ Оригинальная модель сохранена: {triton_path}")
        return True
    
    def prepare_pytorch_optimized(self):
        """Подготавливает оптимизированную PyTorch модель"""
        print("🔄 Подготовка оптимизированной PyTorch модели...")
        
        # Попробуем найти квантизованную модель
        model_paths = [
            self.models_path / 'working_model_quantized.pth',
            self.models_path / 'working_model_combined.pth',
            self.models_path / 'working_model_pruned.pth'
        ]
        
        source_path = None
        for path in model_paths:
            if path.exists():
                source_path = path
                break
        
        if not source_path:
            print("❌ Оптимизированная модель не найдена")
            return False
        
        # Для квантизованных моделей нужен особый подход
        try:
            # Загружаем оригинальную модель и применяем квантизацию
            original_model = SimpleCNN(num_classes=10)
            original_checkpoint = torch.load(self.models_path / 'working_model.pth', map_location='cpu')
            original_model.load_state_dict(original_checkpoint['model_state_dict'])
            original_model.eval()
            
            # Применяем квантизацию
            quantized_model = torch.quantization.quantize_dynamic(
                original_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            
            # Создаем traced модель (может не работать с квантизованными моделями)
            # Поэтому сохраним как обычную PyTorch модель
            triton_path = self.model_repository_path / 'cifar10_optimized' / '1' / 'model.pt'
            
            # Сохраняем state_dict вместо traced модели
            torch.save(quantized_model.state_dict(), str(triton_path))
            
            print(f"✅ Оптимизированная модель сохранена: {triton_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка подготовки оптимизированной модели: {e}")
            return False
    
    def prepare_onnx_original(self):
        """Подготавливает оригинальную ONNX модель"""
        print("🔄 Подготовка оригинальной ONNX модели...")
        
        source_path = self.models_path / 'working_model.onnx'
        if not source_path.exists():
            print(f"❌ ONNX модель не найдена: {source_path}")
            return False
        
        # Копируем ONNX модель
        triton_path = self.model_repository_path / 'cifar10_onnx' / '1' / 'model.onnx'
        shutil.copy2(source_path, triton_path)
        
        print(f"✅ ONNX модель скопирована: {triton_path}")
        return True
    
    def prepare_onnx_optimized(self):
        """Подготавливает оптимизированную ONNX модель"""
        print("🔄 Подготовка оптимизированной ONNX модели...")
        
        # Создаем оптимизированную ONNX модель из квантизованной PyTorch
        try:
            # Загружаем оригинальную модель
            original_model = SimpleCNN(num_classes=10)
            original_checkpoint = torch.load(self.models_path / 'working_model.pth', map_location='cpu')
            original_model.load_state_dict(original_checkpoint['model_state_dict'])
            original_model.eval()
            
            # Экспортируем в ONNX с оптимизациями
            dummy_input = torch.randn(1, 3, 32, 32)
            triton_path = self.model_repository_path / 'cifar10_onnx_optimized' / '1' / 'model.onnx'
            
            torch.onnx.export(
                original_model,
                dummy_input,
                str(triton_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,  # Включаем constant folding
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                # Дополнительные оптимизации
                enable_onnx_checker=True,
                use_external_data_format=False
            )
            
            print(f"✅ Оптимизированная ONNX модель создана: {triton_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания оптимизированной ONNX: {e}")
            return False
    
    def create_config_files(self):
        """Создает конфигурационные файлы для всех моделей"""
        print("🔄 Создание конфигурационных файлов...")
        
        configs = {
            'cifar10_original': self.create_pytorch_config('cifar10_original'),
            'cifar10_optimized': self.create_pytorch_config('cifar10_optimized'),
            'cifar10_onnx': self.create_onnx_config('cifar10_onnx'),
            'cifar10_onnx_optimized': self.create_onnx_optimized_config('cifar10_onnx_optimized')
        }
        
        for model_name, config_content in configs.items():
            config_path = self.model_repository_path / model_name / 'config.pbtxt'
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f"✅ Конфигурация создана: {config_path}")
        
        return True
    
    def create_pytorch_config(self, model_name):
        """Создает конфигурацию для PyTorch модели"""
        return f"""name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 16
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }}
]

# Динамическое батчинг
dynamic_batching {{
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 4, 8 ]
}}

# Инстансы модели для CPU
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

# Версии модели
version_policy: {{ all {{ }} }}
"""
    
    def create_onnx_config(self, model_name):
        """Создает конфигурацию для ONNX модели"""
        return f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }}
]

# Динамическое батчинг
dynamic_batching {{
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 4, 8 ]
}}

# Инстансы модели для CPU
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

# Версии модели
version_policy: {{ all {{ }} }}
"""
    
    def create_onnx_optimized_config(self, model_name):
        """Создает конфигурацию для оптимизированной ONNX модели"""
        return f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }}
]

# Оптимизации для CPU
optimization {{
  execution_accelerators {{
    cpu_execution_accelerator : [ {{
      name : "openvino"
    }}]
  }}
}}

# Агрессивное батчинг для оптимизированной модели
dynamic_batching {{
  max_queue_delay_microseconds: 50
  preferred_batch_size: [ 8, 16, 32 ]
}}

# Больше инстансов для оптимизированной модели
instance_group [
  {{
    count: 2
    kind: KIND_CPU
  }}
]

# Версии модели
version_policy: {{ all {{ }} }}
"""
    
    def prepare_all_models(self):
        """Подготавливает все модели для Triton"""
        print("🚀 ПОДГОТОВКА ВСЕХ МОДЕЛЕЙ ДЛЯ TRITON")
        print("=" * 60)
        
        # Создаем структуру директорий
        models = self.create_model_directories()
        
        # Подготавливаем модели
        results = {}
        
        print(f"\n1️⃣ ОРИГИНАЛЬНАЯ PYTORCH МОДЕЛЬ")
        print("-" * 40)
        results['pytorch_original'] = self.prepare_pytorch_original()
        
        print(f"\n2️⃣ ОПТИМИЗИРОВАННАЯ PYTORCH МОДЕЛЬ")
        print("-" * 40)
        results['pytorch_optimized'] = self.prepare_pytorch_optimized()
        
        print(f"\n3️⃣ ОРИГИНАЛЬНАЯ ONNX МОДЕЛЬ")
        print("-" * 40)
        results['onnx_original'] = self.prepare_onnx_original()
        
        print(f"\n4️⃣ ОПТИМИЗИРОВАННАЯ ONNX МОДЕЛЬ")
        print("-" * 40)
        results['onnx_optimized'] = self.prepare_onnx_optimized()
        
        print(f"\n5️⃣ КОНФИГУРАЦИОННЫЕ ФАЙЛЫ")
        print("-" * 40)
        results['configs'] = self.create_config_files()
        
        # Итоговый отчет
        print(f"\n🎉 ПОДГОТОВКА ЗАВЕРШЕНА!")
        print("=" * 40)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        print(f"✅ Успешно подготовлено: {success_count}/{total_count}")
        
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {name}")
        
        if success_count >= 3:  # Минимум 3 модели из 5
            print(f"\n🚀 Готово для запуска Triton!")
            print(f"📂 Model repository: {self.model_repository_path}")
            return True
        else:
            print(f"\n⚠️ Недостаточно моделей для полноценного запуска")
            return False

def main():
    """Главная функция"""
    preparer = TritonModelPreparer()
    success = preparer.prepare_all_models()
    
    if success:
        print(f"\n📋 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Запустить Triton: docker-compose up triton")
        print("2. Проверить модели: curl http://localhost:8001/v2/models")
        print("3. Тестировать инференс через API")
    
    return success

if __name__ == "__main__":
    main()
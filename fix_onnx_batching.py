"""
Исправление ONNX модели для поддержки dynamic batching в Triton
"""
import torch
import onnx
from onnx import version_converter
import os
from working_demo import SimpleCNN

def recreate_onnx_with_dynamic_batching():
    """Пересоздает ONNX модель с dynamic batching"""
    print("🔄 Пересоздание ONNX модели с dynamic batching...")
    
    # Загружаем PyTorch модель
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load("models/working_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Экспортируем с dynamic axes для batch dimension
    onnx_path = "triton/model_repository/cifar10_onnx/1/model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},    # Делаем batch размерность динамической
            'output': {0: 'batch_size'}    # Делаем batch размерность динамической
        }
    )
    
    print(f"✅ ONNX модель пересоздана: {onnx_path}")
    
    # Проверяем модель
    onnx_model = onnx.load(onnx_path)
    
    # Анализируем входы
    for inp in onnx_model.graph.input:
        print(f"   Input: {inp.name}")
        for i, dim in enumerate(inp.type.tensor_type.shape.dim):
            if dim.dim_param:
                print(f"      Dim {i}: {dim.dim_param} (dynamic)")
            else:
                print(f"      Dim {i}: {dim.dim_value} (fixed)")
    
    return True

def fix_onnx_optimized():
    """Исправляет оптимизированную ONNX модель"""
    print("\n🔄 Исправление оптимизированной ONNX модели...")
    
    # Загружаем PyTorch модель
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load("models/working_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Экспортируем оптимизированную версию
    onnx_opt_path = "triton/model_repository/cifar10_onnx_optimized/1/model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_opt_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,  # Больше оптимизаций
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Оптимизированная ONNX модель пересоздана: {onnx_opt_path}")
    return True

def fix_onnx_optimized_config():
    """Исправляет синтаксическую ошибку в конфиге"""
    print("\n🔧 Исправление конфигурации cifar10_onnx_optimized...")
    
    config_content = '''name: "cifar10_onnx_optimized"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

optimization {
  execution_accelerators {
    cpu_execution_accelerator : [ {
      name : "openvino"
    }]
  }
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 50
  preferred_batch_size: [ 4, 8 ]
}

version_policy: { all { }}
'''
    
    config_path = "triton/model_repository/cifar10_onnx_optimized/config.pbtxt"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Конфигурация исправлена: {config_path}")
    return True

def main():
    """Главная функция"""
    print("🔧 ИСПРАВЛЕНИЕ ONNX МОДЕЛЕЙ ДЛЯ TRITON")
    print("=" * 50)
    
    # Проверяем, что директории существуют
    os.makedirs("triton/model_repository/cifar10_onnx/1", exist_ok=True)
    os.makedirs("triton/model_repository/cifar10_onnx_optimized/1", exist_ok=True)
    
    success = True
    
    # Исправляем базовую ONNX модель
    if not recreate_onnx_with_dynamic_batching():
        success = False
    
    # Исправляем оптимизированную ONNX модель
    if not fix_onnx_optimized():
        success = False
    
    # Исправляем конфигурацию
    if not fix_onnx_optimized_config():
        success = False
    
    if success:
        print(f"\n🎉 ВСЕ ONNX МОДЕЛИ ИСПРАВЛЕНЫ!")
        print("=" * 40)
        print("✅ cifar10_onnx - dynamic batching включен")
        print("✅ cifar10_onnx_optimized - dynamic batching + OpenVINO")
        print("✅ Конфигурации исправлены")
        
        print(f"\n🚀 Следующий шаг:")
        print("docker-compose -f docker-compose-triton.yml restart triton")
        
    return success

if __name__ == "__main__":
    main()
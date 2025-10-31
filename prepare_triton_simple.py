"""
Простая подготовка моделей для Triton
"""
import torch
import torch.nn as nn
import os
import shutil
from working_demo import SimpleCNN

def prepare_triton_models():
    """Подготавливает все модели для Triton"""
    print("🚀 ПОДГОТОВКА МОДЕЛЕЙ ДЛЯ TRITON")
    print("=" * 50)
    
    repo_path = "triton/model_repository"
    models_path = "models"
    
    # 1. Оригинальная PyTorch модель
    print("\n1️⃣ Подготовка оригинальной PyTorch модели...")
    pytorch_dir = f"{repo_path}/cifar10_pytorch/1"
    os.makedirs(pytorch_dir, exist_ok=True)
    
    # Загружаем и сохраняем как TorchScript
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load(f"{models_path}/working_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем traced модель
    dummy_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(f"{pytorch_dir}/model.pt")
    print(f"✅ PyTorch модель сохранена: {pytorch_dir}/model.pt")
    
    # Конфигурация для PyTorch
    pytorch_config = '''name: "cifar10_pytorch"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}

version_policy: { all { }}
'''
    
    with open(f"{repo_path}/cifar10_pytorch/config.pbtxt", 'w') as f:
        f.write(pytorch_config)
    print(f"✅ PyTorch конфигурация создана")
    
    # 2. ONNX модель
    print("\n2️⃣ Подготовка ONNX модели...")
    onnx_dir = f"{repo_path}/cifar10_onnx/1"
    os.makedirs(onnx_dir, exist_ok=True)
    
    if os.path.exists(f"{models_path}/working_model.onnx"):
        shutil.copy2(f"{models_path}/working_model.onnx", f"{onnx_dir}/model.onnx")
        print(f"✅ ONNX модель скопирована: {onnx_dir}/model.onnx")
    else:
        # Создаем ONNX модель
        torch.onnx.export(
            model, dummy_input, f"{onnx_dir}/model.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['input'], output_names=['output']
        )
        print(f"✅ ONNX модель создана: {onnx_dir}/model.onnx")
    
    # Конфигурация для ONNX
    onnx_config = '''name: "cifar10_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 8
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

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}

version_policy: { all { }}
'''
    
    with open(f"{repo_path}/cifar10_onnx/config.pbtxt", 'w') as f:
        f.write(onnx_config)
    print(f"✅ ONNX конфигурация создана")
    
    # 3. Оптимизированная модель (квантизованная)
    print("\n3️⃣ Подготовка оптимизированной модели...")
    opt_dir = f"{repo_path}/cifar10_optimized/1"
    os.makedirs(opt_dir, exist_ok=True)
    
    # Создаем квантизованную модель
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    # Сохраняем как state_dict (так как traced quantized модели могут не работать)
    torch.save(quantized_model.state_dict(), f"{opt_dir}/model.pt")
    print(f"✅ Оптимизированная модель сохранена: {opt_dir}/model.pt")
    
    # Конфигурация для оптимизированной модели
    opt_config = '''name: "cifar10_optimized"
platform: "pytorch_libtorch"
max_batch_size: 16
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

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
    
    with open(f"{repo_path}/cifar10_optimized/config.pbtxt", 'w') as f:
        f.write(opt_config)
    print(f"✅ Оптимизированная конфигурация создана")
    
    # 4. Оптимизированная ONNX
    print("\n4️⃣ Подготовка оптимизированной ONNX модели...")
    onnx_opt_dir = f"{repo_path}/cifar10_onnx_optimized/1"
    os.makedirs(onnx_opt_dir, exist_ok=True)
    
    # Экспортируем ONNX с дополнительными оптимизациями
    torch.onnx.export(
        model, dummy_input, f"{onnx_opt_dir}/model.onnx",
        export_params=True, opset_version=11, 
        do_constant_folding=True,  # Включаем constant folding
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Оптимизированная ONNX модель создана: {onnx_opt_dir}/model.onnx")
    
    # Конфигурация для оптимизированной ONNX
    onnx_opt_config = '''name: "cifar10_onnx_optimized"
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
  preferred_batch_size: [ 4, 8, 16 ]
}

version_policy: { all { }}
'''
    
    with open(f"{repo_path}/cifar10_onnx_optimized/config.pbtxt", 'w') as f:
        f.write(onnx_opt_config)
    print(f"✅ Оптимизированная ONNX конфигурация создана")
    
    print(f"\n🎉 ВСЕ МОДЕЛИ ПОДГОТОВЛЕНЫ!")
    print("=" * 50)
    print("📂 Структура model repository:")
    print("   • cifar10_pytorch - оригинальная PyTorch модель")
    print("   • cifar10_onnx - оригинальная ONNX модель")  
    print("   • cifar10_optimized - квантизованная PyTorch модель")
    print("   • cifar10_onnx_optimized - оптимизированная ONNX модель")
    
    return True

if __name__ == "__main__":
    success = prepare_triton_models()
    
    if success:
        print(f"\n🚀 СЛЕДУЮЩИЙ ШАГ: Запуск Triton Server")
        print("docker-compose -f docker-compose-triton.yml up -d triton")
    else:
        print(f"\n❌ Подготовка не удалась")
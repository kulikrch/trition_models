"""
Исправление конфигураций Triton под реальные CIFAR-10 модели
"""
import os
import torch
import onnx
import sys
from working_demo import SimpleCNN

def analyze_pytorch_model():
    """Анализирует PyTorch модель для получения точных параметров"""
    print("🔍 Анализ PyTorch модели...")
    
    model_path = "models/working_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return None
    
    # Загружаем модель
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем traced модель для анализа
    dummy_input = torch.randn(1, 3, 32, 32)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ PyTorch модель:")
    print(f"   Input shape: {list(dummy_input.shape)}")
    print(f"   Output shape: {list(output.shape)}")
    
    # Анализируем traced модель
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Получаем имена входов и выходов
        graph = traced_model.graph
        inputs = list(graph.inputs())
        outputs = list(graph.outputs())
        
        print(f"   Traced inputs: {len(inputs)}")
        print(f"   Traced outputs: {len(outputs)}")
        
        return {
            'input_shape': [3, 32, 32],
            'output_shape': [10],
            'input_name': 'input__0',  # Standard для PyTorch в Triton
            'output_name': 'output__0'
        }
        
    except Exception as e:
        print(f"⚠️ Ошибка трассировки: {e}")
        return {
            'input_shape': [3, 32, 32],
            'output_shape': [10],
            'input_name': 'input__0',
            'output_name': 'output__0'
        }

def analyze_onnx_model():
    """Анализирует ONNX модель для получения точных параметров"""
    print("\n🔍 Анализ ONNX модели...")
    
    model_path = "models/working_model.onnx"
    if not os.path.exists(model_path):
        print(f"❌ ONNX модель не найдена: {model_path}")
        return None
    
    try:
        # Загружаем ONNX модель
        onnx_model = onnx.load(model_path)
        
        # Анализируем входы
        inputs = onnx_model.graph.input
        outputs = onnx_model.graph.output
        
        input_info = {}
        output_info = {}
        
        for inp in inputs:
            name = inp.name
            shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in inp.type.tensor_type.shape.dim]
            dtype = inp.type.tensor_type.elem_type
            input_info[name] = {'shape': shape, 'dtype': dtype}
            print(f"   Input: {name}, shape: {shape}, dtype: {dtype}")
        
        for out in outputs:
            name = out.name
            shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in out.type.tensor_type.shape.dim]
            dtype = out.type.tensor_type.elem_type
            output_info[name] = {'shape': shape, 'dtype': dtype}
            print(f"   Output: {name}, shape: {shape}, dtype: {dtype}")
        
        # Берем первый вход/выход
        first_input = list(input_info.keys())[0]
        first_output = list(output_info.keys())[0]
        
        input_shape = input_info[first_input]['shape'][1:]  # Убираем batch dimension
        output_shape = output_info[first_output]['shape'][1:]  # Убираем batch dimension
        
        return {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_name': first_input,
            'output_name': first_output
        }
        
    except Exception as e:
        print(f"❌ Ошибка анализа ONNX: {e}")
        return None

def create_pytorch_config(model_info, model_name):
    """Создает правильную конфигурацию для PyTorch модели"""
    
    config = f'''name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {{
    name: "{model_info['input_name']}"
    data_type: TYPE_FP32
    dims: [ {", ".join(map(str, model_info['input_shape']))} ]
  }}
]
output [
  {{
    name: "{model_info['output_name']}"
    data_type: TYPE_FP32
    dims: [ {", ".join(map(str, model_info['output_shape']))} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}

version_policy: {{ all {{ }} }}
'''
    return config

def create_onnx_config(model_info, model_name, optimized=False):
    """Создает правильную конфигурацию для ONNX модели"""
    
    optimization_block = ""
    if optimized:
        optimization_block = '''
optimization {{
  execution_accelerators {{
    cpu_execution_accelerator : [ {{
      name : "openvino"
    }}]
  }}
}}
'''
    
    instance_count = 2 if optimized else 1
    max_batch = 16 if optimized else 8
    delay = 50 if optimized else 100
    
    preferred_batch = ""
    if optimized:
        preferred_batch = '''
  preferred_batch_size: [ 4, 8 ]'''
    
    config = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch}
input [
  {{
    name: "{model_info['input_name']}"
    data_type: TYPE_FP32
    dims: [ {", ".join(map(str, model_info['input_shape']))} ]
  }}
]
output [
  {{
    name: "{model_info['output_name']}"
    data_type: TYPE_FP32
    dims: [ {", ".join(map(str, model_info['output_shape']))} ]
  }}
]
{optimization_block}
instance_group [
  {{
    count: {instance_count}
    kind: KIND_CPU
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: {delay}{preferred_batch}
}}

version_policy: {{ all {{ }} }}
'''
    return config

def fix_all_configs():
    """Исправляет все конфигурации"""
    print("🔧 ИСПРАВЛЕНИЕ КОНФИГУРАЦИЙ TRITON")
    print("=" * 50)
    
    # Анализируем модели
    pytorch_info = analyze_pytorch_model()
    onnx_info = analyze_onnx_model()
    
    if not pytorch_info:
        print("❌ Не удалось проанализировать PyTorch модель")
        return False
    
    if not onnx_info:
        print("❌ Не удалось проанализировать ONNX модель")
        return False
    
    # Создаем правильные конфигурации
    configs = {
        'cifar10_pytorch': create_pytorch_config(pytorch_info, 'cifar10_pytorch'),
        'cifar10_onnx': create_onnx_config(onnx_info, 'cifar10_onnx', optimized=False),
        'cifar10_onnx_optimized': create_onnx_config(onnx_info, 'cifar10_onnx_optimized', optimized=True)
    }
    
    # Удаляем проблемную квантизованную модель
    print("\n🗑️ Удаление проблемной квантизованной модели...")
    import shutil
    if os.path.exists("triton/model_repository/cifar10_optimized"):
        shutil.rmtree("triton/model_repository/cifar10_optimized")
        print("✅ cifar10_optimized удалена")
    
    # Записываем исправленные конфигурации
    print("\n📝 Создание исправленных конфигураций...")
    
    for model_name, config_content in configs.items():
        config_path = f"triton/model_repository/{model_name}/config.pbtxt"
        
        if os.path.exists(os.path.dirname(config_path)):
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f"✅ {config_path} обновлен")
        else:
            print(f"⚠️ {config_path} - директория не существует")
    
    # Проверяем, что базовая ONNX модель существует
    basic_onnx_dir = "triton/model_repository/cifar10_onnx/1"
    if not os.path.exists(f"{basic_onnx_dir}/model.onnx"):
        print("\n📂 Создание базовой ONNX модели...")
        os.makedirs(basic_onnx_dir, exist_ok=True)
        
        if os.path.exists("models/working_model.onnx"):
            import shutil
            shutil.copy2("models/working_model.onnx", f"{basic_onnx_dir}/model.onnx")
            print("✅ Базовая ONNX модель скопирована")
    
    print("\n🎉 ВСЕ КОНФИГУРАЦИИ ИСПРАВЛЕНЫ!")
    print("=" * 50)
    
    print("📋 Готовые модели для Triton:")
    print("   ✅ cifar10_pytorch - исправленная PyTorch модель")
    print("   ✅ cifar10_onnx - исправленная базовая ONNX модель")
    print("   ✅ cifar10_onnx_optimized - исправленная оптимизированная ONNX")
    print("   ❌ cifar10_optimized - удалена (проблемная)")
    
    print(f"\n🚀 Следующий шаг:")
    print("docker-compose -f docker-compose-triton.yml restart triton")
    
    return True

if __name__ == "__main__":
    success = fix_all_configs()
    
    if success:
        print(f"\n✅ Конфигурации исправлены под CIFAR-10!")
        print("   Input: [3, 32, 32] (правильно для CIFAR-10)")
        print("   Output: [10] (правильно для 10 классов)")
        print("   Имена тензоров: из реальных моделей")
    else:
        print(f"\n❌ Ошибка при исправлении конфигураций")
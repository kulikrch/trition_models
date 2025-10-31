"""
Создание минимальной рабочей конфигурации Triton
"""
import os
import shutil
import torch
from working_demo import SimpleCNN

def create_minimal_triton():
    """Создает минимальную рабочую конфигурацию"""
    print("🔧 СОЗДАНИЕ МИНИМАЛЬНОЙ КОНФИГУРАЦИИ TRITON")
    print("=" * 50)
    
    # Очищаем и создаем чистую структуру
    repo_path = "triton_minimal/model_repository"
    if os.path.exists("triton_minimal"):
        shutil.rmtree("triton_minimal")
    
    # Создаем только одну простую модель
    model_dir = f"{repo_path}/simple_cifar10/1"
    os.makedirs(model_dir, exist_ok=True)
    
    print("1️⃣ Создание простой PyTorch модели...")
    
    # Загружаем и сохраняем как простую traced модель
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load("models/working_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем traced модель
    dummy_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Сохраняем
    model_path = f"{model_dir}/model.pt"
    traced_model.save(model_path)
    print(f"✅ Модель сохранена: {model_path}")
    
    # Создаем максимально простую конфигурацию
    config_content = '''name: "simple_cifar10"
platform: "pytorch_libtorch"
max_batch_size: 0
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

version_policy: { all { }}
'''
    
    config_path = f"{repo_path}/simple_cifar10/config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Конфигурация создана: {config_path}")
    
    # Создаем простой docker-compose
    docker_compose_content = '''version: '3.8'

services:
  triton-minimal:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    container_name: triton-minimal
    ports:
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./triton_minimal/model_repository:/models
    command: tritonserver --model-repository=/models --log-verbose=1
    restart: unless-stopped
'''
    
    with open("docker-compose-triton-minimal.yml", 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Docker compose создан: docker-compose-triton-minimal.yml")
    
    print(f"\n🎉 МИНИМАЛЬНАЯ КОНФИГУРАЦИЯ ГОТОВА!")
    print("=" * 50)
    print("📂 Структура:")
    print("   triton_minimal/model_repository/")
    print("   └── simple_cifar10/")
    print("       ├── 1/model.pt")
    print("       └── config.pbtxt")
    
    print(f"\n🚀 Запуск:")
    print("1. Остановить старый Triton: docker-compose -f docker-compose-triton.yml down")
    print("2. Запустить новый: docker-compose -f docker-compose-triton-minimal.yml up -d")
    print("3. Проверить: curl http://localhost:8002/v2/models")
    
    return True

if __name__ == "__main__":
    create_minimal_triton()
"""
Главный скрипт для запуска всего ML проекта от начала до конца
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path


class MLProjectRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def run_command(self, command, cwd=None, shell=True):
        """Выполняет команду и возвращает результат"""
        print(f"Running: {command}")
        try:
            if cwd:
                result = subprocess.run(command, shell=shell, cwd=cwd, check=True, 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(command, shell=shell, check=True, 
                                      capture_output=True, text=True)
            print(f"✓ Command completed successfully")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed: {e}")
            print(f"Error output: {e.stderr}")
            return False, e.stderr
    
    def check_dependencies(self):
        """Проверяет наличие необходимых зависимостей"""
        print("Checking dependencies...")
        
        # Проверяем Python пакеты
        required_packages = [
            'torch', 'torchvision', 'onnx', 'onnxruntime', 
            'fastapi', 'uvicorn', 'tritonclient', 'prometheus_client'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package} is missing")
        
        if missing_packages:
            print(f"Installing missing packages: {missing_packages}")
            success, _ = self.run_command(f"pip install {' '.join(missing_packages)}")
            if not success:
                print("Failed to install dependencies")
                return False
        
        # Проверяем Docker
        success, _ = self.run_command("docker --version")
        if not success:
            print("✗ Docker is not installed or not running")
            return False
        print("✓ Docker is available")
        
        # Проверяем Docker Compose
        success, _ = self.run_command("docker-compose --version")
        if not success:
            print("✗ Docker Compose is not installed")
            return False
        print("✓ Docker Compose is available")
        
        return True
    
    def train_model(self, epochs=10):  # Уменьшаем по умолчанию для CPU
        """Обучает модель"""
        print(f"\n{'='*60}")
        print("STEP 1: TRAINING MODEL")
        print(f"{'='*60}")
        
        if os.path.exists(self.models_dir / "cifar10_model.pth"):
            response = input("Model already exists. Retrain? (y/N): ")
            if response.lower() != 'y':
                print("Skipping training")
                return True
        
        train_script = self.project_root / "src" / "train" / "train.py"
        success, output = self.run_command(f"python {train_script}", cwd=self.project_root)
        
        if success:
            print("✓ Model training completed")
            return True
        else:
            print("✗ Model training failed")
            return False
    
    def convert_to_onnx(self):
        """Конвертирует модель в ONNX"""
        print(f"\n{'='*60}")
        print("STEP 2: CONVERTING TO ONNX")
        print(f"{'='*60}")
        
        if os.path.exists(self.models_dir / "cifar10_model.onnx"):
            response = input("ONNX model already exists. Reconvert? (y/N): ")
            if response.lower() != 'y':
                print("Skipping ONNX conversion")
                return True
        
        convert_script = self.project_root / "src" / "convert" / "to_onnx.py"
        success, output = self.run_command(f"python {convert_script}", cwd=self.project_root)
        
        if success:
            print("✓ ONNX conversion completed")
            return True
        else:
            print("✗ ONNX conversion failed")
            return False
    
    def optimize_models(self):
        """Оптимизирует модели"""
        print(f"\n{'='*60}")
        print("STEP 3: OPTIMIZING MODELS")
        print(f"{'='*60}")
        
        optimize_script = self.project_root / "src" / "optimize" / "pytorch_optimize.py"
        success, output = self.run_command(f"python {optimize_script}", cwd=self.project_root)
        
        if success:
            print("✓ Model optimization completed")
            return True
        else:
            print("✗ Model optimization failed")
            return False
    
    def prepare_triton_models(self):
        """Подготавливает модели для Triton"""
        print(f"\n{'='*60}")
        print("STEP 4: PREPARING TRITON MODELS")
        print(f"{'='*60}")
        
        prepare_script = self.project_root / "src" / "convert" / "prepare_triton.py"
        success, output = self.run_command(f"python {prepare_script}", cwd=self.project_root)
        
        if success:
            print("✓ Triton model preparation completed")
            return True
        else:
            print("✗ Triton model preparation failed")
            return False
    
    def start_services(self):
        """Запускает все сервисы"""
        print(f"\n{'='*60}")
        print("STEP 5: STARTING SERVICES")
        print(f"{'='*60}")
        
        # Останавливаем существующие контейнеры
        print("Stopping existing containers...")
        self.run_command("docker-compose down", cwd=self.project_root)
        
        # Собираем образы
        print("Building Docker images...")
        success, _ = self.run_command("docker-compose build", cwd=self.project_root)
        if not success:
            print("✗ Failed to build Docker images")
            return False
        
        # Запускаем сервисы
        print("Starting services...")
        success, _ = self.run_command("docker-compose up -d", cwd=self.project_root)
        if not success:
            print("✗ Failed to start services")
            return False
        
        print("✓ Services started successfully")
        print("Waiting for services to be ready...")
        
        # Ждем готовности сервисов
        max_wait = 120  # 2 минуты
        for i in range(max_wait):
            # Проверяем Triton
            success_triton, _ = self.run_command(
                "curl -f http://localhost:8000/v2/health/ready", 
                cwd=self.project_root
            )
            
            # Проверяем preprocessing service
            success_preprocess, _ = self.run_command(
                "curl -f http://localhost:8080/health", 
                cwd=self.project_root
            )
            
            if success_triton and success_preprocess:
                print("✓ All services are ready!")
                return True
            
            time.sleep(5)
            if (i + 1) % 6 == 0:
                print(f"Still waiting... ({i+1}/{max_wait})")
        
        print("✗ Services did not become ready in time")
        return False
    
    def run_tests(self):
        """Запускает тесты"""
        print(f"\n{'='*60}")
        print("STEP 6: RUNNING TESTS")
        print(f"{'='*60}")
        
        # Тест preprocessing service
        print("Testing preprocessing service...")
        test_script = self.project_root / "src" / "preprocess_service" / "test_service.py"
        success, _ = self.run_command(f"python {test_script} http://localhost:8080", cwd=self.project_root)
        
        if not success:
            print("✗ Preprocessing service tests failed")
            return False
        
        # Тест inference через Triton
        print("Testing inference pipeline...")
        success, _ = self.run_command(
            "docker-compose run --rm test-client", 
            cwd=self.project_root
        )
        
        if success:
            print("✓ All tests passed")
            return True
        else:
            print("✗ Some tests failed")
            return False
    
    def show_status(self):
        """Показывает статус сервисов"""
        print(f"\n{'='*60}")
        print("SERVICE STATUS")
        print(f"{'='*60}")
        
        services = [
            ("Triton HTTP", "http://localhost:8000/v2/health/ready"),
            ("Triton Metrics", "http://localhost:8002/metrics"),
            ("Preprocessing Service", "http://localhost:8080/health"),
            ("Prometheus", "http://localhost:9090"),
            ("Grafana", "http://localhost:3000")
        ]
        
        for name, url in services:
            success, _ = self.run_command(f"curl -f {url}")
            status = "✓ Running" if success else "✗ Not responding"
            print(f"{name:<25} {status}")
        
        print("\nAccess URLs:")
        print("- Triton Server: http://localhost:8000")
        print("- Preprocessing Service: http://localhost:8080")
        print("- Prometheus: http://localhost:9090")
        print("- Grafana: http://localhost:3000 (admin/admin)")
        
        print("\nTo view logs: docker-compose logs <service-name>")
        print("To stop services: docker-compose down")
    
    def generate_report(self):
        """Генерирует отчет о проекте"""
        print(f"\n{'='*60}")
        print("GENERATING PROJECT REPORT")
        print(f"{'='*60}")
        
        report_content = f"""# ML Deployment Project Report

## Project Overview
Этот проект демонстрирует полный пайплайн развертывания модели машинного обучения от обучения до продакшена.

## Completed Steps

### 1. Model Training
- ✓ Обучена CNN модель для классификации CIFAR-10
- ✓ Модель сохранена в: `{self.models_dir}/cifar10_model.pth`

### 2. Model Conversion & Optimization
- ✓ Конвертация в ONNX: `{self.models_dir}/cifar10_model.onnx`
- ✓ Квантизация модели: `{self.models_dir}/cifar10_model_quantized.pth`
- ✓ Прунинг модели: `{self.models_dir}/cifar10_model_pruned.pth`
- ✓ Комбинированная оптимизация: `{self.models_dir}/cifar10_model_combined.pth`

### 3. Microservice Architecture
- ✓ FastAPI микросервис предобработки изображений
- ✓ Triton Inference Server для высокопроизводительного инференса
- ✓ Docker контейнеризация всех компонентов

### 4. Monitoring & Observability
- ✓ Prometheus для сбора метрик
- ✓ Grafana для визуализации
- ✓ Health checks для всех сервисов

### 5. Deployment & Orchestration
- ✓ Docker Compose для оркестрации
- ✓ Автоматические тесты
- ✓ CI/CD готовность

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│ Preprocess API   │───▶│ Triton Server   │
│                 │    │ (FastAPI)        │    │ (ONNX/PyTorch)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Prometheus     │    │   GPU/CPU       │
                       │   (Metrics)      │    │   Inference     │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │     Grafana      │
                       │   (Dashboard)    │
                       └──────────────────┘
```

## Performance Results

См. логи бенчмарков для детальных метрик производительности.

## Access Points

- **Triton Server**: http://localhost:8000
- **Preprocessing API**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Usage Examples

### 1. Preprocessing API
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8080/preprocess/single
```

### 2. Direct Triton Inference
```python
import tritonclient.http as httpclient
client = httpclient.InferenceServerClient(url="localhost:8000")
# ... (см. src/client/test_inference.py для полного примера)
```

## File Structure

```
{self.project_root}/
├── models/                    # Обученные модели
├── src/
│   ├── train/                # Код обучения
│   ├── convert/              # Конвертация моделей
│   ├── optimize/             # Оптимизация
│   ├── preprocess_service/   # FastAPI микросервис
│   └── client/               # Клиентские скрипты
├── triton/
│   └── model_repository/     # Модели для Triton
├── monitoring/               # Конфигурация мониторинга
├── docker/                   # Dockerfile'ы
└── docker-compose.yml        # Оркестрация
```

## Next Steps

1. Добавить аутентификацию и авторизацию
2. Реализовать A/B тестирование моделей
3. Добавить автоматическое масштабирование
4. Интегрировать с Kubernetes
5. Добавить модель дрифта мониторинг

## Заключение

Проект успешно демонстрирует современный подход к развертыванию ML моделей с использованием:
- Контейнеризации (Docker)
- Микросервисной архитектуры
- Высокопроизводительного инференса (Triton)
- Комплексного мониторинга
- Автоматизированного тестирования

Система готова к продакшен развертыванию и может быть легко масштабирована.
"""
        
        report_path = self.project_root / "PROJECT_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ Project report generated: {report_path}")
        return True
    
    def run_full_pipeline(self, epochs=10):  # Уменьшаем по умолчанию для CPU
        """Запускает полный пайплайн проекта"""
        print("="*80)
        print("STARTING FULL ML DEPLOYMENT PIPELINE")
        print("="*80)
        
        steps = [
            ("Checking dependencies", self.check_dependencies),
            ("Training model", lambda: self.train_model(epochs)),
            ("Converting to ONNX", self.convert_to_onnx),
            ("Optimizing models", self.optimize_models),
            ("Preparing Triton models", self.prepare_triton_models),
            ("Starting services", self.start_services),
            ("Running tests", self.run_tests),
            ("Generating report", self.generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name.upper()} {'='*20}")
            success = step_func()
            
            if not success:
                print(f"\n❌ PIPELINE FAILED AT: {step_name}")
                print("Check the logs above for details.")
                return False
        
        print(f"\n{'='*80}")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        self.show_status()
        
        return True


def main():
    parser = argparse.ArgumentParser(description='ML Project Pipeline Runner')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10 for CPU)')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--status-only', action='store_true', help='Only show service status')
    parser.add_argument('--stop', action='store_true', help='Stop all services')
    
    args = parser.parse_args()
    
    runner = MLProjectRunner()
    
    if args.stop:
        print("Stopping all services...")
        runner.run_command("docker-compose down", cwd=runner.project_root)
        return
    
    if args.status_only:
        runner.show_status()
        return
    
    if args.skip_training:
        # Проверяем наличие модели
        if not os.path.exists(runner.models_dir / "cifar10_model.pth"):
            print("❌ Model not found and training is skipped!")
            print("Either train the model first or remove --skip-training flag")
            sys.exit(1)
        
        print("Skipping training, starting from conversion...")
        steps = [
            ("Checking dependencies", runner.check_dependencies),
            ("Converting to ONNX", runner.convert_to_onnx),
            ("Optimizing models", runner.optimize_models),
            ("Preparing Triton models", runner.prepare_triton_models),
            ("Starting services", runner.start_services),
            ("Running tests", runner.run_tests),
            ("Generating report", runner.generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name.upper()} {'='*20}")
            success = step_func()
            
            if not success:
                print(f"\n❌ PIPELINE FAILED AT: {step_name}")
                sys.exit(1)
        
        runner.show_status()
    else:
        success = runner.run_full_pipeline(epochs=args.epochs)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
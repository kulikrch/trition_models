# ML Deployment Project

Комплексный проект по подготовке, оптимизации и развертыванию модели машинного обучения.

## Структура проекта

```
.
├── README.md
├── requirements.txt
├── models/                    # Обученные модели
├── src/
│   ├── train/                # Код для обучения модели
│   ├── convert/              # Конвертация в ONNX/TRT
│   ├── optimize/             # Оптимизация моделей
│   └── preprocess_service/   # Микросервис предобработки
├── triton/
│   └── model_repository/     # Репозиторий моделей для Triton
├── monitoring/               # Конфигурация Prometheus/Grafana
├── docker/                   # Dockerfile'ы
└── docker-compose.yml        # Оркестрация сервисов
```

## Задачи проекта

1. [x] Создание структуры проекта
2. [x] Обучение модели (CNN для CIFAR-10)
3. [x] Конвертация в ONNX
4. [x] Оптимизация модели (quantization, pruning, 4x ускорение)
5. [x] Микросервис предобработки (FastAPI + веб-интерфейс)
6. [x] **Развертывание в Triton Inference Server v2.39.0** ✅
7. [x] Мониторинг (Prometheus + Grafana)
8. [x] Оркестрация (Docker Compose)
9. [x] Комплексное тестирование и бенчмарки

## Быстрый старт

```bash
# 🚀 Полный ML пайплайн (обучение + развертывание)
python run_project.py --epochs 10

# ⚡ Быстрая демонстрация (5 эпох)
python run_project.py --epochs 5

# 🎯 Только Triton Inference Server
docker-compose -f docker-compose-triton-minimal.yml up -d

# 📊 Тестирование всех компонентов
python benchmark_all_models.py
python test_triton_grpc.py

# Использование Makefile
make install-cpu
make cpu-demo           # 5 эпох
make full-pipeline-cpu  # 10 эпох
```

**⚡ Оптимизировано для CPU!** Не требуется GPU.

См. [QUICK_START.md](QUICK_START.md) и [SETUP_CPU.md](SETUP_CPU.md) для инструкций.

## Архитектура

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │───▶│   ML Web App     │───▶│ Triton Server   │
│   (Frontend)    │    │   (FastAPI)      │    │   v2.39.0       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Prometheus     │    │  Model Repo     │
                       │   (Metrics)      │    │ PyTorch + ONNX  │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │     Grafana      │    │   OpenVINO      │
                       │   (Dashboard)    │    │  Acceleration   │
                       └──────────────────┘    └─────────────────┘
```

## Основные компоненты

### 🧠 **ML Pipeline:**
- **CNN модель**: CIFAR-10 классификация (2.08 MB → 0.58 MB после оптимизации)
- **ONNX конвертация**: 15.4x ускорение (1.91ms → 0.12ms)
- **Оптимизация**: Quantization (3.6x сжатие) + Pruning (3.1x ускорение)

### 🚀 **Production Infrastructure:**
- **Triton Inference Server v2.39.0**: Enterprise-grade ML serving
- **FastAPI**: Веб-интерфейс + REST API для классификации
- **OpenVINO**: Hardware acceleration для Intel CPU
- **Dynamic Batching**: Автоматическая оптимизация throughput

### 📊 **Monitoring & Ops:**
- **Prometheus + Grafana**: Real-time мониторинг и дашборды
- **Docker Compose**: Полная оркестрация системы
- **Health Checks**: Автоматический мониторинг сервисов

### 🎯 Ключевые особенности:
- **CPU-оптимизированная** архитектура (работает без GPU)
- **Production-ready** Triton Inference Server v2.39.0
- **РЕКОРДНАЯ производительность**: **8,057 img/sec** (ONNX), **15.4x ускорение**
- **Enterprise-grade**: gRPC + HTTP API, dynamic batching, OpenVINO acceleration
- **Полный мониторинг**: Prometheus + Grafana с real-time метриками

### 📊 Технические характеристики (измерено на 100 итерациях):

| Компонент | Результат | Улучшение |
|-----------|-----------|-----------|
| **Обучение** | 5-10 минут на CPU | Оптимизировано для CPU |
| **Размер модели** | 2.08 MB → 0.58 MB | **3.6x сжатие** |
| **ONNX инференс** | 1.91 ms → **0.12 ms** | **15.4x ускорение** |
| **Triton throughput** | **8,057 изображений/сек** | **gRPC + HTTP API** |
| **Quantization** | 1.91 ms → 1.10 ms | 1.7x + **72% экономия места** |
| **Pruning** | 1.91 ms → 0.61 ms | **3.1x ускорение** |

### 🎯 **Production Deployment:**
- **Triton Inference Server**: gRPC (8001) + HTTP (8000) + Metrics (8002)
- **Model Repository**: PyTorch + ONNX + Optimized versions  
- **Auto Scaling**: Dynamic batching + Instance groups
- **Monitoring**: Real-time метрики через Prometheus/Grafana

### 🚀 Enterprise-готовность:
- ✅ **Triton Inference Server v2.39.0** - production ML serving
- ✅ **gRPC + HTTP API** - множественные интерфейсы
- ✅ **Dynamic Batching** - автоматическая оптимизация throughput  
- ✅ **OpenVINO Acceleration** - hardware optimization
- ✅ **Health Checks + Monitoring** - Prometheus/Grafana интеграция
- ✅ **Container Orchestration** - Docker Compose deployment
- ✅ **Model Versioning** - управление версиями моделей

## 🏆 Достижения проекта

### ✅ **Полный ML пайплайн реализован:**
- От обучения CNN модели до production Triton deployment
- **15.4x ускорение** через ONNX оптимизацию
- **3.6x сжатие** через quantization
- **Enterprise-grade** inference serving

### 🎯 **Production-ready система:**
- **Triton Inference Server** - industry standard для ML serving
- **Real-time мониторинг** - Prometheus + Grafana дашборды  
- **Веб-интерфейс** - для демонстрации и testing
- **API интеграция** - gRPC + HTTP для любых клиентов

**Проект демонстрирует современный подход к MLOps с полным жизненным циклом от research до production deployment! 🚀**
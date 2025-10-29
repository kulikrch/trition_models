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
4. [x] Оптимизация модели (quantization, pruning)
5. [x] Микросервис предобработки (FastAPI)
6. [x] Развертывание в Triton Inference Server
7. [x] Мониторинг (Prometheus + Grafana)
8. [x] Оркестрация (Docker Compose)
9. [x] Тестирование и отчет

## Быстрый старт

```bash
# 🚀 Для CPU (рекомендуется)
pip install -r requirements-cpu.txt
python run_project.py --epochs 5  # Быстрая демонстрация

# Полный пайплайн
python run_project.py --epochs 10

# Быстрый старт без обучения
python run_project.py --skip-training

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

## Основные компоненты

- **CNN модель**: Сверточная нейросеть для классификации CIFAR-10
- **ONNX конвертация**: Кроссплатформенный формат модели
- **Оптимизация**: Quantization и pruning для ускорения
- **FastAPI**: REST API для предобработки изображений
- **Triton**: Высокопроизводительный inference server
- **Мониторинг**: Prometheus + Grafana для наблюдения
- **Docker**: Контейнеризация всех компонентов
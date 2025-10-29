# 💻 Настройка для CPU (без GPU)

Специальные инструкции для запуска проекта на компьютерах без NVIDIA GPU.

## Быстрая установка

```bash
# Установка CPU-версии зависимостей
pip install -r requirements-cpu.txt

# Или установка отдельных пакетов
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime  # CPU версия
```

## Оптимизированные настройки для CPU

### 1. Уменьшенные batch sizes
- Обучение: batch_size=64 (вместо 128)
- Инференс: batch_size=1-8 (для тестирования)

### 2. Количество воркеров
- DataLoader: num_workers=0 (для избежания проблем на Windows)

### 3. Triton конфигурация
- Все модели настроены на `KIND_CPU`
- Убраны GPU-специфичные оптимизации
- Используется OpenVINO для ONNX (если доступен)

## Ожидаемая производительность

### Обучение модели (50 эпох)
- **CPU (4 ядра)**: ~30-60 минут
- **CPU (8 ядер)**: ~15-30 минут

### Инференс
- **PyTorch CPU**: ~50-100ms на изображение
- **ONNX CPU**: ~20-50ms на изображение
- **Quantized модель**: ~10-30ms на изображение

## Быстрый запуск для CPU

```bash
# 1. Установка зависимостей
pip install -r requirements-cpu.txt

# 2. Полный пайплайн (с уменьшенным количеством эпох для тестирования)
python run_project.py --epochs 10

# 3. Или быстрый старт без обучения
python run_project.py --skip-training
```

## Оптимизации для ускорения на CPU

### 1. Уменьшите количество эпох для тестирования
```bash
python run_project.py --epochs 5
```

### 2. Используйте квантизованные модели
Квантизация дает значительное ускорение на CPU:
- Размер модели: ~4x меньше
- Скорость инференса: ~2-3x быстрее

### 3. Batch inference
Для максимальной производительности используйте батчи:
```python
# Вместо 10 отдельных запросов
batch_size = 4
# Обрабатывайте по 4 изображения за раз
```

## Docker настройки для CPU

Docker Compose автоматически настроен для CPU. Triton будет использовать:
- CPU backend для PyTorch моделей
- ONNX Runtime CPU для ONNX моделей
- Без GPU ускорения

## Мониторинг CPU использования

В Grafana дашборде отслеживайте:
- CPU utilization (через node-exporter)
- Memory usage
- Request latency
- Throughput (requests/second)

## Troubleshooting для CPU

### Проблема: Медленная загрузка данных
```bash
# Решение: уменьшить num_workers
num_workers=0  # в DataLoader
```

### Проблема: OutOfMemory на CPU
```bash
# Решение: уменьшить batch_size
batch_size=32  # или меньше
```

### Проблема: Triton не запускается
```bash
# Проверьте, что используется CPU backend
docker-compose logs triton
```

## Альтернативы для ускорения

### 1. Intel OpenVINO (если поддерживается)
```bash
pip install openvino
# Автоматически используется Triton для ONNX моделей
```

### 2. ONNX Runtime с дополнительными провайдерами
```bash
# Для Intel CPU
pip install onnxruntime-extensions
```

### 3. TensorFlow Lite (опционально)
Для еще большего ускорения можно добавить конвертацию в TFLite.

## Результаты на типичном ноутбуке

**Конфигурация**: Intel i5-8250U, 8GB RAM

- **Обучение (10 эпох)**: ~15 минут
- **Конвертация в ONNX**: ~30 секунд
- **Оптимизация**: ~2 минуты
- **Запуск всех сервисов**: ~1 минута
- **Тестирование**: ~30 секунд

**Итого**: ~20 минут для полного пайплайна

Производительность вполне приемлема для демонстрации и обучения! 🚀
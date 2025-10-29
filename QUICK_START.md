# 🚀 Quick Start Guide

Быстрый старт для ML Deployment проекта.

## Предварительные требования

- Python 3.8+
- Docker & Docker Compose
- NVIDIA GPU (опционально, для ускорения)
- 8GB+ RAM

## Установка зависимостей

```bash
# Установка Python зависимостей
pip install -r requirements.txt

# Проверка Docker
docker --version
docker-compose --version
```

## Вариант 1: Полный пайплайн (с обучением)

```bash
# Запустить весь пайплайн от начала до конца
python run_project.py

# Или с помощью Makefile
make full-pipeline
```

Этот процесс займет ~30-60 минут в зависимости от вашего железа.

## Вариант 2: Быстрый старт (без обучения)

Если вы хотите пропустить обучение модели и использовать предобученную:

```bash
# Загрузите предобученную модель (если есть) в models/cifar10_model.pth
# Затем запустите:
python run_project.py --skip-training

# Или
make quick-start
```

## Вариант 3: Пошаговый запуск

```bash
# 1. Обучение модели
make train

# 2. Конвертация в ONNX
make convert

# 3. Оптимизация моделей
make optimize

# 4. Подготовка для Triton
make prepare

# 5. Развертывание сервисов
make deploy

# 6. Тестирование
make test
```

## Проверка статуса

```bash
# Проверить статус всех сервисов
make status

# Быстрая проверка здоровья
make test-health

# Просмотр логов
make logs
```

## Доступ к сервисам

После успешного запуска:

- **Triton Server**: http://localhost:8000
- **Preprocessing API**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Тестирование API

```bash
# Тест preprocessing service
curl -X POST -F "file=@test_image.jpg" http://localhost:8080/preprocess/single

# Информация о preprocessing
curl http://localhost:8080/info

# Метрики
curl http://localhost:8080/metrics
```

## Остановка сервисов

```bash
# Остановить все сервисы
make stop

# Полная очистка
make clean
```

## Решение проблем

### Ошибка "CUDA out of memory"
```bash
# Уменьшите batch size в конфигурации или используйте CPU
export CUDA_VISIBLE_DEVICES=""
```

### Сервисы не запускаются
```bash
# Проверьте логи
docker-compose logs triton
docker-compose logs preprocess-service

# Перезапустите
make stop
make deploy
```

### Порты заняты
Измените порты в `docker-compose.yml` если нужно.

## Кастомизация

- **Модель**: Измените архитектуру в `src/train/model.py`
- **Данные**: Замените CIFAR-10 на свой датасет в `src/train/train.py`
- **API**: Добавьте новые эндпоинты в `src/preprocess_service/main.py`
- **Мониторинг**: Настройте дашборды в `monitoring/grafana-dashboard.json`

## Следующие шаги

1. Изучите сгенерированный отчет в `PROJECT_REPORT.md`
2. Настройте Grafana дашборды для мониторинга
3. Протестируйте с реальными данными
4. Адаптируйте под свою задачу ML

Успешного деплоя! 🎉
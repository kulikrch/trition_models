# 🚀 ML Deployment Project - Extended Version

Полный ML пайплайн с веб-интерфейсом, мониторингом и контейнеризацией.

## 🎯 Что реализовано

### ✅ Базовый ML пайплайн
- 🏋️ **Обучение модели** на CPU
- 🔄 **Конвертация** PyTorch → ONNX
- ⚡ **Оптимизация** для производительности
- 🧪 **Тестирование** всех компонентов

### 🌐 Веб-приложение
- 📤 **Загрузка изображений** через браузер
- 🤖 **Классификация в реальном времени**
- 📊 **Статистика** предсказаний
- 🎨 **Красивый интерфейс** с drag&drop
- ⚡ **Показ времени инференса**

### 📊 Система мониторинга
- 📈 **Prometheus** для сбора метрик
- 📊 **Grafana** дашборды
- 💻 **Системные метрики** (CPU, память)
- 🐳 **Метрики контейнеров**

### 🐳 Контейнеризация
- 🏗️ **Docker** образы для всех компонентов
- 🎼 **Docker Compose** оркестрация
- 🔄 **Автоматические health checks**
- 🌐 **Nginx** реверс-прокси

## 🚀 Быстрый старт

### 1. Базовая настройка
```bash
# Установка зависимостей
pip install -r requirements-windows.txt
pip install "numpy<2.0" jinja2 python-multipart

# Обучение модели (если еще не сделано)
python working_demo.py
```

### 2. Запуск веб-приложения
```bash
# Простой запуск
python web_app.py

# Или автоматический запуск с выбором опций
python launch_complete_system.py
```

### 3. Полная система с мониторингом
```bash
# Запуск всех сервисов в Docker
docker-compose -f docker-compose-extended.yml up -d

# Или через скрипт
python launch_complete_system.py
# Выберите опцию 2
```

## 🌐 Доступные сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| 🤖 **ML Web App** | http://localhost:8000 | Основное приложение |
| 📊 **Prometheus** | http://localhost:9090 | Метрики |
| 📈 **Grafana** | http://localhost:3000 | Дашборды (admin/admin) |
| 💻 **Node Exporter** | http://localhost:9100 | Системные метрики |
| 🐳 **cAdvisor** | http://localhost:8080 | Метрики контейнеров |

## 📱 Использование веб-приложения

### 🖼️ Загрузка изображений
1. Откройте http://localhost:8000
2. Перетащите изображение или кликните для выбора
3. Получите результат классификации
4. Просматривайте статистику в реальном времени

### 🔗 API эндпоинты
```bash
# Классификация изображения
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict

# Статистика приложения
curl http://localhost:8000/stats

# Проверка здоровья
curl http://localhost:8000/health
```

## 📊 Мониторинг и метрики

### Grafana дашборды
- 📈 **Скорость предсказаний** (predictions/sec)
- ⏱️ **Время инференса** (95th percentile)
- 💻 **Использование CPU/памяти**
- 🎯 **Распределение по классам**
- 🌐 **HTTP метрики**

### Ключевые метрики
- `ml_predictions_total` - общее количество предсказаний
- `ml_inference_duration_seconds` - время инференса
- `ml_predictions_by_class` - предсказания по классам
- `http_request_duration_seconds` - время HTTP запросов

## 🐳 Docker команды

```bash
# Запуск всех сервисов
docker-compose -f docker-compose-extended.yml up -d

# Просмотр логов
docker-compose -f docker-compose-extended.yml logs -f

# Остановка сервисов
docker-compose -f docker-compose-extended.yml down

# Пересборка образов
docker-compose -f docker-compose-extended.yml build --no-cache
```

## ⚡ Производительность

### На типичном CPU (i5-8250U):
- **Время инференса**: ~0.7 мс
- **Пропускная способность**: ~1400 изображений/сек
- **Время отклика веб-интерфейса**: ~50-100 мс
- **Использование памяти**: ~150 MB

### Оптимизации
- ✅ CPU-оптимизированные модели
- ✅ ONNX для ускорения
- ✅ Асинхронный веб-сервер
- ✅ Кеширование статических файлов

## 🔧 Конфигурация

### Переменные окружения
```bash
export ML_MODEL_PATH=models/working_model.pth
export WEB_PORT=8000
export PROMETHEUS_PORT=9090
export GRAFANA_PORT=3000
```

### Кастомизация
- **Модель**: Измените `working_demo.py`
- **Веб-интерфейс**: Редактируйте HTML в `web_app.py`
- **Метрики**: Настройте `prometheus-extended.yml`
- **Дашборды**: Обновите `grafana-dashboard-extended.json`

## 🧪 Тестирование

```bash
# Тест базовых компонентов
python final_test.py

# Тест веб-приложения
curl http://localhost:8000/health

# Нагрузочное тестирование
for i in {1..100}; do
    curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
done
```

## 🛠️ Решение проблем

### Веб-приложение не запускается
```bash
# Проверьте модель
ls -la models/working_model.pth

# Переустановите зависимости
pip install "numpy<2.0" fastapi uvicorn pillow
```

### Docker проблемы
```bash
# Очистка
docker system prune -f
docker-compose -f docker-compose-extended.yml down -v

# Пересборка
docker-compose -f docker-compose-extended.yml build --no-cache
```

### Проблемы с портами
- Измените порты в `docker-compose-extended.yml`
- Или остановите конфликтующие сервисы

## 🎓 Архитектура системы

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │───▶│   ML Web App     │───▶│   ML Model      │
│   (Frontend)    │    │   (FastAPI)      │    │   (PyTorch)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Prometheus     │    │   ONNX Model    │
                       │   (Metrics)      │    │   (Optimized)   │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │     Grafana      │
                       │   (Dashboard)    │
                       └──────────────────┘
```

## 🎯 Что дальше

1. **Автоскейлинг**: Kubernetes deployment
2. **CI/CD**: GitHub Actions пайплайн
3. **A/B тестирование**: Сравнение моделей
4. **Продвинутые метрики**: Drift detection
5. **Безопасность**: Аутентификация и HTTPS

---

**🎉 Поздравляем! У вас работает полноценная ML система в продакшене! 🎉**
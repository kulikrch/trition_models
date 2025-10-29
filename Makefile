# Makefile для ML проекта

.PHONY: help install train convert optimize deploy test clean status stop report

help:  ## Показать это сообщение помощи
	@echo "Доступные команды:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Установить все зависимости
	pip install -r requirements.txt

train:  ## Обучить модель
	python src/train/train.py

convert:  ## Конвертировать модель в ONNX
	python src/convert/to_onnx.py

optimize:  ## Оптимизировать модели
	python src/optimize/pytorch_optimize.py

prepare:  ## Подготовить модели для Triton
	python src/convert/prepare_triton.py

build:  ## Собрать Docker образы
	docker-compose build

deploy:  ## Развернуть все сервисы
	docker-compose up -d

test:  ## Запустить тесты
	python src/preprocess_service/test_service.py http://localhost:8080
	docker-compose run --rm test-client

full-pipeline:  ## Запустить полный пайплайн
	python run_project.py

quick-start:  ## Быстрый старт (без обучения)
	python run_project.py --skip-training

status:  ## Показать статус сервисов
	python run_project.py --status-only

logs:  ## Показать логи всех сервисов
	docker-compose logs -f

stop:  ## Остановить все сервисы
	docker-compose down

clean:  ## Очистить Docker ресурсы
	docker-compose down -v
	docker system prune -f

report:  ## Генерировать отчет
	python run_project.py --status-only

# Отдельные сервисы
triton-logs:  ## Логи Triton
	docker-compose logs -f triton

preprocess-logs:  ## Логи сервиса предобработки
	docker-compose logs -f preprocess-service

prometheus-logs:  ## Логи Prometheus
	docker-compose logs -f prometheus

grafana-logs:  ## Логи Grafana
	docker-compose logs -f grafana

# Быстрые тесты
test-health:  ## Проверить здоровье сервисов
	@echo "Testing Triton..."
	@curl -f http://localhost:8000/v2/health/ready || echo "Triton not ready"
	@echo "Testing Preprocessing Service..."
	@curl -f http://localhost:8080/health || echo "Preprocessing service not ready"

# Развертывание с GPU
deploy-gpu:  ## Развернуть с GPU поддержкой
	docker-compose --profile gpu up -d

# Производственное развертывание
deploy-prod:  ## Производственное развертывание
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
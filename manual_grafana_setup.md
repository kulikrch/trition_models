# 📊 Ручная настройка Grafana Dashboard

## 🔧 Пошаговая инструкция:

### 1. **Доступ к Grafana**
- URL: http://localhost:3000
- Логин: `admin`
- Пароль: `admin`

### 2. **Добавление Data Source**
1. В левом меню нажмите **⚙️ Configuration** → **Data Sources**
2. Нажмите **Add data source**
3. Выберите **Prometheus**
4. Настройки:
   - **Name**: `Prometheus`
   - **URL**: `http://prometheus:9090`
   - Оставьте остальные настройки по умолчанию
5. Нажмите **Save & Test**

### 3. **Создание Dashboard**
1. В левом меню нажмите **➕** → **Dashboard**
2. Нажмите **Add new panel**

#### Panel 1: CPU Usage
- **Query**: `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
- **Title**: `CPU Usage`
- **Y-Axis Label**: `CPU %`

#### Panel 2: Memory Usage  
- **Query**: `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100`
- **Title**: `Memory Usage`
- **Y-Axis Label**: `Memory %`

#### Panel 3: Container Memory
- **Query**: `container_memory_usage_bytes`
- **Title**: `Container Memory`
- **Legend**: `{{name}}`

#### Panel 4: Network Traffic
- **Query**: `rate(container_network_receive_bytes_total[5m])`
- **Title**: `Network RX`

### 4. **Быстрые метрики для демонстрации**

Если Prometheus не собирает метрики, используйте эти простые запросы:

```
# Время работы Prometheus
up

# Количество targets
prometheus_notifications_total

# Использование памяти Prometheus
prometheus_tsdb_symbol_table_size_bytes
```

### 5. **Альтернативный способ - импорт JSON**

Создайте новый dashboard и вставьте этот JSON:

```json
{
  "dashboard": {
    "title": "Simple ML Dashboard",
    "panels": [
      {
        "id": 1,
        "title": "Prometheus Targets",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{instance}}"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "5s"
  }
}
```

## 🎯 Для демонстрации достаточно:

1. **Открыть Grafana**: http://localhost:3000
2. **Войти**: admin/admin  
3. **Показать Explore**: левое меню → Explore
4. **Ввести запрос**: `up` (покажет статус сервисов)
5. **Показать графики**: любые базовые метрики

## 🚨 Если Grafana недоступна:

```bash
# Проверить статус
docker-compose -f docker-compose-extended.yml ps

# Перезапустить Grafana
docker-compose -f docker-compose-extended.yml restart grafana

# Посмотреть логи
docker-compose -f docker-compose-extended.yml logs grafana
```

## 📈 Главное для отчета:

**Показать, что система мониторинга настроена и работает!**
- ✅ Grafana интерфейс открывается
- ✅ Prometheus как data source
- ✅ Любые базовые метрики отображаются
- ✅ Демонстрация real-time обновления
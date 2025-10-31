"""
Скрипт для настройки дашборда Grafana через API
"""
import requests
import json
import time

def wait_for_grafana(url="http://localhost:3000", max_retries=30):
    """Ждет запуска Grafana"""
    print("⏳ Ожидание запуска Grafana...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Grafana запущена!")
                return True
        except:
            pass
        
        time.sleep(2)
        if (i + 1) % 5 == 0:
            print(f"   Попытка {i+1}/{max_retries}...")
    
    print("❌ Grafana не отвечает")
    return False

def create_dashboard():
    """Создает дашборд через API"""
    grafana_url = "http://localhost:3000"
    auth = ("admin", "admin")
    
    # Дашборд конфигурация
    dashboard_config = {
        "dashboard": {
            "title": "ML Web Application Dashboard",
            "tags": ["ml", "webapp", "cpu"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "Requests/sec",
                            "refId": "A"
                        }
                    ],
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 0
                    },
                    "yAxes": [
                        {
                            "label": "Requests/sec"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "CPU Usage",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                            "legendFormat": "CPU Usage %",
                            "refId": "A"
                        }
                    ],
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 0
                    },
                    "yAxes": [
                        {
                            "label": "CPU %",
                            "max": 100
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Memory Usage",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                            "legendFormat": "Memory Usage %",
                            "refId": "A"
                        }
                    ],
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 8
                    },
                    "yAxes": [
                        {
                            "label": "Memory %",
                            "max": 100
                        }
                    ]
                },
                {
                    "id": 4,
                    "title": "Container Stats",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "container_memory_usage_bytes",
                            "legendFormat": "{{name}}",
                            "refId": "A"
                        }
                    ],
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 8
                    }
                }
            ],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "refresh": "5s"
        },
        "overwrite": True
    }
    
    return dashboard_config

def setup_grafana():
    """Настраивает Grafana"""
    print("🔧 НАСТРОЙКА GRAFANA DASHBOARD")
    print("=" * 40)
    
    # Ждем запуска Grafana
    if not wait_for_grafana():
        return False
    
    grafana_url = "http://localhost:3000"
    auth = ("admin", "admin")
    
    try:
        # 1. Проверяем подключение
        print("🔍 Проверка подключения...")
        response = requests.get(f"{grafana_url}/api/org", auth=auth)
        if response.status_code != 200:
            print(f"❌ Ошибка подключения: {response.status_code}")
            return False
        print("✅ Подключение установлено")
        
        # 2. Проверяем data source
        print("🔍 Проверка Prometheus data source...")
        response = requests.get(f"{grafana_url}/api/datasources", auth=auth)
        
        datasources = response.json()
        prometheus_exists = any(ds.get('type') == 'prometheus' for ds in datasources)
        
        if not prometheus_exists:
            print("📊 Создание Prometheus data source...")
            datasource_config = {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy",
                "isDefault": True
            }
            
            response = requests.post(
                f"{grafana_url}/api/datasources",
                json=datasource_config,
                auth=auth
            )
            
            if response.status_code in [200, 409]:  # 409 = already exists
                print("✅ Prometheus data source создан")
            else:
                print(f"❌ Ошибка создания data source: {response.status_code}")
                print(response.text)
        else:
            print("✅ Prometheus data source уже существует")
        
        # 3. Создаем дашборд
        print("📈 Создание дашборда...")
        dashboard_config = create_dashboard()
        
        response = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json=dashboard_config,
            auth=auth
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"{grafana_url}{result.get('url', '')}"
            print("✅ Дашборд создан успешно!")
            print(f"🔗 URL: {dashboard_url}")
            return True
        else:
            print(f"❌ Ошибка создания дашборда: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Главная функция"""
    print("🚀 АВТОМАТИЧЕСКАЯ НАСТРОЙКА GRAFANA")
    print("=" * 50)
    
    success = setup_grafana()
    
    if success:
        print("\n🎉 НАСТРОЙКА ЗАВЕРШЕНА!")
        print("📋 Что делать дальше:")
        print("   1. Откройте http://localhost:3000")
        print("   2. Войдите: admin / admin")
        print("   3. Найдите дашборд 'ML Web Application Dashboard'")
        print("   4. Наслаждайтесь метриками!")
    else:
        print("\n❌ Настройка не удалась")
        print("💡 Попробуйте:")
        print("   1. Проверить, что Grafana запущена: docker-compose logs grafana")
        print("   2. Проверить, что Prometheus запущен: docker-compose logs prometheus")
        print("   3. Перезапустить: docker-compose restart grafana prometheus")

if __name__ == "__main__":
    main()
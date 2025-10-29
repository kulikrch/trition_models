"""
Скрипт для запуска полной системы
"""
import subprocess
import time
import webbrowser
import requests
import os
import sys

def run_command(command, background=False):
    """Запускает команду"""
    print(f"🔧 Running: {command}")
    if background:
        return subprocess.Popen(command, shell=True)
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0

def check_service(url, name, timeout=5):
    """Проверяет доступность сервиса"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name} is running")
            return True
        else:
            print(f"⚠️ {name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} is not accessible: {e}")
        return False

def wait_for_services():
    """Ждет запуска всех сервисов"""
    services = [
        ("http://localhost:8000/health", "ML Web App"),
        ("http://localhost:9090", "Prometheus"),
        ("http://localhost:3000", "Grafana"),
    ]
    
    print("⏳ Waiting for services to start...")
    max_retries = 30
    
    for retry in range(max_retries):
        all_ready = True
        
        for url, name in services:
            if not check_service(url, name):
                all_ready = False
        
        if all_ready:
            print("🎉 All services are ready!")
            return True
        
        time.sleep(5)
        print(f"   Retry {retry + 1}/{max_retries}...")
    
    print("⚠️ Some services may not be ready, but continuing...")
    return False

def main():
    """Главная функция"""
    print("🚀 " + "="*60)
    print("🚀 LAUNCHING COMPLETE ML SYSTEM")
    print("🚀 " + "="*60)
    
    # Проверяем, что модель существует
    if not os.path.exists("models/working_model.pth"):
        print("❌ Model not found! Please run working_demo.py first:")
        print("   python working_demo.py")
        sys.exit(1)
    
    # Опции запуска
    print("\nВыберите вариант запуска:")
    print("1. 🌐 Только веб-приложение (уже запущено)")
    print("2. 📊 Полная система с мониторингом (Docker)")
    print("3. 🔧 Проверить текущие сервисы")
    
    choice = input("\nВаш выбор (1-3): ").strip()
    
    if choice == "1":
        print("\n🌐 Веб-приложение уже запущено!")
        print("📱 Откройте браузер: http://localhost:8000")
        
        # Открываем браузер
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass
        
        print("\n✨ Возможности веб-приложения:")
        print("   📤 Загрузка изображений")
        print("   🤖 Классификация в реальном времени")
        print("   📊 Статистика предсказаний")
        print("   ⚡ Показ времени инференса")
        
    elif choice == "2":
        print("\n📊 Запуск полной системы с мониторингом...")
        
        # Проверяем Docker
        if not run_command("docker --version"):
            print("❌ Docker не найден! Установите Docker и повторите.")
            sys.exit(1)
        
        if not run_command("docker-compose --version"):
            print("❌ Docker Compose не найден!")
            sys.exit(1)
        
        print("🐳 Запуск Docker контейнеров...")
        
        # Останавливаем существующие контейнеры
        run_command("docker-compose -f docker-compose-extended.yml down")
        
        # Собираем и запускаем
        if run_command("docker-compose -f docker-compose-extended.yml build"):
            print("✅ Образы собраны успешно")
        else:
            print("⚠️ Ошибка при сборке образов")
        
        # Запускаем сервисы
        process = run_command("docker-compose -f docker-compose-extended.yml up -d", background=True)
        
        if wait_for_services():
            print("\n🎉 Система запущена успешно!")
            
            print("\n📋 Доступные сервисы:")
            print("   🌐 ML Web App:  http://localhost:8000")
            print("   📊 Prometheus:  http://localhost:9090")
            print("   📈 Grafana:     http://localhost:3000 (admin/admin)")
            print("   💻 Node Stats:  http://localhost:9100")
            print("   🐳 cAdvisor:    http://localhost:8080")
            
            # Открываем основные сервисы
            try:
                webbrowser.open("http://localhost:8000")
                time.sleep(2)
                webbrowser.open("http://localhost:3000")
            except:
                pass
        
    elif choice == "3":
        print("\n🔍 Проверка текущих сервисов...")
        
        services = [
            ("http://localhost:8000", "ML Web App"),
            ("http://localhost:8000/health", "ML App Health"),
            ("http://localhost:8000/stats", "ML App Stats"),
            ("http://localhost:9090", "Prometheus"),
            ("http://localhost:3000", "Grafana"),
            ("http://localhost:9100", "Node Exporter"),
            ("http://localhost:8080", "cAdvisor"),
        ]
        
        for url, name in services:
            check_service(url, name, timeout=3)
        
        print("\n📊 Для просмотра логов Docker:")
        print("   docker-compose -f docker-compose-extended.yml logs -f")
        
        print("\n🛑 Для остановки всех сервисов:")
        print("   docker-compose -f docker-compose-extended.yml down")
    
    else:
        print("❌ Неверный выбор!")
        sys.exit(1)
    
    print("\n🎯 " + "="*60)
    print("🎯 СИСТЕМА ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
    print("🎯 " + "="*60)
    
    print("\n💡 Что можно попробовать:")
    print("   🖼️ Загрузить изображение через веб-интерфейс")
    print("   📈 Посмотреть метрики в Grafana")
    print("   🧪 Протестировать API: curl -X POST -F 'file=@image.jpg' http://localhost:8000/predict")
    print("   📊 Посмотреть статистику: http://localhost:8000/stats")

if __name__ == "__main__":
    main()
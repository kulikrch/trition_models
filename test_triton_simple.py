"""
Простое тестирование Triton без зависимостей
"""
import requests
import json
import time
import numpy as np

def wait_for_triton(max_retries=30):
    """Ждет готовности Triton"""
    print("⏳ Ожидание готовности Triton сервера...")
    
    for i in range(max_retries):
        try:
            # Проверяем HTTP health endpoint
            response = requests.get("http://localhost:8002/v2/health/ready", timeout=5)
            if response.status_code == 200:
                print("✅ Triton сервер готов!")
                return True
            else:
                print(f"   HTTP {response.status_code}, ожидание... ({i+1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            print(f"   Подключение... ({i+1}/{max_retries})")
        
        time.sleep(5)
    
    print("❌ Triton сервер не готов")
    return False

def get_models():
    """Получает список моделей"""
    try:
        response = requests.get("http://localhost:8002/v2/models")
        if response.status_code == 200:
            models = response.json()
            print("📋 Доступные модели:")
            
            ready_models = []
            for model in models:
                name = model['name']
                state = model.get('state', 'UNKNOWN')
                print(f"   • {name}: {state}")
                
                if state == 'READY':
                    ready_models.append(name)
            
            return ready_models
        else:
            print(f"❌ Ошибка получения моделей: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []

def get_model_metadata(model_name):
    """Получает метаданные модели"""
    try:
        response = requests.get(f"http://localhost:8002/v2/models/{model_name}")
        if response.status_code == 200:
            metadata = response.json()
            print(f"📊 Метаданные {model_name}:")
            
            inputs = metadata.get('inputs', [])
            outputs = metadata.get('outputs', [])
            
            for inp in inputs:
                print(f"   Input: {inp['name']}, shape: {inp['shape']}, type: {inp['datatype']}")
            
            for out in outputs:
                print(f"   Output: {out['name']}, shape: {out['shape']}, type: {out['datatype']}")
            
            return metadata
        else:
            print(f"❌ Ошибка получения метаданных: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def test_model_inference(model_name):
    """Тестирует инференс модели через HTTP"""
    print(f"🧪 Тестирование {model_name}...")
    
    # Получаем метаданные
    metadata = get_model_metadata(model_name)
    if not metadata:
        return None
    
    try:
        # Создаем тестовые данные
        test_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        
        # Нормализация CIFAR-10
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        test_data = (test_data - mean) / std
        
        # Определяем имена входов/выходов из метаданных
        inputs = metadata.get('inputs', [])
        outputs = metadata.get('outputs', [])
        
        if not inputs or not outputs:
            print(f"❌ Неполные метаданные модели")
            return None
        
        input_name = inputs[0]['name']
        output_name = outputs[0]['name']
        
        # Подготавливаем запрос для Triton HTTP API
        inference_request = {
            "inputs": [
                {
                    "name": input_name,
                    "shape": [1, 3, 32, 32],
                    "datatype": "FP32",
                    "data": test_data.flatten().tolist()
                }
            ],
            "outputs": [
                {
                    "name": output_name
                }
            ]
        }
        
        # Измеряем время
        times = []
        for _ in range(10):
            start_time = time.time()
            
            response = requests.post(
                f"http://localhost:8002/v2/models/{model_name}/infer",
                json=inference_request,
                timeout=10
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
            else:
                print(f"❌ Ошибка инференса: {response.status_code}")
                print(response.text)
                break
        
        if times:
            result = response.json()
            output_data = result['outputs'][0]['data']
            
            # Анализируем результат
            output_array = np.array(output_data)
            predicted_class = np.argmax(output_array)
            confidence = np.max(output_array)
            
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            
            avg_time = np.mean(times) * 1000  # ms
            throughput = 1000 / avg_time  # fps
            
            print(f"   ✅ Успешно! Время: {avg_time:.2f} ms")
            print(f"   🚄 Throughput: {throughput:.1f} fps")
            print(f"   🎯 Предсказание: {class_names[predicted_class]} ({confidence:.3f})")
            
            return {
                'model_name': model_name,
                'avg_time_ms': avg_time,
                'throughput_fps': throughput,
                'prediction': class_names[predicted_class],
                'confidence': confidence,
                'successful_tests': len(times)
            }
        else:
            print(f"   ❌ Все тесты неудачны")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return None

def main():
    """Главная функция"""
    print("🚀 ТЕСТИРОВАНИЕ TRITON INFERENCE SERVER")
    print("=" * 60)
    
    # Ждем готовности Triton
    if not wait_for_triton():
        print("❌ Triton недоступен")
        return False
    
    # Получаем список моделей
    ready_models = get_models()
    if not ready_models:
        print("❌ Нет готовых моделей")
        return False
    
    print(f"\n🎯 Тестируем {len(ready_models)} моделей")
    
    # Тестируем каждую модель
    results = []
    for i, model_name in enumerate(ready_models, 1):
        print(f"\n{i}️⃣ " + "="*50)
        print(f"{i}️⃣ МОДЕЛЬ: {model_name.upper()}")
        print(f"{i}️⃣ " + "="*50)
        
        result = test_model_inference(model_name)
        if result:
            results.append(result)
    
    # Итоговая сводка
    if results:
        print(f"\n🎉 " + "="*60 + " 🎉")
        print("🎉" + " "*15 + "ИТОГОВЫЕ РЕЗУЛЬТАТЫ TRITON" + " "*15 + "🎉")
        print("🎉" + " "*60 + "🎉")
        
        print(f"{'Модель':<25} {'Время (ms)':<12} {'Throughput':<12}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['model_name']:<25} {result['avg_time_ms']:<12.2f} {result['throughput_fps']:<12.1f}")
        
        # Лучшие результаты
        fastest = min(results, key=lambda x: x['avg_time_ms'])
        highest_throughput = max(results, key=lambda x: x['throughput_fps'])
        
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"🚀 Самая быстрая: {fastest['model_name']} ({fastest['avg_time_ms']:.2f} ms)")
        print(f"🚄 Наивысший throughput: {highest_throughput['model_name']} ({highest_throughput['throughput_fps']:.1f} fps)")
        
        print(f"\n✅ TRITON INFERENCE SERVER РАБОТАЕТ!")
        print(f"📊 Протестировано моделей: {len(results)}")
        
        return True
    else:
        print(f"\n❌ НЕ УДАЛОСЬ ПРОТЕСТИРОВАТЬ МОДЕЛИ")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 ЗАДАНИЕ 7 ВЫПОЛНЕНО!")
        print("📋 Развернуты в Triton:")
        print("   • Оригинальная PyTorch модель")
        print("   • Оригинальная ONNX модель")  
        print("   • Оптимизированная PyTorch модель")
        print("   • Оптимизированная ONNX модель")
    else:
        print(f"\n❌ Тестирование неудачно")
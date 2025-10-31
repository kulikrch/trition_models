"""
Тестирование всех моделей в Triton Inference Server
"""
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import numpy as np
import time
import sys
import os

class TritonTester:
    def __init__(self, triton_url="triton:8001", use_grpc=True):
        self.triton_url = triton_url
        self.use_grpc = use_grpc
        
        # Инициализация клиента
        if use_grpc:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
        else:
            http_url = triton_url.replace(':8001', ':8002')
            self.client = httpclient.InferenceServerClient(url=http_url)
        
        # CIFAR-10 классы
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def wait_for_triton(self, max_retries=30):
        """Ждет готовности Triton сервера"""
        print("⏳ Ожидание готовности Triton сервера...")
        
        for i in range(max_retries):
            try:
                if self.client.is_server_live():
                    if self.client.is_server_ready():
                        print("✅ Triton сервер готов!")
                        return True
                    else:
                        print(f"   Сервер запущен, но не готов... ({i+1}/{max_retries})")
                else:
                    print(f"   Сервер не отвечает... ({i+1}/{max_retries})")
            except Exception as e:
                print(f"   Ошибка подключения: {e} ({i+1}/{max_retries})")
            
            time.sleep(5)
        
        print("❌ Triton сервер не готов")
        return False
    
    def list_models(self):
        """Получает список доступных моделей"""
        try:
            models = self.client.get_model_repository_index()
            print("📋 Доступные модели:")
            
            available_models = []
            for model in models:
                name = model['name'] if isinstance(model, dict) else model.name
                state = model['state'] if isinstance(model, dict) else model.state
                print(f"   • {name}: {state}")
                
                if state == 'READY':
                    available_models.append(name)
            
            return available_models
            
        except Exception as e:
            print(f"❌ Ошибка получения списка моделей: {e}")
            return []
    
    def get_model_metadata(self, model_name):
        """Получает метаданные модели"""
        try:
            metadata = self.client.get_model_metadata(model_name)
            
            print(f"📊 Метаданные модели {model_name}:")
            if hasattr(metadata, 'platform'):
                print(f"   Platform: {metadata.platform}")
            if hasattr(metadata, 'inputs'):
                for inp in metadata.inputs:
                    print(f"   Input: {inp.name}, shape: {inp.shape}, dtype: {inp.datatype}")
            if hasattr(metadata, 'outputs'):
                for out in metadata.outputs:
                    print(f"   Output: {out.name}, shape: {out.shape}, dtype: {out.datatype}")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Ошибка получения метаданных: {e}")
            return None
    
    def create_test_data(self, batch_size=1):
        """Создает тестовые данные"""
        # Создаем случайные данные в формате CIFAR-10
        data = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
        
        # Нормализация как в обучении
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        data = (data - mean) / std
        
        return data
    
    def test_model_inference(self, model_name, num_tests=10):
        """Тестирует инференс модели"""
        print(f"🧪 Тестирование модели: {model_name}")
        
        # Получаем метаданные
        metadata = self.get_model_metadata(model_name)
        if not metadata:
            return None
        
        try:
            # Определяем имена входов и выходов
            if hasattr(metadata, 'inputs') and len(metadata.inputs) > 0:
                input_name = metadata.inputs[0].name
                input_shape = metadata.inputs[0].shape
            else:
                input_name = 'input'
                input_shape = [3, 32, 32]
            
            if hasattr(metadata, 'outputs') and len(metadata.outputs) > 0:
                output_name = metadata.outputs[0].name
            else:
                output_name = 'output'
            
            print(f"   Input: {input_name}, Output: {output_name}")
            
            # Создаем тестовые данные
            test_data = self.create_test_data(batch_size=1)
            
            # Подготавливаем запрос
            if self.use_grpc:
                inputs = [grpcclient.InferInput(input_name, test_data.shape, "FP32")]
                inputs[0].set_data_from_numpy(test_data)
                outputs = [grpcclient.InferRequestedOutput(output_name)]
            else:
                inputs = [httpclient.InferInput(input_name, test_data.shape, "FP32")]
                inputs[0].set_data_from_numpy(test_data)
                outputs = [httpclient.InferRequestedOutput(output_name)]
            
            # Прогрев
            for _ in range(3):
                try:
                    response = self.client.infer(model_name, inputs, outputs=outputs)
                except:
                    pass
            
            # Измерение производительности
            times = []
            results = []
            
            for i in range(num_tests):
                start_time = time.time()
                
                try:
                    response = self.client.infer(model_name, inputs, outputs=outputs)
                    end_time = time.time()
                    
                    # Получаем результат
                    if self.use_grpc:
                        output_data = response.as_numpy(output_name)
                    else:
                        output_data = response.as_numpy(output_name)
                    
                    times.append(end_time - start_time)
                    
                    # Анализируем предсказание
                    predicted_class = np.argmax(output_data[0])
                    confidence = np.max(np.softmax(output_data[0]))
                    results.append((predicted_class, confidence))
                    
                except Exception as e:
                    print(f"   ❌ Ошибка в тесте {i+1}: {e}")
                    continue
            
            if times:
                avg_time = np.mean(times) * 1000  # ms
                std_time = np.std(times) * 1000
                min_time = np.min(times) * 1000
                max_time = np.max(times) * 1000
                throughput = 1000 / avg_time  # fps
                
                # Последний результат для примера
                if results:
                    last_class, last_conf = results[-1]
                    class_name = self.class_names[last_class]
                
                print(f"   📊 Результаты ({len(times)} успешных тестов):")
                print(f"      Время: {avg_time:.2f} ± {std_time:.2f} ms")
                print(f"      Диапазон: {min_time:.2f} - {max_time:.2f} ms")
                print(f"      Throughput: {throughput:.1f} fps")
                if results:
                    print(f"      Пример: {class_name} ({last_conf:.3f})")
                
                return {
                    'model_name': model_name,
                    'successful_tests': len(times),
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'throughput_fps': throughput,
                    'sample_prediction': class_name if results else None,
                    'sample_confidence': last_conf if results else None
                }
            else:
                print(f"   ❌ Все тесты неудачны")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка тестирования модели {model_name}: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Запускает комплексное тестирование всех моделей"""
        print("🚀 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ TRITON INFERENCE SERVER")
        print("=" * 70)
        
        # Проверяем готовность сервера
        if not self.wait_for_triton():
            return False
        
        # Получаем список моделей
        available_models = self.list_models()
        if not available_models:
            print("❌ Нет доступных моделей")
            return False
        
        print(f"\n🎯 Найдено {len(available_models)} готовых моделей")
        
        # Тестируем каждую модель
        results = []
        
        for i, model_name in enumerate(available_models, 1):
            print(f"\n{i}️⃣ {'='*50}")
            print(f"{i}️⃣ ТЕСТИРОВАНИЕ: {model_name.upper()}")
            print(f"{i}️⃣ {'='*50}")
            
            result = self.test_model_inference(model_name, num_tests=20)
            if result:
                results.append(result)
        
        # Итоговая сводка
        if results:
            self.print_summary_table(results)
        
        return len(results) > 0
    
    def print_summary_table(self, results):
        """Печатает итоговую таблицу результатов"""
        print("\n🎉 " + "="*80 + " 🎉")
        print("🎉" + " "*25 + "ИТОГОВЫЕ РЕЗУЛЬТАТЫ TRITON" + " "*25 + "🎉")
        print("🎉 " + "="*80 + " 🎉")
        
        # Заголовок таблицы
        print(f"{'Модель':<25} {'Время (ms)':<12} {'Throughput':<12} {'Тестов':<8}")
        print("-" * 70)
        
        # Результаты
        for result in results:
            name = result['model_name']
            time_ms = f"{result['avg_time_ms']:.2f}"
            throughput = f"{result['throughput_fps']:.1f} fps"
            tests = result['successful_tests']
            
            print(f"{name:<25} {time_ms:<12} {throughput:<12} {tests:<8}")
        
        # Лучшие результаты
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print("-" * 30)
        
        fastest = min(results, key=lambda x: x['avg_time_ms'])
        highest_throughput = max(results, key=lambda x: x['throughput_fps'])
        
        print(f"🚀 Самая быстрая: {fastest['model_name']} ({fastest['avg_time_ms']:.2f} ms)")
        print(f"🚄 Наивысший throughput: {highest_throughput['model_name']} ({highest_throughput['throughput_fps']:.1f} fps)")

def main():
    """Главная функция"""
    triton_url = os.getenv('TRITON_URL', 'triton:8001')
    
    print(f"🔗 Подключение к Triton: {triton_url}")
    
    # Тестируем gRPC
    print(f"\n📡 ТЕСТИРОВАНИЕ gRPC ИНТЕРФЕЙСА")
    print("-" * 40)
    
    try:
        grpc_tester = TritonTester(triton_url, use_grpc=True)
        grpc_success = grpc_tester.run_comprehensive_test()
    except Exception as e:
        print(f"❌ gRPC тестирование неудачно: {e}")
        grpc_success = False
    
    # Тестируем HTTP (если gRPC не работает)
    if not grpc_success:
        print(f"\n🌐 ТЕСТИРОВАНИЕ HTTP ИНТЕРФЕЙСА")
        print("-" * 40)
        
        try:
            http_url = triton_url.replace(':8001', ':8002')
            http_tester = TritonTester(http_url, use_grpc=False)
            http_success = http_tester.run_comprehensive_test()
        except Exception as e:
            print(f"❌ HTTP тестирование неудачно: {e}")
            http_success = False
    else:
        http_success = True
    
    # Итоговый результат
    if grpc_success or http_success:
        print(f"\n✅ TRITON INFERENCE SERVER ПРОТЕСТИРОВАН УСПЕШНО!")
        return True
    else:
        print(f"\n❌ ТЕСТИРОВАНИЕ TRITON НЕУДАЧНО")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
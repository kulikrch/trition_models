"""
Клиент для тестирования инференса через Triton Inference Server
"""
import tritonclient.http as httpclient
import numpy as np
import time
import json
from PIL import Image
import requests
import io
import sys
import os


class TritonClient:
    def __init__(self, triton_url="triton:8000", preprocess_url="http://preprocess-service:8000"):
        self.triton_url = triton_url
        self.preprocess_url = preprocess_url
        self.client = httpclient.InferenceServerClient(url=triton_url)
        
        # CIFAR-10 классы
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
    def check_server_health(self):
        """Проверяет здоровье Triton сервера"""
        try:
            if self.client.is_server_live():
                print("✓ Triton server is live")
            else:
                print("✗ Triton server is not live")
                return False
                
            if self.client.is_server_ready():
                print("✓ Triton server is ready")
            else:
                print("✗ Triton server is not ready")
                return False
                
            return True
        except Exception as e:
            print(f"✗ Error connecting to Triton server: {e}")
            return False
    
    def check_preprocess_service(self):
        """Проверяет здоровье сервиса предобработки"""
        try:
            response = requests.get(f"{self.preprocess_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Preprocessing service is healthy")
                return True
            else:
                print(f"✗ Preprocessing service returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error connecting to preprocessing service: {e}")
            return False
    
    def list_models(self):
        """Выводит список доступных моделей"""
        try:
            models = self.client.get_model_repository_index()
            print("Available models:")
            for model in models:
                print(f"  - {model['name']} (state: {model['state']})")
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def get_model_metadata(self, model_name):
        """Получает метаданные модели"""
        try:
            metadata = self.client.get_model_metadata(model_name)
            print(f"Model {model_name} metadata:")
            print(f"  Platform: {metadata['platform']}")
            print(f"  Inputs: {metadata['inputs']}")
            print(f"  Outputs: {metadata['outputs']}")
            return metadata
        except Exception as e:
            print(f"Error getting model metadata: {e}")
            return None
    
    def create_test_image(self):
        """Создает тестовое изображение CIFAR-10"""
        # Создаем случайное изображение 32x32
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, 'RGB')
        return image
    
    def preprocess_image(self, image):
        """Предобрабатывает изображение через микросервис"""
        try:
            # Конвертируем изображение в bytes
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG')
            img_bytes = img_byte_array.getvalue()
            
            # Отправляем запрос к сервису предобработки
            files = {'file': ('test.png', img_bytes, 'image/png')}
            response = requests.post(f"{self.preprocess_url}/preprocess/single", files=files)
            
            if response.status_code == 200:
                data = response.json()
                processed_data = np.array(data['data'], dtype=np.float32)
                return processed_data
            else:
                print(f"Preprocessing failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def infer_single(self, model_name, input_data):
        """Выполняет инференс для одного изображения"""
        try:
            # Подготавливаем входные данные
            inputs = []
            inputs.append(httpclient.InferInput('input', input_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_data)
            
            # Подготавливаем выходные данные
            outputs = []
            outputs.append(httpclient.InferRequestedOutput('output'))
            
            # Выполняем инференс
            start_time = time.time()
            response = self.client.infer(model_name, inputs, outputs=outputs)
            inference_time = time.time() - start_time
            
            # Получаем результат
            output_data = response.as_numpy('output')
            
            return output_data, inference_time
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None
    
    def infer_batch(self, model_name, input_batch):
        """Выполняет инференс для батча изображений"""
        try:
            # Подготавливаем входные данные
            inputs = []
            inputs.append(httpclient.InferInput('input', input_batch.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_batch)
            
            # Подготавливаем выходные данные
            outputs = []
            outputs.append(httpclient.InferRequestedOutput('output'))
            
            # Выполняем инференс
            start_time = time.time()
            response = self.client.infer(model_name, inputs, outputs=outputs)
            inference_time = time.time() - start_time
            
            # Получаем результат
            output_data = response.as_numpy('output')
            
            return output_data, inference_time
            
        except Exception as e:
            print(f"Error during batch inference: {e}")
            return None, None
    
    def benchmark_model(self, model_name, num_requests=100, batch_size=1):
        """Бенчмарк производительности модели"""
        print(f"Benchmarking {model_name} with {num_requests} requests, batch_size={batch_size}")
        
        times = []
        successful_requests = 0
        
        for i in range(num_requests):
            # Создаем тестовые данные
            if batch_size == 1:
                test_image = self.create_test_image()
                input_data = self.preprocess_image(test_image)
                if input_data is None:
                    continue
                input_data = np.expand_dims(input_data, axis=0)  # Добавляем batch dimension
            else:
                batch_images = [self.create_test_image() for _ in range(batch_size)]
                input_batch = []
                for img in batch_images:
                    processed = self.preprocess_image(img)
                    if processed is not None:
                        input_batch.append(processed)
                
                if len(input_batch) != batch_size:
                    continue
                    
                input_data = np.stack(input_batch, axis=0)
            
            # Выполняем инференс
            if batch_size == 1:
                output, inference_time = self.infer_single(model_name, input_data)
            else:
                output, inference_time = self.infer_batch(model_name, input_data)
            
            if output is not None and inference_time is not None:
                times.append(inference_time)
                successful_requests += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        if times:
            avg_time = np.mean(times) * 1000  # в миллисекундах
            min_time = np.min(times) * 1000
            max_time = np.max(times) * 1000
            p95_time = np.percentile(times, 95) * 1000
            
            throughput = batch_size / np.mean(times)  # images per second
            
            print(f"Benchmark results for {model_name}:")
            print(f"  Successful requests: {successful_requests}/{num_requests}")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Min time: {min_time:.2f} ms")
            print(f"  Max time: {max_time:.2f} ms")
            print(f"  95th percentile: {p95_time:.2f} ms")
            print(f"  Throughput: {throughput:.2f} images/second")
            
            return {
                'successful_requests': successful_requests,
                'total_requests': num_requests,
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'p95_time_ms': p95_time,
                'throughput_fps': throughput
            }
        else:
            print(f"No successful requests for {model_name}")
            return None
    
    def test_classification(self, model_name):
        """Тестирует классификацию изображений"""
        print(f"Testing classification with {model_name}")
        
        # Создаем тестовое изображение
        test_image = self.create_test_image()
        input_data = self.preprocess_image(test_image)
        
        if input_data is None:
            print("Failed to preprocess image")
            return
        
        # Добавляем batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        # Выполняем инференс
        output, inference_time = self.infer_single(model_name, input_data)
        
        if output is not None:
            # Получаем предсказанный класс
            predicted_class_idx = np.argmax(output[0])
            confidence = np.softmax(output[0])[predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            print(f"Classification result:")
            print(f"  Predicted class: {predicted_class} (confidence: {confidence:.3f})")
            print(f"  Inference time: {inference_time*1000:.2f} ms")
            
            # Показываем топ-3 предсказания
            top3_indices = np.argsort(output[0])[-3:][::-1]
            print("  Top 3 predictions:")
            for i, idx in enumerate(top3_indices):
                conf = np.softmax(output[0])[idx]
                print(f"    {i+1}. {self.class_names[idx]}: {conf:.3f}")
        else:
            print("Inference failed")
    
    def run_full_test(self):
        """Запускает полный тест системы"""
        print("="*60)
        print("RUNNING FULL INFERENCE TEST")
        print("="*60)
        
        # Проверяем здоровье сервисов
        if not self.check_server_health():
            return False
        
        if not self.check_preprocess_service():
            return False
        
        # Показываем доступные модели
        models = self.list_models()
        if not models:
            print("No models available")
            return False
        
        # Тестируем каждую модель
        for model_info in models:
            model_name = model_info['name']
            
            if model_info['state'] != 'READY':
                print(f"Skipping {model_name} - not ready")
                continue
            
            print(f"\n{'='*40}")
            print(f"Testing model: {model_name}")
            print(f"{'='*40}")
            
            # Получаем метаданные
            self.get_model_metadata(model_name)
            
            # Тестируем классификацию
            self.test_classification(model_name)
            
            # Бенчмарк
            print(f"\nRunning benchmark for {model_name}...")
            self.benchmark_model(model_name, num_requests=50, batch_size=1)
            
            # Бенчмарк с батчем
            print(f"\nRunning batch benchmark for {model_name}...")
            self.benchmark_model(model_name, num_requests=20, batch_size=4)
        
        print("\n" + "="*60)
        print("FULL TEST COMPLETED")
        print("="*60)
        
        return True


def main():
    """Основная функция"""
    # Настройки из переменных окружения
    triton_url = os.getenv('TRITON_URL', 'triton:8000')
    preprocess_url = os.getenv('PREPROCESS_URL', 'http://preprocess-service:8000')
    
    print(f"Connecting to Triton at: {triton_url}")
    print(f"Connecting to preprocessing service at: {preprocess_url}")
    
    # Ждем запуска сервисов
    print("Waiting for services to start...")
    max_retries = 60
    
    for i in range(max_retries):
        try:
            client = TritonClient(triton_url, preprocess_url)
            if client.check_server_health() and client.check_preprocess_service():
                print("All services are ready!")
                break
        except:
            pass
        
        time.sleep(5)
        print(f"Retrying... ({i+1}/{max_retries})")
    else:
        print("Services are not responding after maximum retries")
        sys.exit(1)
    
    # Запускаем тесты
    success = client.run_full_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
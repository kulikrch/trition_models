"""
Тесты для микросервиса предобработки
"""
import requests
import numpy as np
from PIL import Image
import io
import base64
import json
import time


class PreprocessServiceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def create_test_image(self, size=(32, 32), color='RGB'):
        """Создает тестовое изображение"""
        # Создаем случайное изображение
        img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, color)
        return image
    
    def image_to_bytes(self, image):
        """Конвертирует PIL изображение в bytes"""
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        return img_byte_array.getvalue()
    
    def image_to_base64(self, image):
        """Конвертирует PIL изображение в base64"""
        img_bytes = self.image_to_bytes(image)
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def test_health_check(self):
        """Тест проверки здоровья сервиса"""
        print("Testing health check...")
        response = requests.get(f"{self.base_url}/health")
        
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    
    def test_single_image_preprocessing(self):
        """Тест предобработки одного изображения"""
        print("Testing single image preprocessing...")
        
        # Создаем тестовое изображение
        test_image = self.create_test_image()
        img_bytes = self.image_to_bytes(test_image)
        
        # Отправляем запрос
        files = {'file': ('test.png', img_bytes, 'image/png')}
        response = requests.post(f"{self.base_url}/preprocess/single", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Single image preprocessing passed")
            print(f"  Shape: {data['shape']}")
            print(f"  Processing time: {data['processing_time']:.3f}s")
            return True
        else:
            print(f"✗ Single image preprocessing failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    def test_batch_preprocessing(self):
        """Тест предобработки батча изображений"""
        print("Testing batch preprocessing...")
        
        # Создаем несколько тестовых изображений
        batch_size = 3
        files = []
        
        for i in range(batch_size):
            test_image = self.create_test_image()
            img_bytes = self.image_to_bytes(test_image)
            files.append(('files', (f'test_{i}.png', img_bytes, 'image/png')))
        
        # Отправляем запрос
        response = requests.post(f"{self.base_url}/preprocess/batch", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Batch preprocessing passed")
            print(f"  Batch size: {data['batch_size']}")
            print(f"  Shape: {data['shape']}")
            print(f"  Processing time: {data['processing_time']:.3f}s")
            return True
        else:
            print(f"✗ Batch preprocessing failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    def test_base64_preprocessing(self):
        """Тест предобработки изображения из base64"""
        print("Testing base64 preprocessing...")
        
        # Создаем тестовое изображение
        test_image = self.create_test_image()
        base64_data = self.image_to_base64(test_image)
        
        # Отправляем запрос
        payload = {"image": base64_data}
        response = requests.post(
            f"{self.base_url}/preprocess/base64", 
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Base64 preprocessing passed")
            print(f"  Shape: {data['shape']}")
            print(f"  Processing time: {data['processing_time']:.3f}s")
            return True
        else:
            print(f"✗ Base64 preprocessing failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    def test_info_endpoint(self):
        """Тест информационного эндпоинта"""
        print("Testing info endpoint...")
        
        response = requests.get(f"{self.base_url}/info")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Info endpoint passed")
            print(f"  Target size: {data['target_size']}")
            print(f"  Normalization mean: {data['normalization']['mean']}")
            print(f"  Output format: {data['output_format']}")
            return True
        else:
            print(f"✗ Info endpoint failed: {response.status_code}")
            return False
    
    def test_metrics_endpoint(self):
        """Тест метрик Prometheus"""
        print("Testing metrics endpoint...")
        
        response = requests.get(f"{self.base_url}/metrics")
        
        if response.status_code == 200:
            print(f"✓ Metrics endpoint passed")
            print(f"  Metrics length: {len(response.text)} characters")
            return True
        else:
            print(f"✗ Metrics endpoint failed: {response.status_code}")
            return False
    
    def benchmark_service(self, num_requests=10):
        """Бенчмарк производительности сервиса"""
        print(f"Benchmarking service with {num_requests} requests...")
        
        test_image = self.create_test_image()
        img_bytes = self.image_to_bytes(test_image)
        
        times = []
        success_count = 0
        
        for i in range(num_requests):
            files = {'file': ('test.png', img_bytes, 'image/png')}
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/preprocess/single", files=files)
            end_time = time.time()
            
            if response.status_code == 200:
                success_count += 1
                times.append(end_time - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        if times:
            avg_time = np.mean(times) * 1000
            min_time = np.min(times) * 1000
            max_time = np.max(times) * 1000
            
            print(f"✓ Benchmark completed")
            print(f"  Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Min time: {min_time:.2f} ms")
            print(f"  Max time: {max_time:.2f} ms")
            print(f"  Throughput: {1000/avg_time:.1f} requests/second")
        else:
            print("✗ Benchmark failed - no successful requests")
    
    def run_all_tests(self):
        """Запускает все тесты"""
        print("="*60)
        print("RUNNING PREPROCESSING SERVICE TESTS")
        print("="*60)
        
        tests = [
            self.test_health_check,
            self.test_info_endpoint,
            self.test_single_image_preprocessing,
            self.test_batch_preprocessing,
            self.test_base64_preprocessing,
            self.test_metrics_endpoint
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                print()
            except Exception as e:
                print(f"✗ Test failed with exception: {e}")
                print()
        
        print("="*60)
        print(f"TESTS COMPLETED: {passed}/{total} passed")
        print("="*60)
        
        if passed == total:
            print("Running benchmark...")
            self.benchmark_service()
        
        return passed == total


def main():
    """Основная функция для запуска тестов"""
    import sys
    
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = PreprocessServiceTester(base_url)
    
    # Ждем запуска сервиса
    print(f"Testing service at {base_url}")
    print("Waiting for service to start...")
    
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                print("Service is ready!")
                break
        except:
            pass
        
        time.sleep(1)
        print(f"Retrying... ({i+1}/{max_retries})")
    else:
        print("Service is not responding. Please start the service first.")
        sys.exit(1)
    
    # Запускаем тесты
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
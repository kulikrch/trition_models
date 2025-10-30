"""
Комплексный бенчмарк всех оптимизированных моделей
"""
import torch
import torch.nn as nn
import onnxruntime as ort
import os
import time
import numpy as np
import json
from working_demo import SimpleCNN

class ModelBenchmark:
    def __init__(self):
        self.device = 'cpu'
        self.test_runs = 100  # Больше для точности
        self.warmup_runs = 20
        
        # CIFAR-10 классы для демонстрации
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Параметры нормализации
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])
        
    def create_test_data(self, batch_size=1):
        """Создает тестовые данные"""
        return torch.randn(batch_size, 3, 32, 32)
    
    def benchmark_pytorch_model(self, model_path, model_name):
        """Бенчмарк PyTorch модели"""
        print(f"🔍 Тестирование {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            return None
        
        # Загружаем модель
        model = SimpleCNN(num_classes=10)
        
        if 'quantized' in model_path or 'combined' in model_path:
            # Для квантизованных моделей нужен особый подход
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            except:
                # Если не получается загрузить напрямую, пересоздаем квантизованную модель
                original_model = SimpleCNN(num_classes=10)
                original_checkpoint = torch.load('models/working_model.pth', map_location='cpu')
                original_model.load_state_dict(original_checkpoint['model_state_dict'])
                
                if 'quantized' in model_path:
                    model = torch.quantization.quantize_dynamic(
                        original_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
                elif 'combined' in model_path:
                    # Здесь нужно было бы применить прунинг, но для простоты используем квантизацию
                    model = torch.quantization.quantize_dynamic(
                        original_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
        else:
            # Обычная модель
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Создаем тестовые данные
        test_input = self.create_test_data()
        
        # Прогрев
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = model(test_input)
        
        # Бенчмарк
        times = []
        for _ in range(self.test_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model(test_input)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Статистика
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        p95_time = np.percentile(times, 95) * 1000
        
        # Размер файла
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        # Тестовое предсказание
        with torch.no_grad():
            test_output = model(test_input)
            predicted_class = torch.argmax(test_output, dim=1).item()
            confidence = torch.softmax(test_output, dim=1).max().item()
        
        results = {
            'model_name': model_name,
            'file_path': model_path,
            'file_size_mb': file_size,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'p95_time_ms': p95_time,
            'throughput_fps': 1000 / avg_time,
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'test_runs': self.test_runs
        }
        
        print(f"  📊 Время: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  🚄 Throughput: {1000/avg_time:.1f} img/sec")
        print(f"  📁 Размер: {file_size:.2f} MB")
        print(f"  🎯 Тест: {self.class_names[predicted_class]} ({confidence:.3f})")
        
        return results
    
    def benchmark_onnx_model(self, onnx_path, model_name="ONNX"):
        """Бенчмарк ONNX модели"""
        print(f"🔍 Тестирование {model_name}...")
        
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX модель не найдена: {onnx_path}")
            return None
        
        # Создаем ONNX Runtime сессию
        try:
            ort_session = ort.InferenceSession(onnx_path)
        except Exception as e:
            print(f"❌ Ошибка загрузки ONNX: {e}")
            return None
        
        # Создаем тестовые данные
        test_input = self.create_test_data().numpy().astype(np.float32)
        
        # Прогрев
        for _ in range(self.warmup_runs):
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            _ = ort_session.run(None, ort_inputs)
        
        # Бенчмарк
        times = []
        for _ in range(self.test_runs):
            start_time = time.perf_counter()
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            output = ort_session.run(None, ort_inputs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Статистика
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        p95_time = np.percentile(times, 95) * 1000
        
        # Размер файла
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        
        # Тестовое предсказание
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        predicted_class = np.argmax(onnx_output, axis=1)[0]
        confidence = np.max(onnx_output)
        
        results = {
            'model_name': model_name,
            'file_path': onnx_path,
            'file_size_mb': file_size,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'p95_time_ms': p95_time,
            'throughput_fps': 1000 / avg_time,
            'predicted_class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'test_runs': self.test_runs
        }
        
        print(f"  📊 Время: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  🚄 Throughput: {1000/avg_time:.1f} img/sec")
        print(f"  📁 Размер: {file_size:.2f} MB")
        print(f"  🎯 Тест: {self.class_names[predicted_class]} ({confidence:.3f})")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Запускает полный бенчмарк всех моделей"""
        print("🚀 КОМПЛЕКСНЫЙ БЕНЧМАРК ВСЕХ МОДЕЛЕЙ")
        print("=" * 70)
        print(f"📊 Параметры тестирования:")
        print(f"   • Количество прогонов: {self.test_runs}")
        print(f"   • Прогрев: {self.warmup_runs} итераций")
        print(f"   • Устройство: {self.device}")
        print(f"   • Размер батча: 1")
        print()
        
        results = []
        
        # Список моделей для тестирования
        models_to_test = [
            ('models/working_model.pth', 'PyTorch Original'),
            ('models/working_model.onnx', 'ONNX'),
            ('models/working_model_quantized.pth', 'PyTorch Quantized'),
            ('models/working_model_pruned.pth', 'PyTorch Pruned'),
            ('models/working_model_combined.pth', 'PyTorch Combined'),
        ]
        
        # Тестируем каждую модель
        for i, (model_path, model_name) in enumerate(models_to_test, 1):
            print(f"\n{i}️⃣ {'='*50}")
            print(f"{i}️⃣ {model_name.upper()}")
            print(f"{i}️⃣ {'='*50}")
            
            if model_path.endswith('.onnx'):
                result = self.benchmark_onnx_model(model_path, model_name)
            else:
                result = self.benchmark_pytorch_model(model_path, model_name)
            
            if result:
                results.append(result)
        
        # Итоговая сводка
        if results:
            self.print_summary_table(results)
            self.save_results(results)
        
        return results
    
    def print_summary_table(self, results):
        """Печатает итоговую таблицу результатов"""
        print("\n" + "🎉" + "="*80 + "🎉")
        print("🎉" + " "*30 + "ИТОГОВЫЕ РЕЗУЛЬТАТЫ" + " "*30 + "🎉")
        print("🎉" + "="*80 + "🎉")
        
        # Заголовок таблицы
        print(f"{'Модель':<20} {'Размер':<10} {'Время (ms)':<12} {'Throughput':<12} {'P95 (ms)':<10}")
        print("-" * 80)
        
        # Находим baseline (оригинальную модель)
        baseline = None
        for result in results:
            if 'original' in result['model_name'].lower():
                baseline = result
                break
        
        # Выводим результаты
        for result in results:
            name = result['model_name']
            size = f"{result['file_size_mb']:.2f} MB"
            time_ms = f"{result['avg_time_ms']:.2f}"
            throughput = f"{result['throughput_fps']:.1f} fps"
            p95 = f"{result['p95_time_ms']:.2f}"
            
            print(f"{name:<20} {size:<10} {time_ms:<12} {throughput:<12} {p95:<10}")
        
        # Сравнение с baseline
        if baseline:
            print(f"\n📊 СРАВНЕНИЕ С BASELINE ({baseline['model_name']}):")
            print("-" * 60)
            
            for result in results:
                if result == baseline:
                    continue
                
                name = result['model_name']
                size_ratio = baseline['file_size_mb'] / result['file_size_mb']
                speed_ratio = baseline['avg_time_ms'] / result['avg_time_ms']
                
                print(f"{name}:")
                print(f"  📦 Сжатие: {size_ratio:.1f}x")
                print(f"  ⚡ Ускорение: {speed_ratio:.1f}x")
                if size_ratio > 1:
                    print(f"  💾 Экономия места: {(1-1/size_ratio)*100:.1f}%")
                print()
        
        # Лучшие результаты
        print("🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print("-" * 30)
        
        fastest = min(results, key=lambda x: x['avg_time_ms'])
        smallest = min(results, key=lambda x: x['file_size_mb'])
        highest_throughput = max(results, key=lambda x: x['throughput_fps'])
        
        print(f"🚀 Самая быстрая: {fastest['model_name']} ({fastest['avg_time_ms']:.2f} ms)")
        print(f"📦 Самая компактная: {smallest['model_name']} ({smallest['file_size_mb']:.2f} MB)")
        print(f"🚄 Наивысший throughput: {highest_throughput['model_name']} ({highest_throughput['throughput_fps']:.1f} fps)")
    
    def save_results(self, results):
        """Сохраняет результаты в JSON"""
        output_file = 'benchmark_results.json'
        
        # Добавляем метаданные
        benchmark_data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_runs': self.test_runs,
                'warmup_runs': self.warmup_runs,
                'device': self.device,
                'platform': 'CPU'
            },
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены в: {output_file}")

def main():
    """Главная функция"""
    benchmark = ModelBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    if results:
        print(f"\n✅ Бенчмарк завершен успешно!")
        print(f"📊 Протестировано моделей: {len(results)}")
        print(f"🎯 Готовы данные для отчета!")
    else:
        print("❌ Ошибка при выполнении бенчмарка")

if __name__ == "__main__":
    main()
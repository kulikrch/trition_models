"""
Скрипт для конвертации PyTorch модели в ONNX формат
"""
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys

# Добавляем путь к модели
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import get_model


class ONNXConverter:
    def __init__(self, model_path, output_dir='../../models'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = 'cpu'  # Принудительно используем CPU
        
    def load_pytorch_model(self):
        """Загружает обученную PyTorch модель"""
        print(f"Loading PyTorch model from {self.model_path}")
        
        # Создаем модель
        model = get_model(num_classes=10)
        
        # Загружаем веса
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully. Test accuracy: {checkpoint.get('test_acc', 'Unknown'):.2f}%")
        return model
    
    def convert_to_onnx(self, model, onnx_path, input_shape=(1, 3, 32, 32)):
        """Конвертирует PyTorch модель в ONNX"""
        print(f"Converting model to ONNX format...")
        
        # Создаем dummy input для экспорта
        dummy_input = torch.randn(input_shape)
        
        # Экспорт в ONNX
        torch.onnx.export(
            model,                      # модель
            dummy_input,               # пример входных данных
            onnx_path,                 # путь для сохранения
            export_params=True,        # сохранять параметры модели
            opset_version=11,          # версия ONNX opset
            do_constant_folding=True,  # оптимизация констант
            input_names=['input'],     # имя входного тензора
            output_names=['output'],   # имя выходного тензора
            dynamic_axes={             # динамические размерности
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model successfully exported to {onnx_path}")
        
    def verify_onnx_model(self, onnx_path, pytorch_model):
        """Проверяет корректность ONNX модели"""
        print("Verifying ONNX model...")
        
        # Загружаем ONNX модель
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")
        
        # Создаем ONNX Runtime сессию
        ort_session = ort.InferenceSession(onnx_path)
        
        # Тестовые данные
        test_input = torch.randn(1, 3, 32, 32)
        
        # PyTorch предсказание
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX предсказание
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Сравнение результатов
        np.testing.assert_allclose(pytorch_output, ort_output, rtol=1e-03, atol=1e-05)
        print("ONNX model verification successful! Outputs match PyTorch model.")
        
        return ort_session
    
    def benchmark_models(self, pytorch_model, onnx_path, num_runs=100):
        """Сравнение производительности PyTorch и ONNX моделей"""
        print(f"Benchmarking models over {num_runs} runs...")
        
        # Подготовка данных
        test_input = torch.randn(1, 3, 32, 32)
        
        # Benchmark PyTorch
        pytorch_model.eval()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        # Прогрев
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(test_input)
        
        # Измерение времени PyTorch
        import time
        pytorch_times = []
        
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = pytorch_model(test_input)
            end = time.time()
            pytorch_times.append(end - start)
        
        pytorch_avg_time = np.mean(pytorch_times) * 1000  # в миллисекундах
        
        # Benchmark ONNX
        ort_session = ort.InferenceSession(onnx_path)
        ort_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
        
        # Прогрев
        for _ in range(10):
            _ = ort_session.run(None, ort_input)
        
        # Измерение времени ONNX
        onnx_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = ort_session.run(None, ort_input)
            end = time.time()
            onnx_times.append(end - start)
        
        onnx_avg_time = np.mean(onnx_times) * 1000  # в миллисекундах
        
        speedup = pytorch_avg_time / onnx_avg_time
        
        print(f"PyTorch average inference time: {pytorch_avg_time:.2f} ms")
        print(f"ONNX average inference time: {onnx_avg_time:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        return {
            'pytorch_time': pytorch_avg_time,
            'onnx_time': onnx_avg_time,
            'speedup': speedup
        }
    
    def convert(self):
        """Полный процесс конвертации"""
        # Создаем выходную директорию
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Загружаем PyTorch модель
        pytorch_model = self.load_pytorch_model()
        
        # Путь для ONNX модели
        onnx_path = os.path.join(self.output_dir, 'cifar10_model.onnx')
        
        # Конвертируем в ONNX
        self.convert_to_onnx(pytorch_model, onnx_path)
        
        # Проверяем корректность
        self.verify_onnx_model(onnx_path, pytorch_model)
        
        # Бенчмарк
        benchmark_results = self.benchmark_models(pytorch_model, onnx_path)
        
        print(f"\nConversion completed successfully!")
        print(f"ONNX model saved to: {onnx_path}")
        
        return {
            'onnx_path': onnx_path,
            'benchmark': benchmark_results
        }


def main():
    """Основная функция"""
    model_path = '../../models/cifar10_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first using src/train/train.py")
        return
    
    converter = ONNXConverter(model_path)
    results = converter.convert()
    
    print("\nConversion results:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
"""
Быстрая оптимизация существующей модели
"""
import torch
import os
import time
import numpy as np
from working_demo import SimpleCNN

def quick_optimize():
    print("🔍 БЫСТРАЯ ОПТИМИЗАЦИЯ МОДЕЛИ")
    print("=" * 50)
    
    # Находим модель
    model_paths = ['models/working_model.pth', 'models/cifar10_model.pth']
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Модель не найдена!")
        return
    
    print(f"✅ Используем модель: {model_path}")
    
    # Загружаем модель
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    orig_acc = checkpoint.get('test_acc', 'Unknown')
    print(f"📊 Исходная точность: {orig_acc}")
    
    # Измеряем исходную производительность
    print("\n⏱️ ИЗМЕРЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("-" * 30)
    
    test_input = torch.randn(1, 3, 32, 32)
    
    # Прогрев
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    # Измерение оригинальной модели
    times = []
    for _ in range(50):
        start = time.time()
        with torch.no_grad():
            _ = model(test_input)
        times.append(time.time() - start)
    
    orig_time = np.mean(times) * 1000
    print(f"🔵 Оригинальная модель: {orig_time:.2f} ms")
    
    # Квантизация
    print("\n🔄 ПРИМЕНЕНИЕ КВАНТИЗАЦИИ:")
    print("-" * 30)
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # Измерение квантизованной модели
    times = []
    for _ in range(50):
        start = time.time()
        with torch.no_grad():
            _ = quantized_model(test_input)
        times.append(time.time() - start)
    
    quant_time = np.mean(times) * 1000
    print(f"🟢 Квантизованная модель: {quant_time:.2f} ms")
    
    # Сохранение
    quant_path = model_path.replace('.pth', '_quantized.pth')
    torch.save(quantized_model.state_dict(), quant_path)
    
    # Сравнение размеров
    orig_size = os.path.getsize(model_path) / (1024*1024)
    quant_size = os.path.getsize(quant_path) / (1024*1024)
    
    print("\n📊 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 50)
    print(f"📁 Размер:")
    print(f"   Оригинал:     {orig_size:.2f} MB")
    print(f"   Квантизованная: {quant_size:.2f} MB")
    print(f"   Сжатие:       {orig_size/quant_size:.1f}x")
    print(f"   Экономия:     {(1-quant_size/orig_size)*100:.1f}%")
    
    print(f"\n⚡ Скорость:")
    print(f"   Оригинал:     {orig_time:.2f} ms")
    print(f"   Квантизованная: {quant_time:.2f} ms")
    print(f"   Ускорение:    {orig_time/quant_time:.1f}x")
    print(f"   Throughput:   {1000/quant_time:.1f} img/sec")
    
    print(f"\n✅ Квантизованная модель сохранена: {quant_path}")
    
    # Проверим точность
    print("\n🧪 ТЕСТ ТОЧНОСТИ:")
    print("-" * 20)
    
    # Создаем тестовые данные
    test_images = torch.randn(100, 3, 32, 32)
    
    # Предсказания оригинальной модели
    with torch.no_grad():
        orig_outputs = model(test_images)
        orig_preds = torch.argmax(orig_outputs, dim=1)
    
    # Предсказания квантизованной модели
    with torch.no_grad():
        quant_outputs = quantized_model(test_images)
        quant_preds = torch.argmax(quant_outputs, dim=1)
    
    # Сравнение предсказаний
    agreement = (orig_preds == quant_preds).float().mean().item() * 100
    print(f"🎯 Совпадение предсказаний: {agreement:.1f}%")
    
    # Разница в выходах
    output_diff = torch.abs(orig_outputs - quant_outputs).mean().item()
    print(f"📏 Средняя разница выходов: {output_diff:.6f}")
    
    print("\n🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    
    return {
        'orig_size': orig_size,
        'quant_size': quant_size,
        'orig_time': orig_time,
        'quant_time': quant_time,
        'speedup': orig_time/quant_time,
        'compression': orig_size/quant_size,
        'agreement': agreement
    }

if __name__ == "__main__":
    results = quick_optimize()
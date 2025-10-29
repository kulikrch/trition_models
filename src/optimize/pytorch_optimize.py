"""
Оптимизация PyTorch модели с использованием quantization и pruning
"""
import torch
import torch.nn as nn
import torch.quantization
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import copy
import time
import numpy as np

# Добавляем путь к модели
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import get_model


class PyTorchOptimizer:
    def __init__(self, model_path, output_dir='../../models'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = 'cpu'  # Quantization работает только на CPU
        
    def load_model(self):
        """Загружает обученную модель"""
        print(f"Loading model from {self.model_path}")
        
        model = get_model(num_classes=10)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded. Original accuracy: {checkpoint.get('test_acc', 'Unknown'):.2f}%")
        return model
    
    def get_test_loader(self, batch_size=128):
        """Создает test data loader для оценки"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    def evaluate_model(self, model, test_loader, model_name="Model"):
        """Оценивает точность модели"""
        model.eval()
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Измеряем время инференса
                start_time = time.time()
                output = model(data)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_inference_time = np.mean(inference_times) * 1000  # в миллисекундах
        
        print(f"{model_name} - Accuracy: {accuracy:.2f}%, "
              f"Avg inference time: {avg_inference_time:.2f} ms")
        
        return accuracy, avg_inference_time
    
    def quantize_model(self, model):
        """Применяет динамическую квантизацию"""
        print("Applying dynamic quantization...")
        
        # Создаем копию модели
        quantized_model = copy.deepcopy(model)
        
        # Применяем динамическую квантизацию
        quantized_model = torch.quantization.quantize_dynamic(
            quantized_model,
            {nn.Linear, nn.Conv2d},  # Слои для квантизации
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def prune_model(self, model, pruning_amount=0.3):
        """Применяет структурированное прунинг"""
        print(f"Applying pruning with {pruning_amount*100}% sparsity...")
        
        # Создаем копию модели
        pruned_model = copy.deepcopy(model)
        
        # Применяем глобальное неструктурированное прунинг
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Глобальное прунинг
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )
        
        # Удаляем маски прунинга (делаем прунинг постоянным)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return pruned_model
    
    def combine_optimizations(self, model):
        """Комбинирует прунинг и квантизацию"""
        print("Applying combined optimizations (pruning + quantization)...")
        
        # Сначала прунинг
        pruned_model = self.prune_model(model, pruning_amount=0.3)
        
        # Затем квантизация
        combined_model = self.quantize_model(pruned_model)
        
        return combined_model
    
    def get_model_size(self, model, model_name="Model"):
        """Вычисляет размер модели"""
        # Сохраняем модель во временный файл
        temp_path = f"tmp_rovodev_{model_name.lower().replace(' ', '_')}.pth"
        torch.save(model.state_dict(), temp_path)
        
        # Получаем размер файла
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        # Удаляем временный файл
        os.remove(temp_path)
        
        print(f"{model_name} size: {size_mb:.2f} MB")
        return size_mb
    
    def optimize(self):
        """Полный процесс оптимизации"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Загружаем оригинальную модель
        original_model = self.load_model()
        test_loader = self.get_test_loader()
        
        # Оценка оригинальной модели
        print("\n" + "="*50)
        print("EVALUATING ORIGINAL MODEL")
        print("="*50)
        original_acc, original_time = self.evaluate_model(original_model, test_loader, "Original Model")
        original_size = self.get_model_size(original_model, "Original Model")
        
        results = {
            'original': {
                'accuracy': original_acc,
                'inference_time': original_time,
                'size_mb': original_size
            }
        }
        
        # Квантизация
        print("\n" + "="*50)
        print("APPLYING QUANTIZATION")
        print("="*50)
        quantized_model = self.quantize_model(original_model)
        quantized_acc, quantized_time = self.evaluate_model(quantized_model, test_loader, "Quantized Model")
        quantized_size = self.get_model_size(quantized_model, "Quantized Model")
        
        # Сохраняем квантизованную модель
        quantized_path = os.path.join(self.output_dir, 'cifar10_model_quantized.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        
        results['quantized'] = {
            'accuracy': quantized_acc,
            'inference_time': quantized_time,
            'size_mb': quantized_size,
            'path': quantized_path
        }
        
        # Прунинг
        print("\n" + "="*50)
        print("APPLYING PRUNING")
        print("="*50)
        pruned_model = self.prune_model(original_model)
        pruned_acc, pruned_time = self.evaluate_model(pruned_model, test_loader, "Pruned Model")
        pruned_size = self.get_model_size(pruned_model, "Pruned Model")
        
        # Сохраняем модель с прунингом
        pruned_path = os.path.join(self.output_dir, 'cifar10_model_pruned.pth')
        torch.save(pruned_model.state_dict(), pruned_path)
        
        results['pruned'] = {
            'accuracy': pruned_acc,
            'inference_time': pruned_time,
            'size_mb': pruned_size,
            'path': pruned_path
        }
        
        # Комбинированная оптимизация
        print("\n" + "="*50)
        print("APPLYING COMBINED OPTIMIZATIONS")
        print("="*50)
        combined_model = self.combine_optimizations(original_model)
        combined_acc, combined_time = self.evaluate_model(combined_model, test_loader, "Combined Model")
        combined_size = self.get_model_size(combined_model, "Combined Model")
        
        # Сохраняем комбинированную модель
        combined_path = os.path.join(self.output_dir, 'cifar10_model_combined.pth')
        torch.save(combined_model.state_dict(), combined_path)
        
        results['combined'] = {
            'accuracy': combined_acc,
            'inference_time': combined_time,
            'size_mb': combined_size,
            'path': combined_path
        }
        
        # Сводка результатов
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"{'Model':<15} {'Accuracy':<12} {'Time (ms)':<12} {'Size (MB)':<12} {'Speedup':<10}")
        print("-" * 60)
        
        for name, stats in results.items():
            speedup = original_time / stats['inference_time']
            compression = original_size / stats['size_mb']
            print(f"{name:<15} {stats['accuracy']:<12.2f} {stats['inference_time']:<12.2f} "
                  f"{stats['size_mb']:<12.2f} {speedup:<10.2f}")
        
        return results


def main():
    """Основная функция"""
    model_path = '../../models/cifar10_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first using src/train/train.py")
        return
    
    optimizer = PyTorchOptimizer(model_path)
    results = optimizer.optimize()
    
    return results


if __name__ == "__main__":
    main()
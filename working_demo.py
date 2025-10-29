"""
Рабочая демонстрация ML пайплайна с совместимыми моделями
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image
import json
import time

# Простая модель (совместимая с обученной)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLPipeline:
    def __init__(self):
        self.model = None
        self.device = 'cpu'
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Параметры нормализации CIFAR-10
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])
    
    def create_and_train_model(self, epochs=3):
        """Создает и обучает модель"""
        print(f"🏋️ Training model for {epochs} epochs...")
        
        # Создаем модель
        self.model = SimpleCNN(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Создаем тестовые данные
        print("📊 Creating training data...")
        X_train = torch.randn(1000, 3, 32, 32)
        y_train = torch.randint(0, 10, (1000,))
        
        X_test = torch.randn(200, 3, 32, 32)
        y_test = torch.randint(0, 10, (200,))
        
        # Обучение
        self.model.train()
        for epoch in range(epochs):
            batch_size = 32
            total_loss = 0
            num_batches = len(X_train) // batch_size
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            
            # Тестирование
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test).float().mean().item() * 100
            
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            self.model.train()
        
        # Сохранение модели
        os.makedirs("models", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'test_acc': accuracy,
            'test_loss': avg_loss,
        }, "models/working_model.pth")
        
        print(f"✅ Model saved! Final accuracy: {accuracy:.2f}%")
        return accuracy
    
    def load_model(self, model_path="models/working_model.pth"):
        """Загружает обученную модель"""
        print(f"📂 Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return False
        
        self.model = SimpleCNN(num_classes=10)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded! Test accuracy: {checkpoint.get('test_acc', 'Unknown'):.2f}%")
        return True
    
    def convert_to_onnx(self, onnx_path="models/working_model.onnx"):
        """Конвертирует модель в ONNX"""
        print("🔄 Converting to ONNX...")
        
        if self.model is None:
            print("❌ Model not loaded!")
            return False
        
        try:
            dummy_input = torch.randn(1, 3, 32, 32)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            print(f"✅ ONNX model saved to {onnx_path}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX conversion failed: {e}")
            return False
    
    def test_onnx_inference(self, onnx_path="models/working_model.onnx"):
        """Тестирует ONNX инференс"""
        print("🧪 Testing ONNX inference...")
        
        try:
            import onnxruntime as ort
            
            # Загружаем ONNX модель
            ort_session = ort.InferenceSession(onnx_path)
            
            # Тестовые данные
            test_input = torch.randn(1, 3, 32, 32)
            
            # ONNX инференс
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            # PyTorch инференс для сравнения
            if self.model:
                with torch.no_grad():
                    pytorch_output = self.model(test_input).numpy()
                
                # Сравниваем результаты
                diff = np.max(np.abs(pytorch_output - ort_output))
                print(f"✅ ONNX inference works! Max difference: {diff:.6f}")
            else:
                print(f"✅ ONNX inference works! Output shape: {ort_output.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            return False
    
    def preprocess_image(self, image):
        """Предобрабатывает изображение"""
        if isinstance(image, str):
            # Если это путь к файлу
            image = Image.open(image)
        
        # Конвертируем в RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Изменяем размер
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Конвертируем в numpy и нормализуем
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return torch.from_numpy(img_array).unsqueeze(0).float()
    
    def predict_image(self, image):
        """Предсказывает класс изображения"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        # Предобработка
        input_tensor = self.preprocess_image(image)
        
        # Инференс
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class_id': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }
    
    def benchmark_inference(self, num_samples=100):
        """Бенчмарк производительности"""
        print(f"⚡ Benchmarking inference ({num_samples} samples)...")
        
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        # Создаем тестовые данные
        test_data = torch.randn(num_samples, 3, 32, 32)
        
        # Прогрев
        with torch.no_grad():
            _ = self.model(test_data[:5])
        
        # Измерение времени
        times = []
        for i in range(num_samples):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(test_data[i:i+1])
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # в миллисекундах
        throughput = 1000 / avg_time  # изображений в секунду
        
        print(f"✅ Average inference time: {avg_time:.2f} ms")
        print(f"✅ Throughput: {throughput:.1f} images/second")
        
        return {
            'avg_time_ms': avg_time,
            'throughput_fps': throughput,
            'times': times
        }
    
    def create_test_image(self):
        """Создает тестовое изображение"""
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        return Image.fromarray(img_array, 'RGB')
    
    def run_complete_demo(self):
        """Запускает полную демонстрацию"""
        print("🚀 " + "="*60)
        print("🚀 COMPLETE ML PIPELINE DEMONSTRATION")
        print("🚀 " + "="*60)
        
        results = {}
        
        # 1. Обучение модели
        print("\n1️⃣ TRAINING MODEL")
        print("-" * 40)
        accuracy = self.create_and_train_model(epochs=3)
        results['training_accuracy'] = accuracy
        
        # 2. Конвертация в ONNX
        print("\n2️⃣ CONVERTING TO ONNX")
        print("-" * 40)
        onnx_success = self.convert_to_onnx()
        results['onnx_conversion'] = onnx_success
        
        # 3. Тестирование ONNX
        if onnx_success:
            print("\n3️⃣ TESTING ONNX INFERENCE")
            print("-" * 40)
            onnx_test = self.test_onnx_inference()
            results['onnx_inference'] = onnx_test
        
        # 4. Тестирование предобработки
        print("\n4️⃣ TESTING IMAGE PREPROCESSING")
        print("-" * 40)
        test_image = self.create_test_image()
        prediction = self.predict_image(test_image)
        if prediction:
            print(f"✅ Prediction: {prediction['class_name']} ({prediction['confidence']:.3f})")
            results['prediction_test'] = prediction
        
        # 5. Бенчмарк производительности
        print("\n5️⃣ PERFORMANCE BENCHMARK")
        print("-" * 40)
        benchmark = self.benchmark_inference(50)
        results['benchmark'] = benchmark
        
        # Итоговый отчет
        print("\n🎉 " + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("🎉 " + "="*60)
        
        print(f"📊 Training accuracy: {results.get('training_accuracy', 'N/A'):.2f}%")
        print(f"🔄 ONNX conversion: {'✅ Success' if results.get('onnx_conversion') else '❌ Failed'}")
        print(f"⚡ Avg inference time: {results.get('benchmark', {}).get('avg_time_ms', 'N/A'):.2f} ms")
        print(f"🚄 Throughput: {results.get('benchmark', {}).get('throughput_fps', 'N/A'):.1f} img/sec")
        
        # Сохраняем результаты
        with open("demo_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to demo_results.json")
        
        print("\n🎯 ML Pipeline is working on CPU! 🎯")
        return results

def main():
    pipeline = MLPipeline()
    results = pipeline.run_complete_demo()
    return results

if __name__ == "__main__":
    main()
"""
Веб-приложение для загрузки и классификации изображений
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
from PIL import Image
import io
import base64
import time
import uvicorn
from working_demo import SimpleCNN
import os

# Создаем FastAPI приложение
app = FastAPI(
    title="ML Image Classifier",
    description="Веб-интерфейс для классификации изображений",
    version="1.0.0"
)

# Настройка статических файлов и шаблонов
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except:
    templates = None

class MLWebApp:
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
        
        # Статистика
        self.stats = {
            'total_predictions': 0,
            'start_time': time.time(),
            'predictions_by_class': {name: 0 for name in self.class_names}
        }
        
        self.load_model()
    
    def load_model(self):
        """Загружает обученную модель"""
        model_path = "models/working_model.pth"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return False
        
        try:
            self.model = SimpleCNN(num_classes=10)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Предобрабатывает изображение"""
        # Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Изменяем размер до 32x32
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Конвертируем в numpy array и нормализуем
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return torch.from_numpy(img_array).unsqueeze(0)
    
    def predict(self, image):
        """Выполняет предсказание"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        start_time = time.time()
        
        # Предобработка
        input_tensor = self.preprocess_image(image)
        
        # Инференс
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        inference_time = time.time() - start_time
        
        # Обновляем статистику
        self.stats['total_predictions'] += 1
        self.stats['predictions_by_class'][self.class_names[predicted_class]] += 1
        
        # Топ-3 предсказания
        top3_indices = torch.argsort(probabilities[0], descending=True)[:3]
        top3_predictions = []
        for idx in top3_indices:
            top3_predictions.append({
                'class_name': self.class_names[idx],
                'confidence': probabilities[0][idx].item()
            })
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'inference_time_ms': inference_time * 1000,
            'top3_predictions': top3_predictions,
            'image_size': input_tensor.shape
        }

# Создаем глобальный экземпляр приложения
ml_app = MLWebApp()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Image Classifier</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; background: #fafafa; }
            .upload-area:hover { border-color: #007bff; background: #f0f8ff; }
            input[type="file"] { margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .result { margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 10px; border-left: 4px solid #007bff; }
            .result h3 { margin-top: 0; color: #007bff; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }
            .stat-card { background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }
            .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
            .preview { text-align: center; margin: 20px 0; }
            .preview img { max-width: 200px; max-height: 200px; border: 1px solid #ddd; border-radius: 5px; }
            .top3 { margin-top: 15px; }
            .prediction-item { margin: 10px 0; padding: 10px; background: white; border-radius: 5px; display: flex; justify-content: space-between; }
            .confidence-bar { height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin-top: 5px; }
            .confidence-fill { height: 100%; background: linear-gradient(90deg, #28a745, #ffc107, #dc3545); transition: width 0.3s; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 ML Image Classifier</h1>
            <p style="text-align: center; color: #666;">Загрузите изображение для классификации с помощью нейронной сети</p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Нажмите здесь или перетащите изображение</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
            </div>
            
            <div id="preview" class="preview" style="display: none;"></div>
            <div id="result" style="display: none;"></div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number" id="totalPredictions">0</div>
                    <div>Всего предсказаний</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="avgTime">0</div>
                    <div>Среднее время (мс)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">CPU</div>
                    <div>Устройство</div>
                </div>
            </div>
        </div>

        <script>
            let totalTime = 0;
            let predictionCount = 0;

            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) return;

                // Показываем превью
                const preview = document.getElementById('preview');
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                preview.innerHTML = '<p>📸 Загруженное изображение:</p>';
                preview.appendChild(img);
                preview.style.display = 'block';

                // Подготавливаем данные для отправки
                const formData = new FormData();
                formData.append('file', file);

                try {
                    // Отправляем запрос
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    
                    if (response.ok) {
                        displayResult(result);
                        updateStats(result);
                    } else {
                        document.getElementById('result').innerHTML = '<div class="result">❌ Ошибка: ' + result.detail + '</div>';
                        document.getElementById('result').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = '<div class="result">❌ Ошибка сети: ' + error.message + '</div>';
                    document.getElementById('result').style.display = 'block';
                }
            }

            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                
                let top3Html = '';
                result.top3_predictions.forEach((pred, index) => {
                    const percentage = (pred.confidence * 100);
                    top3Html += `
                        <div class="prediction-item">
                            <span>${index + 1}. ${pred.class_name}</span>
                            <span>${percentage.toFixed(1)}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${percentage}%"></div>
                        </div>
                    `;
                });

                resultDiv.innerHTML = `
                    <div class="result">
                        <h3>🎯 Результат классификации</h3>
                        <p><strong>Предсказанный класс:</strong> ${result.predicted_class}</p>
                        <p><strong>Уверенность:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Время инференса:</strong> ${result.inference_time_ms.toFixed(1)} мс</p>
                        
                        <div class="top3">
                            <h4>📊 Топ-3 предсказания:</h4>
                            ${top3Html}
                        </div>
                    </div>
                `;
                resultDiv.style.display = 'block';
            }

            function updateStats(result) {
                predictionCount++;
                totalTime += result.inference_time_ms;
                
                document.getElementById('totalPredictions').textContent = predictionCount;
                document.getElementById('avgTime').textContent = (totalTime / predictionCount).toFixed(1);
            }

            // Поддержка drag & drop
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#007bff';
                uploadArea.style.background = '#f0f8ff';
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#ccc';
                uploadArea.style.background = '#fafafa';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                uploadArea.style.background = '#fafafa';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    uploadImage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """API для предсказания класса изображения"""
    
    # Проверяем тип файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Выполняем предсказание
        result = ml_app.predict(image)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

@app.get("/stats")
async def get_stats():
    """API для получения статистики"""
    uptime = time.time() - ml_app.stats['start_time']
    
    return {
        'total_predictions': ml_app.stats['total_predictions'],
        'uptime_seconds': uptime,
        'predictions_by_class': ml_app.stats['predictions_by_class'],
        'model_loaded': ml_app.model is not None
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        'status': 'healthy',
        'model_loaded': ml_app.model is not None,
        'timestamp': time.time()
    }

if __name__ == "__main__":
    print("🚀 Starting ML Web Application...")
    print("📂 Make sure model exists at: models/working_model.pth")
    
    if ml_app.model is None:
        print("❌ Model not loaded! Please train a model first.")
        exit(1)
    
    print("✅ Model loaded successfully!")
    print("🌐 Starting web server at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
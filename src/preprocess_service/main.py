"""
FastAPI микросервис для предобработки изображений
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import base64
import logging
from typing import List, Optional
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus метрики
REQUEST_COUNT = Counter('preprocess_requests_total', 'Total preprocessing requests')
REQUEST_DURATION = Histogram('preprocess_request_duration_seconds', 'Request duration')
ERROR_COUNT = Counter('preprocess_errors_total', 'Total preprocessing errors')

app = FastAPI(
    title="Image Preprocessing Service",
    description="Микросервис для предобработки изображений CIFAR-10",
    version="1.0.0"
)


class ImagePreprocessor:
    """Класс для предобработки изображений"""
    
    def __init__(self):
        # CIFAR-10 параметры нормализации
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])
        self.target_size = (32, 32)
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Предобработка одного изображения
        
        Args:
            image: PIL изображение
            
        Returns:
            Предобработанный numpy массив
        """
        try:
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Изменяем размер до 32x32
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Конвертируем в numpy array и нормализуем в [0, 1]
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Применяем CIFAR-10 нормализацию
            img_array = (img_array - self.mean) / self.std
            
            # Изменяем порядок осей с HWC на CHW (channels first)
            img_array = np.transpose(img_array, (2, 0, 1))
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
    
    def preprocess_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Предобработка батча изображений
        
        Args:
            images: Список PIL изображений
            
        Returns:
            Numpy массив с батчем предобработанных изображений
        """
        processed_images = []
        
        for i, image in enumerate(images):
            try:
                processed_img = self.preprocess_image(image)
                processed_images.append(processed_img)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to process image {i}: {str(e)}"
                )
        
        # Складываем в батч
        batch = np.stack(processed_images, axis=0)
        return batch


# Глобальный экземпляр препроцессора
preprocessor = ImagePreprocessor()


@app.get("/")
async def root():
    """Главная страница"""
    return {"message": "Image Preprocessing Service", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics")
async def metrics():
    """Prometheus метрики"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/preprocess/single")
async def preprocess_single_image(file: UploadFile = File(...)):
    """
    Предобработка одного изображения
    
    Args:
        file: Загруженный файл изображения
        
    Returns:
        JSON с предобработанными данными
    """
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # Проверяем тип файла
        if not file.content_type.startswith('image/'):
            ERROR_COUNT.inc()
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Читаем изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Предобработка
        processed_image = preprocessor.preprocess_image(image)
        
        # Конвертируем в список для JSON сериализации
        processed_list = processed_image.tolist()
        
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        
        logger.info(f"Successfully processed single image in {duration:.3f}s")
        
        return {
            "status": "success",
            "shape": processed_image.shape,
            "data": processed_list,
            "processing_time": duration
        }
        
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error processing single image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/batch")
async def preprocess_batch_images(files: List[UploadFile] = File(...)):
    """
    Предобработка батча изображений
    
    Args:
        files: Список загруженных файлов изображений
        
    Returns:
        JSON с предобработанными данными
    """
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        if len(files) == 0:
            ERROR_COUNT.inc()
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 32:  # Ограничиваем размер батча
            ERROR_COUNT.inc()
            raise HTTPException(status_code=400, detail="Batch size too large (max 32)")
        
        images = []
        
        # Загружаем все изображения
        for i, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                ERROR_COUNT.inc()
                raise HTTPException(status_code=400, detail=f"File {i} must be an image")
            
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        
        # Предобработка батча
        processed_batch = preprocessor.preprocess_batch(images)
        
        # Конвертируем в список для JSON сериализации
        processed_list = processed_batch.tolist()
        
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        
        logger.info(f"Successfully processed batch of {len(files)} images in {duration:.3f}s")
        
        return {
            "status": "success",
            "batch_size": len(files),
            "shape": processed_batch.shape,
            "data": processed_list,
            "processing_time": duration
        }
        
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/base64")
async def preprocess_base64_image(image_data: dict):
    """
    Предобработка изображения из base64 строки
    
    Args:
        image_data: Словарь с base64 данными изображения
        
    Returns:
        JSON с предобработанными данными
    """
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        if "image" not in image_data:
            ERROR_COUNT.inc()
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Декодируем base64
        base64_string = image_data["image"]
        if base64_string.startswith('data:image'):
            # Убираем data URL prefix
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Предобработка
        processed_image = preprocessor.preprocess_image(image)
        
        # Конвертируем в список для JSON сериализации
        processed_list = processed_image.tolist()
        
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        
        logger.info(f"Successfully processed base64 image in {duration:.3f}s")
        
        return {
            "status": "success",
            "shape": processed_image.shape,
            "data": processed_list,
            "processing_time": duration
        }
        
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error processing base64 image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_preprocessing_info():
    """Информация о параметрах предобработки"""
    return {
        "target_size": preprocessor.target_size,
        "normalization": {
            "mean": preprocessor.mean.tolist(),
            "std": preprocessor.std.tolist()
        },
        "output_format": "CHW (channels first)",
        "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
Упрощенный скрипт обучения без torchvision (для случаев конфликтов)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class SimpleCNN(nn.Module):
    """Простая CNN для демонстрации"""
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

def create_dummy_data(num_samples=1000):
    """Создает тестовые данные вместо CIFAR-10"""
    print("Creating dummy data (since CIFAR-10 might not load)...")
    
    # Создаем случайные изображения 32x32x3
    X = torch.randn(num_samples, 3, 32, 32)
    
    # Создаем случайные метки 0-9
    y = torch.randint(0, 10, (num_samples,))
    
    return X, y

def simple_train(epochs=5):
    """Простое обучение на тестовых данных"""
    print(f"Training simple CNN for {epochs} epochs...")
    
    # Создаем модель
    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Создаем данные
    X_train, y_train = create_dummy_data(1000)
    X_test, y_test = create_dummy_data(200)
    
    # Обучение
    model.train()
    for epoch in range(epochs):
        # Простое обучение без DataLoader
        batch_size = 32
        total_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # Тестирование
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item() * 100
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        model.train()
    
    # Сохранение модели
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': accuracy,
        'test_loss': avg_loss,
    }, "models/cifar10_model.pth")
    
    print(f"✓ Model saved to models/cifar10_model.pth")
    print(f"Final accuracy: {accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    simple_train(epochs)
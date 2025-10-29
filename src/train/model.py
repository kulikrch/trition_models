"""
Простая CNN модель для классификации CIFAR-10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Простая сверточная нейронная сеть для CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Первый блок: conv -> bn -> relu -> pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Второй блок: conv -> bn -> relu -> pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Третий блок: conv -> bn -> relu -> pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10):
    """Создает и возвращает модель"""
    return SimpleCNN(num_classes=num_classes)


if __name__ == "__main__":
    # Тест модели
    model = get_model()
    print(f"Model: {model}")
    
    # Тестовый вход
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
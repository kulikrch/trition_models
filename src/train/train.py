"""
Скрипт для обучения модели на CIFAR-10
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from model import get_model


class Trainer:
    def __init__(self, model, device='cpu', batch_size=128, learning_rate=0.001):
        self.device = 'cpu'  # Принудительно используем CPU
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Optimizer и loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler для learning rate
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        print(f"Using device: {self.device}")
        
    def get_data_loaders(self):
        """Создает data loaders для CIFAR-10"""
        
        # Трансформации для обучения (с аугментацией)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Трансформации для валидации
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Датасеты
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        # Data loaders (уменьшаем num_workers для CPU)
        train_loader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, test_loader):
        """Валидация модели"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc
    
    def train(self, epochs=50, save_path='../../models/cifar10_model.pth'):
        """Полный цикл обучения"""
        train_loader, test_loader = self.get_data_loaders()
        
        best_acc = 0
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Валидация
            test_loss, test_acc = self.validate(test_loader)
            
            # Scheduler step
            self.scheduler.step()
            
            # Сохранение метрик
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохранение лучшей модели
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }, save_path)
                print(f"New best model saved! Accuracy: {best_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best test accuracy: {best_acc:.2f}%")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'best_acc': best_acc,
            'training_time': training_time
        }


def main():
    """Основная функция для запуска обучения"""
    # Создание модели
    model = get_model(num_classes=10)
    
    # Создание trainer
    trainer = Trainer(
        model=model,
        device='cpu',
        batch_size=64,  # Уменьшаем batch size для CPU
        learning_rate=0.001
    )
    
    # Обучение
    results = trainer.train(epochs=50)
    
    print("Training results:")
    for key, value in results.items():
        if isinstance(value, list):
            print(f"{key}: [показаны последние 5 значений] {value[-5:]}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
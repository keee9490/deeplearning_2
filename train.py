import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report




# ==================== 1. 数据预处理和加载 ====================
class ImprovedFoodDataset(Dataset):
    def __init__(self, image_path, image_size=(224, 224), mode='train'):
        """
        改进的食物数据集类
        Args:
            image_path: 图像文件夹路径
            image_size: 目标图像大小
            mode: 'train', 'val', 或 'test'
        """
        self.image_path = image_path
        self.image_file_list = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
        self.mode = mode
        self.image_size = image_size

        # 类别映射
        self.class_names = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
            'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit'
        ]

        # ==================== 2. 设计train_transform ====================
        # 训练时使用更强的数据增强
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # 先缩放到稍大尺寸
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])

        # 验证和测试时使用简单的转换
        self.test_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.image_path, self.image_file_list[idx])

        # 使用PIL读取图像，与torchvision transforms兼容
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # 如果PIL读取失败，使用OpenCV
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        # 应用转换
        if self.mode == 'train':
            img = self.train_transforms(img)
            # 从文件名提取标签（格式："类别_编号.jpg"）
            filename = self.image_file_list[idx]
            label = int(filename.split("_")[0])
            return img, label
        else:
            img = self.test_transforms(img)
            # 对于验证集，也需要标签
            if self.mode == 'val':
                filename = self.image_file_list[idx]
                label = int(filename.split("_")[0])
                return img, label
            else:  # 测试集
                filename = self.image_file_list[idx]
                return img, filename


# ==================== 3. 可视化数据增强 ====================
def visualize_transforms(dataset, num_samples=5):
    """
    可视化数据增强效果
    """
    plt.figure(figsize=(15, 8))

    # 获取原始图像
    for i in range(num_samples):
        # 读取原始图像
        img_path = os.path.join(dataset.image_path, dataset.image_file_list[i])
        original_img = Image.open(img_path).convert('RGB')

        # 应用训练转换多次，展示不同增强效果
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(original_img)
        plt.title(f"Original\nClass: {dataset.class_names[int(dataset.image_file_list[i].split('_')[0])]}")
        plt.axis('off')

        # 展示三次不同的增强结果
        for j in range(3):
            augmented_img = dataset.train_transforms(original_img)
            # 将tensor转换回图像显示
            img_np = augmented_img.numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            plt.subplot(num_samples, 4, i * 4 + j + 2)
            plt.imshow(img_np)
            plt.title(f"Augmented {j + 1}")
            plt.axis('off')

    plt.suptitle("Data Augmentation Visualization", fontsize=16)
    plt.tight_layout()
    plt.savefig('data_augmentation_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 4. 模型设计 ====================
class FoodClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(FoodClassifier, self).__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            # 第一卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    """
    训练模型
    """
    # 初始化TensorBoard
    writer = SummaryWriter('runs/food_classification_experiment')

    # 将模型移到设备
    model = model.to(device)

    # 记录训练过程中的指标
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_accuracy = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * train_correct / train_total})

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, 'best_food_classifier.pth')

    # 关闭TensorBoard writer
    writer.close()

    # 保存最终模型
    torch.save(model.state_dict(), 'final_food_classifier.pth')

    return model, train_losses, val_losses, val_accuracies, best_model_state


# ==================== 可视化训练过程 ====================
def plot_training_history(train_losses, val_losses, val_accuracies):
    """
    可视化训练过程
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 5. 评估模型 ====================
def evaluate_model(model, val_loader, class_names, device='cuda'):
    """
    在验证集上评估模型
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # 生成分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return accuracy, cm


# ==================== 6. 测试集预测 ====================
def predict_test_set(model, test_loader, device='cuda'):
    """
    对测试集进行预测
    """
    model.eval()
    model = model.to(device)

    predictions = []
    filenames = []

    with torch.no_grad():
        for inputs, filename_batch in tqdm(test_loader, desc='Predicting Test Set'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            filenames.extend(filename_batch)

    # 创建结果DataFrame
    results = pd.DataFrame({
        'image_id': filenames,
        'label': predictions
    })

    # 保存到CSV文件
    results.to_csv('ans_ours.csv', index=False)
    print(f"Predictions saved to ans_ours.csv, total {len(results)} predictions")

    return results


# ==================== 主函数 ====================
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集路径（请根据实际情况修改）
    train_path = 'archive/training'
    val_path = 'archive/validation'
    test_path = 'archive/evaluation'

    # ==================== 1. 构造数据集 ====================
    print("Loading datasets...")
    train_dataset = ImprovedFoodDataset(train_path, image_size=(224, 224), mode='train')
    val_dataset = ImprovedFoodDataset(val_path, image_size=(224, 224), mode='val')
    test_dataset = ImprovedFoodDataset(test_path, image_size=(224, 224), mode='test')

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # ==================== 2. & 3. 可视化数据增强 ====================
    print("\nVisualizing data augmentation...")
    visualize_transforms(train_dataset, num_samples=3)

    # ==================== 创建数据加载器 ====================
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ==================== 4. 初始化模型和优化器 ====================
    print("\nInitializing model...")
    model = FoodClassifier(num_classes=11)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # ==================== 训练模型 ====================
    print("\nStarting training...")
    num_epochs = 15
    trained_model, train_losses, val_losses, val_accuracies, best_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )

    # ==================== 可视化训练过程 ====================
    print("\nPlotting training history...")
    plot_training_history(train_losses, val_losses, val_accuracies)

    # ==================== 5. 在验证集上评估 ====================
    print("\nEvaluating on validation set...")
    # 加载最佳模型
    model.load_state_dict(best_state)
    accuracy, cm = evaluate_model(model, val_loader, train_dataset.class_names, device)

    # ==================== 6. 测试集预测 ====================
    print("\nPredicting test set...")
    results = predict_test_set(model, test_loader, device)

    # 显示一些预测结果
    print("\nSample predictions:")
    print(results.head(10))

    # 统计各类别数量
    print("\nPrediction distribution:")
    print(results['label'].value_counts().sort_index())

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
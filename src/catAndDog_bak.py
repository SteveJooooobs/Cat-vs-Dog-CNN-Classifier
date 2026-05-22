"""
猫狗二分类 - CNN 训练脚本
"""
import time
import os
import torch
import torch_directml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
# from pytorchtools import EarlyStopping


# ============================================================
# 模型
# ============================================================

class CatDogNet(nn.Module):
    """更深更大的模型，添加 Dropout 防止过拟合"""
    def __init__(self,dropout_rate = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32), nn.ReLU(),
            # nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(dropout_rate * 0.5),


            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.ReLU(),
            # nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128), nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.55),
            #
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256), nn.ReLU(),
            # nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.75),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout_rate*0.35),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.features(x)


# ============================================================
# 数据集
# ============================================================
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # [(path, label), ...]

        for label, cls in enumerate(['Cat', 'Dog']):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(cls_dir, fname)
                    # 初始化时过滤损坏图片
                    try:
                        img = Image.open(fpath)
                        img.convert('RGB')
                        self.samples.append((fpath, label))
                    except Exception:
                        pass  # 跳过损坏文件

        cat_n = sum(1 for _, l in self.samples if l == 0)
        dog_n = sum(1 for _, l in self.samples if l == 1)
        print(f"数据集加载完成: 总计 {len(self.samples)} 张 (猫={cat_n}, 狗={dog_n})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = Image.open(fpath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 数据增强
# ============================================================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ============================================================
# 训练 / 评估
# ============================================================
def train_model(model, train_loader, val_loader, device, epochs=5, label=""):
    print(f"\n{'='*50}")
    print(f"训练: {label}  |  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        mode='min',          # 监控指标模式（默认最小化）
        factor=0.5,
        )
    # 早停机制
    # early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item()
                _, pred = torch.max(logits, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        # 关键：根据验证损失调整学习率
        scheduler.step(val_loss)
        # 获取当前学习率（用于调试）
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"训练: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"验证: loss={val_loss:.4f} acc={val_acc:.1f}% | "
              f"lr={current_lr:.6f} | "
              f"耗时: {time.time()-t0:.0f}s")
        # early_stopping(val_loss, model)

        # if early_stopping.early_stop:
        #     print("早停触发")
        #     break
    return train_acc, val_acc


# ============================================================
# 主流程
# ============================================================
def main():
    data_dir = r"F:\PyCode\machinelearn\yolo\runs\detect\猫狗分类图片\PetImages"
    device = torch_directml.device()
    print(f"设备: {device}")

    # 1. 加载全量数据（已过滤损坏图片）
    train_tf, val_tf = get_transforms()
    full_ds = CatDogDataset(data_dir, transform=None)  # 先不加 transform
    n_total = len(full_ds)
    n_train = int(n_total * 0.8)

    # 2. 固定索引划分
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=g).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # 3. 分别构建训练/验证集（各自独立的 transform）
    train_ds = [(full_ds.samples[i][0], full_ds.samples[i][1]) for i in train_idx]
    val_ds = [(full_ds.samples[i][0], full_ds.samples[i][1]) for i in val_idx]

    # 通过设置 full_ds 的 transform 来复用 __getitem__，但用不同的 samples 列表
    # 最简单：创建一个薄包装
    class SubsetWithTransform(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            fpath, label = self.samples[idx]
            image = Image.open(fpath).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    train_loader = DataLoader(
        SubsetWithTransform(train_ds, train_tf),
        batch_size=64, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        SubsetWithTransform(val_ds, val_tf),
        batch_size=64, shuffle=False, num_workers=0,
    )

    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张")

    # 4. 验证：到底是模型架构问题还是数据集问题
    # CatDogNet 已去掉 BN 和 Dropout，如果还是不学，说明 DirectML 撑不住深网络
    # 如果学了，说明之前是 BN/Dropout 的问题
    model = CatDogNet()
    train_model(model, train_loader, val_loader, device, epochs=8, label="CatDogNet (有BN/有Dropout)")

    torch.save(model.state_dict(), 'cat_dog_model.pth')
    print(f"\n模型已保存至 cat_dog_model.pth")


if __name__ == '__main__':
    main()

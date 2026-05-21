import torch
import torch.nn as nn
from torch import optim
import multiprocessing
import time

class CatVsDog_CNNModuleSet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取器 
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),     # 输入彩色3通道，32卷积核，每个3*3，不改变原尺寸
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 2*2池化，尺寸减半     112*112
            nn.Conv2d(32, 64, 3, padding=1),    # 64卷积核提取64管道特征图
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 56*56
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 28*28
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)                     # 14*14
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 分类器-全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 256),            
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)        # 特征先进提取器，输出[batch,128,28,28]
        x = self.avgpool(x)         # 自适应池化，输出[batch,128,7,7]
        x = x.view(x.size(0), -1)   # 拉平到1维，输出[batch,128*7*7]
        x = self.classifier(x)      # 分类器得结果
        return x 
    
if __name__ == "__main__":
    import os
    os.makedirs("models",exist_ok=True)

    # Windows 多进程安全保护
    multiprocessing.freeze_support()
    
    # 动态获取DataLoader
    from data.clean_data import get_train_loader, get_val_loader, get_test_loader
    train_loader = get_train_loader(batch_size=32, num_workers=2)
    val_loader = get_val_loader(batch_size=32, num_workers=2)
    test_loader = get_test_loader(batch_size=32, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatVsDog_CNNModuleSet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()   # 猫狗2分类选交叉熵损失

    optimizer = optim.Adam(model.parameters(), lr=0.0025)    # 优化器Adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.75)    # 学习率调度器

    print("模型构造完成-损失函数构造完成-优化器构造完成-训练中.....")

    epoch = 60      #训练总轮次
    best_acc = 0.0  # 在训练循环前初始化

    for i in range(epoch):
        start = time.time()
        model.train()

        train_total_loss = 0.0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            '''
            # 每批次图片抽检

            # **--** 已完成重要历史使命的代码，保留 **--**

            if batch_idx % 50 == 0:
                print(f"图片张量形状: {images.shape}")
                print(f"图片张量最小值: {images.min().item():.4f}")
                print(f"图片张量最大值: {images.max().item():.4f}")
                print(f"前10个标签: {labels[:10].tolist()}")
                
                # 如果是归一化后的数据，可以逆向反看一张图
                import torchvision.transforms as T
                inv_normalize = T.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                sample_img = inv_normalize(images[0]).clamp(0, 1)
                
                # 用PIL看
                from PIL import Image
                img_to_show = T.ToPILImage()(sample_img)
                img_to_show.show()  # 会弹出一张图片确认
            '''

            images = images.to(device)           # 按批次搬运到显卡进行训练
            labels = labels.to(device)

            optimizer.zero_grad()       # 循环开始清空梯度

            outputs = model(images)     # 前向传播，传入特征图

            loss = criterion(outputs, labels)   # 传入前向传播的预测结果和labels比对计算损失

            loss.backward()             # 反向传播

            # 诊断代码：每个20batch 打印一次梯度
            # if batch_idx % 20 == 0:
                # grad_mean = model.features[0].weight.grad.abs().mean().item()
                # print(f"Batch {batch_idx}, grad_mean: {grad_mean:.6f}")

            optimizer.step()            # 更新参数

            train_total_loss += loss.item() * images.size(0)
            train_total += images.size(0)

        avg_train_loss = train_total_loss / train_total

        end = time.time()

        model.eval()

        val_total_loss = 0.0            # 验证集总损失
        val_correct = 0                 # 验证集正确数
        val_total = 0                   # 验证集样本数量
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                outputs = model(val_images)
                loss = criterion(outputs, val_labels)

                val_total_loss += loss.item() * val_images.size(0)  # 总损失累加
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

            avg_val_loss = val_total_loss / val_total
            val_accuracy = val_correct / val_total
        
        scheduler.step()

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), 'models/best_catdog_model-v2.pth')
            print(f"  >>> 新最佳模型已保存，准确率: {best_acc:.4f}")

        print(f"当前epoch:{i+1}/{epoch}\n",
            f"--当前Loss--: <训练损失>：{avg_train_loss} | <验证损失>:{avg_val_loss}\n",
            f"--验证准确率:{val_accuracy}\n",
            f"--当次耗时:{end-start}\n",
            "==============================================="
            )
            




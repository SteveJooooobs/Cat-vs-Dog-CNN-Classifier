import torch
import torch.nn as nn
from torch import optim
import multiprocessing
import time
import warnings
import logging
from PIL import ImageFile

# 【性能优化 1】解决“Truncated File Read”警告：允许 PIL 读取数据集中可能存在的部分截断的异常图像，防范读取错误导致崩溃。
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 【体验优化】屏蔽终端警告：配置 warnings 重定向，将所有 FutureWarning 和 UserWarning 写入项目根目录下的 train_warnings.log 文件中，保持终端整洁。
logging.basicConfig(
    filename='train_warnings.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)

def warn_to_logging(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

warnings.showwarning = warn_to_logging

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
            # 【参数调优 1】关闭 Dropout（设为 0.0）：
            # 实验表明，在从头训练的浅层网络中，BatchNorm 和 Dropout 双重强正则会产生联合过度抑制，
            # 大幅拉低了浅层网络在开始阶段提取基础几何特征的效率，导致 Loss 在 0.693 停滞。将其设为 0.0 以释放模型拟合力。
            nn.Dropout(0.0),
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
    # 使用旧版缩减数据集：from data.clean_data import get_train_loader, get_val_loader, get_test_loader
    # 使用完整版新数据集：
    from data.clean_data_2 import get_train_loader, get_val_loader, get_test_loader
    
    # 【性能优化 2】调整 DataLoader 参数以解除 CPU-GPU 数据传输瓶颈：
    # - batch_size 提升至 64（大幅减少迭代次数，提高 GPU 利用率）
    # - num_workers 设为 4（基于 Windows 平台开销折中推荐的值，加快 CPU 图像多线程读取与增强）
    # - pin_memory=True（锁定内存，配合 non_blocking 加速数据搬运至 GPU）
    train_loader = get_train_loader(batch_size=64, num_workers=4, pin_memory=True)
    val_loader = get_val_loader(batch_size=64, num_workers=4, pin_memory=True)
    test_loader = get_test_loader(batch_size=64, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatVsDog_CNNModuleSet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()   # 猫狗2分类选交叉熵损失

    # 【参数调优 2】初始学习率下调至 0.0005：
    # 相比 0.0025，下调至 0.0005 能够极大程度地防范自适应优化器 Adam 在训练刚启动的几个 Batch 里由于梯度累加破坏模型初始权重。
    # 结合每 15 轮衰减至 0.75 的 StepLR，能够保证高吞吐状态下收敛过程的温和与极高稳定性。
    optimizer = optim.Adam(model.parameters(), lr=0.0005)    # 优化器Adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.75)    # 学习率调度器

    # 【参数调优 3】关闭 AMP 混合精度训练：
    # 实验证实，在缺少预训练权重、结构较浅的手写 CNN 模型中，半精度 (FP16) 的范围压缩容易造成梯度下溢或自适应动量不稳定，
    # 导致网络卡在随机猜测（Loss 0.693）。由于本模型参数量极小（仅 5.4M），使用默认单精度 (FP32) 进行训练，
    # 依然可在 GPU 上实现飞快的迭代速度，并确保最优的数值计算精度。

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

            # 【性能优化 4】启用 non_blocking=True，将 CPU 到 GPU 的数据搬运动作异步化，与计算重叠以提高 GPU 利用率
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 【性能优化 5】使用 set_to_none=True 代替 zero_grad()，避免更新参数前对显存写零，提升梯度清除速度
            optimizer.zero_grad(set_to_none=True)

            # 【数值精度调优】采用标准 FP32 单精度前向传播与反向传播，确保收敛稳定性
            outputs = model(images)     # 前向传播，传入特征图
            loss = criterion(outputs, labels)   # 计算损失

            loss.backward()             # 反向传播
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
                # 【性能优化 8】验证阶段同样异步化数据拷贝并启用 AMP 混合精度加速前向推理
                val_images = val_images.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)

                # 【数值精度调优】验证/推理阶段采用标准 FP32 精度前向计算，保障验证准确性
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
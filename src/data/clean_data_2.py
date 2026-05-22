import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import ImageFile

# 【优化点 1】解决“Truncated File Read”警告：允许 PIL 读取由于网络下载或解压过程中被截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(BASE_DIR, 'data', 'raw_complate', 'PetImages')

# 图片特征处理（与 clean_data.py 的清洗/增强方式完全一致）
# 训练集
train_transforms = transforms.Compose([
    transforms.Resize(256),         # 先缩放到256
    transforms.RandomCrop(224),     # 再随机缩放裁剪出224*224的区域
    transforms.RandomHorizontalFlip(),      # 随机翻转
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),                  # 转成张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 验证集
val_transforms = transforms.Compose([
    transforms.Resize(256),                 # 固定尺寸缩放
    transforms.CenterCrop(224),             # 固定尺寸裁剪 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 测试集
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 鲁棒的数据集包装器
# 经典猫狗完整版数据集中存在个别损坏或无法读取的图片（如 PetImages/Cat/666.jpg 等），
# 当多进程 DataLoader 读取损坏文件报错时，本包装器会捕获异常并自动、随机替换为另一个样本，确保训练流畅不崩溃。
class SafeDatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        try:
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        except Exception as e:
            # 捕获损坏图像读取异常，随机挑选另一个索引样本替换
            import random
            new_idx = random.randint(0, len(self.subset) - 1)
            return self.__getitem__(new_idx)
            
    def __len__(self):
        return len(self.subset)

# 加载完整版数据集
full_dataset = ImageFolder(data_dir)

# 动态按 8:1:1 比例划分训练、验证、测试集
total_len = len(full_dataset)
train_len = int(total_len * 0.8)
val_len = int(total_len * 0.1)
test_len = total_len - train_len - val_len

# 固定随机种子，保证每次运行划分的数据集高度一致且可复现
generator = torch.Generator().manual_seed(42)
train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_len, val_len, test_len], generator=generator
)

# 使用 SafeDatasetWrapper 包装并应用相应 transforms
train_dataset = SafeDatasetWrapper(train_subset, transform=train_transforms)
val_dataset = SafeDatasetWrapper(val_subset, transform=val_transforms)
test_dataset = SafeDatasetWrapper(test_subset, transform=test_transforms)

# 创建数据加载对象
# 【优化点 2】支持 pin_memory=True，使 DataLoader 的数据直接存放在页锁定内存（Pinned Memory）中，
# 加快 CPU 到 GPU 的数据拷贝速度，结合 non_blocking=True 可进一步重叠数据传输和计算。
def get_train_loader(batch_size=32, num_workers=2, pin_memory=True):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

def get_val_loader(batch_size=32, num_workers=2, pin_memory=True):
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

def get_test_loader(batch_size=32, num_workers=2, pin_memory=True):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

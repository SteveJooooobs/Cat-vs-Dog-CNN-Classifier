from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from PIL import ImageFile

# 解决“Truncated File Read”警告：允许 PIL 读取数据集中损坏的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(BASE_DIR)

train_dir = os.path.join(BASE_DIR, 'data', 'raw', 'train')
val_dir = os.path.join(BASE_DIR, 'data', 'raw', 'val')
test_dir = os.path.join(BASE_DIR, 'data', 'raw', 'test')

# print(f"train_dir : {train_dir},\nval_dir : {val_dir},\ntest_dir : {test_dir}")

# 图片特征处理
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

# 创建读图片的ImageFolder对象
train_dataset = ImageFolder(train_dir, train_transforms)
val_dataset = ImageFolder(val_dir, val_transforms)
test_dataset = ImageFolder(test_dir, test_transforms)

# 创建数据加载对象
def get_train_loader(batch_size=32, num_workers=2, pin_memory=True):
    train_dataset = ImageFolder(train_dir, train_transforms)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
def get_val_loader(batch_size=32, num_workers=2, pin_memory=True):
    val_dataset = ImageFolder(val_dir, val_transforms)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
def get_test_loader(batch_size=32, num_workers=2, pin_memory=True):
    test_dataset = ImageFolder(test_dir, test_transforms)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)



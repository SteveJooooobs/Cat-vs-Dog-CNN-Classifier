import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

# 使用旧版缩减数据集测试：from src.data import get_test_loader
# 使用新版完整数据集测试：
from src.data import get_test_loader_v2 as get_test_loader
from src.train_script import CatVsDog_CNNModuleSet
import torch.nn as nn
import torch
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join(BASE_DIR, 'models','best_catdog_model.pth')
    model = CatVsDog_CNNModuleSet().to(device)
    model.load_state_dict(torch.load(model_dir))

    print("===== 模型加载完成 =====")

    test_loader = get_test_loader(batch_size=32, num_workers=2)
    print("===== 数据加载完成 =====")

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

        avg_test_loss = test_loss / test_total
        test_accuracy = test_correct / test_total

        print(f"平均损失:{avg_test_loss} --- 准确率:{test_accuracy}")






# 导入自定义的数据加载包
import torchvision

from LFWDataset import LFWDataset
# 导入需要的包
import torch
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from tqdm import tqdm
from PIL import Image

# 设置超参数
BATCH_SIZE = 64
LR = 0.1  # 学习率
MM = 0.6  # 随机梯度下降法中的momentum参数
EPOCH = 10  # 训练轮数

# 设置pytorch使用的设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载数据集
print('Loading LFW data...')
lfw_dataset = LFWDataset()
train_data = lfw_dataset.getTrainData()
validation_data = lfw_dataset.getValidationData()
test_data = lfw_dataset.getTestData()

# 构建dataloader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
print('Loading over.')


# 定义网络结构(这里引用了AlexNet的代码)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 创建网络模型
model = Net().to(device)

# 定义损失函数,分类问题采用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化方法,此处使用随机梯度下降法
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)
# 定义每5个epoch，学习率变为之前的0.1
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)


# 训练神经网络
def train_model(model_, criterion, optimizer, scheduler):
    """
    :param model_: 训练使用的模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 调整学习率lr的方法
    :return: model
    """
    start_time = time.time()
    model_.train()

    running_loss = 0.0

    for data in tqdm(train_loader):  # tqdm:进度条
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model_(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.flatten().size(0)
    scheduler.step()

    epoch_loss = running_loss / len(train_data)
    print('training loss: {:.4f}'.format(epoch_loss))

    end_time = time.time()
    print("training time:{:.4f} s".format(end_time - start_time))

    return model_


# 验证神经网络
def validation_model(model_, criterion):
    """
    :param model_: 验证使用的模型
    :param criterion: 损失函数
    :return: 验证正确率
    """
    model_.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for data in tqdm(validation_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.flatten().size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

    accuracy = running_corrects / len(validation_data)
    epoch_loss = running_loss / len(validation_data)

    print('validation loss: {:.4f} accuracy: {:.4f}'.format(epoch_loss, accuracy))

    return accuracy


# 测试神经网络
def test_model(model_, criterion):
    """
    :param model_: 测试使用的模型
    :param criterion: 损失函数
    :return: None
    """
    model_.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.flatten().size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

    accuracy = running_corrects / len(test_data)
    epoch_loss = running_loss / len(test_data)

    print('test loss: {:.4f} accuracy: {:.4f}'.format(epoch_loss, accuracy))

    return


# 训练、验证与测试
if __name__ == '__main__':
    best_accuracy = 0.0
    best_model_index = 0
    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch + 1, EPOCH))

        model = train_model(model, loss_func, optimizer_ft, exp_lr_scheduler)
        # 保存每轮训练后模型到本地
        torch.save(model, "model_{}.pth".format(epoch))
        print("模型已保存")
        epoch_accuracy = validation_model(model, loss_func)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_index = epoch

    # 加载最优模型
    print("the best model is Epoch {}".format(best_model_index + 1))
    best_model = torch.load("model_{}.pth".format(best_model_index))
    # 测试神经网络
    test_model(best_model, loss_func)

from PIL import Image
import torchvision
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
model = torch.load('model_9.pth')


def judge_sex(image_path, model_):
    try:
        test_image = Image.open(image_path)
    except FileNotFoundError:
        print('未找到该文件')
        return
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])
                                                ])
    test_image_tensor = transform(test_image)
    test_image_tensor = torch.reshape(test_image_tensor, (1, 3, 224, 224))
    test_image_tensor.to(device)
    model_.eval()
    with torch.no_grad():
        output = model_(test_image_tensor)

    sex = output.argmax(1).sum().item()
    print('sex:', end='')
    if sex == 0:
        print('female')
    else:
        print('male')
    return


if __name__ == '__main__':
    print("=======性别识别程序=======")
    while True:
        test_image_path = input("请输入想要判断性别的图片路径:")
        judge_sex(test_image_path, model)
        choice = input("是否继续?(y/n)")
        while choice != 'y' and choice != 'n':
            choice = input("请输入y/n:")
        if choice == 'n':
            break

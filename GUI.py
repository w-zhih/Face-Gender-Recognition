import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showwarning
import torchvision
from torch import nn
import torch
import os
from PIL import Image, ImageTk

# 导入性别识别模型
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


model = torch.load('model_9.pth')


# 判断图片性别
def judge_sex(image_path, model_):
    try:
        test_image = Image.open(image_path)
    except FileNotFoundError:
        showwarning(title='警告', message='未找到该文件')
        return ''
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
    return 'female' if sex == 0 else 'male'


def selectImage():
    image_path = askopenfilename(title='选择图片',
                                 initialdir=os.path.dirname(__file__),
                                 filetypes=[('Image', ('*.jpg', '*.jpeg', '*.jfif', '*.png'))])
    if image_path == '':
        showwarning(title='警告', message='未选择图片')
    else:
        var.set(image_path)
        load = Image.open(image_path)
        load = Resize(load, 180)
        global render
        render = ImageTk.PhotoImage(load)
        img.config(image=render)
        if image_sex.get() != '':
            image_sex.set('')


def judgeSex():
    image_path = var.get()
    if image_path == '':
        showwarning(title='警告', message='未选择图片')
    else:
        result = judge_sex(image_path, model)
        image_sex.set(result)


# 修改图像大小以适应容器
def Resize(PIL_image, box_size):
    w, h = PIL_image.size
    major = max(w, h)
    width = int(box_size / major * w)
    height = int(box_size / major * h)
    return PIL_image.resize((width, height), Image.ANTIALIAS)


# 设计图形化界面
top = tk.Tk()
top.title('性别识别程序')
top.geometry('400x350+600+250')
top.resizable(True, False)

# 定义路径label
var = tk.StringVar()  # 图片路径
path = tk.Label(top, textvariable=var, font=('Arial', 12), width=200, height=3, justify='left')
path.pack()
# 定义按钮
select = tk.Button(top, text='选择图片', command=selectImage)
select.pack()
judge = tk.Button(top, text='判断性别', command=judgeSex)
judge.pack()
# 定义图片
image_sex = tk.StringVar()
render = None
img = tk.Label(master=top,
               textvariable=image_sex,
               compound='center',
               width=180,
               height=180,
               font=('Arial', 20),
               foreground='white')
img.pack()

# 进入消息循环
top.mainloop()

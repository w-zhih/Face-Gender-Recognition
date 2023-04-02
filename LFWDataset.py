import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import TensorDataset
import torchvision


# 读取图片数据，得到图片对应的像素值的数组，均一化到0-1之前
def loadPicTensor(picFilePath):
    picData = Image.open(picFilePath)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
    picTensor = transform(picData)
    return picTensor


class LFWDataset:
    images = []
    labels = []
    sexDict = {}

    dataPath = 'lfw_funneled/'

    train_images = []
    train_labels = []

    validation_images = []
    validation_labels = []

    test_images = []
    test_labels = []
    test_females_images = []
    test_females_labels = []

    dataCount = 0

    # 加载LFW数据
    def __init__(self):
        # 导入各照片对应性别
        males = open('male_names.txt', 'r', encoding='utf-8')
        males_list = males.readlines()
        males.close()
        for man in males_list:
            man = man[:-1]
            self.sexDict[man] = 1

        females = open('female_names.txt', 'r', encoding='utf-8')
        females_list = females.readlines()
        females.close()
        for woman in females_list:
            woman = woman[:-1]
            self.sexDict[woman] = 0

        # 加载照片为tensor格式
        all_files = os.listdir(self.dataPath)
        for each_dir in all_files:
            if not each_dir.endswith('.txt'):
                person_files = os.listdir(
                    os.path.join(self.dataPath, each_dir))
                for each_file in person_files:
                    image = loadPicTensor(os.path.join(
                        self.dataPath, each_dir, each_file))
                    if each_file in self.sexDict:
                        label = self.sexDict[each_file]
                        self.images.append(image)
                        self.labels.append(label)
                        self.dataCount += 1

        # 打乱数据，使用相同的次序打乱images、labels，保证数据仍然对应
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

        # 按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.dataCount * 0.4)
        validationIndex = int(self.dataCount * 0.5)
        self.train_images = self.images[:trainIndex]
        self.train_labels = self.labels[:trainIndex]
        self.validation_images = self.images[trainIndex:validationIndex]
        self.validation_labels = self.labels[trainIndex:validationIndex]
        self.test_images = self.images[validationIndex:]
        self.test_labels = self.labels[validationIndex:]

        # 将数据打包成数据集
        self.train_data = TensorDataset(torch.stack(self.train_images), torch.tensor(self.train_labels))
        self.validation_data = TensorDataset(torch.stack(self.validation_images), torch.tensor(self.validation_labels))
        self.test_data = TensorDataset(torch.stack(self.test_images), torch.tensor(self.test_labels))

    def getTrainData(self):
        return self.train_data

    def getValidationData(self):
        return self.validation_data

    def getTestData(self):
        return self.test_data


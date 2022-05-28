import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from DataLookAndStandard import MyDataset
from torchvision.utils import make_grid
import torch.nn as nn  # 神经网络
import torch.optim as optim  # 优化器
import numpy as np
import os

batches = 4
classes = ['男', '女']
PATH = './gender.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集加载

trainset = MyDataset('./data/train',
                     './data/train/train.txt',
                     transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010))
                                                   ]))
testset = MyDataset('./data/test',
                    './data/test/test.txt',
                    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batches, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batches)


#  卷积网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(  # 特征提取
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            nn.ReLU(inplace=True),  # inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27] kernel_num为原论文一半
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全链接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平   或者view()
        x = self.classifier(x)
        return x


net = Net()
net.to(device)


def train(epochs, verify):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    correct = 0
    total = 0
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            # 当前状态
            if i % verify == 0:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('[epoch:%d, %5d] loss: %.3f  准确率:%.4f' % (epoch + 1, i + 1, loss.item(), correct / total))

    torch.save(net.state_dict(), PATH)
    print('训练完毕')


#  验证集
def Verify():
    fflag = False
    if input('要查看图片吗\n要输入1否则无图片反馈') == '1': fflag = True
    print("验证进行中。。。")
    totals = classes.copy()
    corrects = classes.copy()
    totals[0], totals[1] = 0, 0
    corrects[0], corrects[1] = 0, 0
    with torch.no_grad():
        for num, data in enumerate(testloader, 1):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            label = [labels[i].item() for i in labels]
            totals[0] += label.count(0)
            totals[1] += label.count(1)
            predicted = [predicted[i].item() for i in predicted]
            for i in range(len(label)):
                if (label[i] == predicted[i]) and label[i] == 0:
                    corrects[0] += 1
                if (label[i] == predicted[i]) and label[i] == 1:
                    corrects[1] += 1
            if fflag:  # 图片显示模块
                print('标签为:', [labels[i].item() for i in labels])
                print('预测:', [classes[jj] for jj in predicted])
                grid = make_grid(images.to('cpu'))
                plt.imshow(grid.numpy().transpose(1, 2, 0))
                plt.show()
                plt.axis('off')
                plt.ioff()
        print('男人识别准确率:', corrects[0] / totals[0])
        print('女人识别准确率:', corrects[1] / totals[1])


def main():
    if os.path.exists('F:\AI\CNNtorch\gender.pth'):
        net.load_state_dict(torch.load(PATH))
    # train(5, 10)
    # Verify()


if __name__ == "__main__":
    main()

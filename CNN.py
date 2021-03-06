import torch
import matplotlib.pyplot as plt # 可视化
from torch.utils.data import Dataset
from torchvision.transforms import transforms # 图像处理
from DataLookAndStandard import MyDataset # 重写Dataset类
from torchvision.utils import make_grid
import torch.nn as nn  # 神经网络
import torch.optim as optim  # 优化器
import os

fflag = False # True为验证时观看图片
batches = 8
classes = ['男', '女']
PATH = './classfiybaidu.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集加载

trainset = MyDataset('./data/train',
                     './data/train/train.txt',
                     transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                   ]))
testset = MyDataset('./data/test',
                    './data/test/test.txt',
                    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                  ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batches,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batches)


#  卷积网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(  # 特征提取
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            nn.ReLU(inplace=True),  # inplace 原地操作
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
            nn.Dropout(p=0.2),
            # 全连接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
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
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.005,momentum=0.9) # 优化器
    correct = 0
    total = 0
    showaccy = []
    showlossy =[]
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            # 反向传播
            optimizer.zero_grad() # 清除过往梯度
            loss = criterion(outputs, labels) # 损失值计算
            loss.backward() # 反向传播，计算当前梯度
            optimizer.step() # 根据梯度更新网络参数
            # 当前状态
            _, predicted = torch.max(outputs.data, 1)
            if i % verify == 0:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                showaccy.append(correct / total)
                print('[epoch:%d, %5d] loss: %f  准确率:%.4f' % (epoch + 1, i + 1, loss.item(), correct / total))
                showlossy.append(loss.item())
    showaccx = [i for i in range(int(len(showaccy)))]
    # 准确率直线
    plt.subplot(1, 2, 1)
    plt.title("Training real time accuracy")
    plt.xlabel('Batch number of data')
    plt.ylabel('Precision')
    plt.plot(showaccx, showaccy)
    # 损失函数直线
    plt.subplot(1, 2, 2)
    plt.title("Loss value change")
    plt.xlabel('Batch number of data')
    plt.ylabel('Precision')
    plt.plot(showaccx, showlossy)
    plt.show()
    torch.save(net.state_dict(), PATH)
    print('训练完毕')


#  验证集
def Verify():
    # if input('要查看图片吗\n要输入1否则无图片反馈') == '1': fflag = True
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
                print('标签为:', [classes[labels[i].item()] for i in labels])
                print('预测:', [classes[jj] for jj in predicted])
                grid = make_grid(images.to('cpu'))
                plt.imshow(grid.numpy().transpose(1, 2, 0))
                plt.show()
                plt.axis('off')
                plt.ioff()
    if totals[0] != 0:
        print('男人识别准确率:', corrects[0] / totals[0])
    if totals[1] != 0:
        print('女人识别准确率:', corrects[1] / totals[1])


def main():
    from Batchlabeling import batch
    batch('test')
    batch('train')

    if os.path.exists(f'F:\AI\CNNtorch\{PATH[1:]}'):
        net.load_state_dict(torch.load(PATH))
    # train(10, 50)
    Verify()


if __name__ == "__main__":
    main()
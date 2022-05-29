import os

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid # 图片可视化使用
from PIL import Image # 图片读取
from Batchlabeling import batch # 自制打标签模块

class MyDataset(Dataset):
    # stpe1:初始化
    def __init__(self, root, txt, transform=None):
        fh = open(txt, 'r')  # 打开标签文件
        imgs = []  # 创建列表，装东西
        for line in fh:  # 遍历标签文件每行
            line = line.rstrip()  # 删除字符串末尾的空格
            words = line.split()  # 通过空格分割字符串，变成列表
            imgs.append((words[0], int(words[1])))  # 把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transform
        self.root = root

    def __getitem__(self, index):  # 检索函数
        fn, label = self.imgs[index]  # 读取文件名、标签
        img = Image.open(self.root+fn).convert('RGB')  # 通过PIL.Image读取图片
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)



def look(whichset):
    classes = ['男', '女']
    set = MyDataset(f'./data/{whichset}',
                    f'./data/{whichset}/{whichset}.txt',
                    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                  ]))

    setloader = DataLoader(dataset=set,
                           batch_size=4,
                           shuffle=True)

    def show_images_batch(sample_batched):
        images, labels = sample_batched
        images_batch = images
        grid = make_grid(images_batch)
        plt.imshow(grid.numpy().transpose(1, 2, 0))

    plt.figure()
    for i_batch, sample_batch in enumerate(setloader):
        images, labels = sample_batch
        print('标签为:', [classes[j] for j in [labels[i].item() for i in labels]])
        show_images_batch(sample_batch)
        plt.axis('off')
        plt.ioff()
        plt.show()
    plt.show()


if __name__ == "__main__":
    def picstandard(which):
        with open(f'./data/{which}/{which}.txt') as f:
            temp = []
            a = f.read().split('\n')
            a.pop()
            flag = False
            for i in a:
                try:
                    image = Image.open(f'data/{which}' + i[:-2]).convert('RGB')
                    image = np.transpose(image, (2, 0, 1))
                except:
                    flag = True
                    print('无法加载此路径图片，地址为：' + f'data/{which}' + i[:-2])
                    temp.append(f'./data/{which}' + i[:-2])
                if not len(image) == 3:  # 如果图片通道为4则打印出改图片路径
                    flag = True
                    print(f'data/{which}' + i[:-2])
                    temp.append(f'./data/{which}' + i[:-2])
            if flag:
                print('图片通道数有问题!')
            else:
                print('数据集图片无问题')
        with open('F:\AI\CNNTorch\data\problems.txt', 'a') as f:
            f.seek(0)  # 指针定位到0
            f.truncate()  # 清空 文件
            for ii in temp:
                f.writelines(ii)
        print('问题已记录')


    # 数据集可视化
    switch = input('输入1查看数据集\n输入2检测数据集图片标准化\n输入3列出现有问题:')
    if switch  == '1':
        batch('test')
        batch('train')
        set = input('训练集输入1\n否则为测试集')
        ws = 'test'
        if set == '1':
            ws = 'train'
        look(ws)
    elif switch == '2':
        batch('test')
        batch('train')
        set = input('训练集输入1\n否则为测试集')
        ws = 'test'
        if set == '1':
            ws = 'train'
        picstandard(ws)
    else:
        data = None
        with open('F:\AI\CNNTorch\data\problems.txt') as f:
            a = list(f.read().split())
        if len(a) != 0:
            print('问题如下：')
            for i in a:
                print(f'{i}')
            if input('要处理输入1') == '1':
                for i in a:
                    os.remove(i)
                    print(f'{i}已删除')
                print(f'一共处理了{len(a)}张图片')
                with open('F:\AI\CNNTorch\data\problems.txt','a') as up:
                    up.seek(0)  # 指针定位到0
                    up.truncate()  # 清空 文件
                print('问题记录已经清空')
                batch('test')
                batch('train')
                print('标签文件已经更新')
            else:
                print('未处理！')
        else:
            print('无问题记录！')


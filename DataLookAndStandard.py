import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image


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
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

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
    def picstandard():
        with open(r'F:\AI\CNNtorch\data\test\test.txt') as f:
            a = f.read().split('\n')
            a.pop()
            flag = False
            for i in a:
                try:
                    image = io.imread('data/test' + i[:-2])
                except:
                    flag = False
                    print('无法加载此路径图片，地址为：' + 'data/test' + i[:-2])
                image = np.transpose(image, (2, 0, 1))
                if not len(image) == 3:  # 如果图片通道为4则打印出改图片路径
                    flag == True
                    print('data/test' + i[:-2])
            if flag:
                print('图片通道数有问题！\n问题图片路径：' + 'data/test' + i[:-2])
            else:
                print('数据集图片无问题')


    # 数据集可视化

    if input('输入1查看数据集\n否则将检测数据集图篇标准化:') == '1':
        set = input('训练集输入1\n否则为测试集')
        ws = 'test'
        if set == '1':
            ws = 'train'
        look(ws)
    else:
        picstandard()

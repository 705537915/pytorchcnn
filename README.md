# 男女分类项目

## 项目实现分为六个阶段

### 环境配置

运行环境：windows10

使用的开发环境：Pycharm

虚拟环境搭建：anaconda

网络框架：pytorch

版本管理工具：git

使用gpu训练

### 数据集

#### 数据集加载

数据集来源百度

数据集使用到了torch自带的torchvison包

重写Dataset类，使用txt文件加载图片 与图片打标

##### 训练集图片的相对地址与对应的标签写入txt文件

执行的此步骤的python程序为Batch_labeling

图片批量重命名程序rename

数据集预处理使用torchvision的transforms工具包

对数据做了 裁剪， 张量化，以及

##### 归一化处理( 图片张量值在（-1，1）之间)

```
torchvision.transforms.ToTensor()，其作用是将数据归一化到[0,1]（是将数据除以255），会把HWC会变成C *H *W（拓展：格式为(h,w,c)，像素顺序为RGB）
transforms.Normalize((mean1,mean2,mean3),(std1,std2,std3))
output = (input - mean) / std
```

### 网络搭建

ouput计算公式![image-20220529074505767](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529074505767.png)



使用的是sequential 方法打包网络的特征提取层以及全连接层

网络是**复现**Hinton和他的学生Alex Krizhevsky设计的Alexnet



#### Alexnet

![image-20220529072316462](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529072316462.png)

2012年imagene冠军

使用了Relu激活函数

最大池化池化方法

使用了Dropout层

整个网络由**5层卷积层**和**3层全连接层**构成





### 可视化

有想过使用pytorch自带的tensorboard 最终选择了matplotlib 工具包绘制

使用了matplotlib的pyplot工具包

可视化包括将图片展示，准确率展示，损失值展示，预测标签展示

#### 曲线

matplotlib的列表绘制

![image-20220528183055393](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528183055393.png)

![image-20220529084211651](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529084211651.png)



### 参数调整

alexnet的性能优秀 参数保持保持默认

![image-20220528183634572](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528183634572.png)

### 上传至github

**git项目管理**使用到了pycharm开发环境提供的git管理工具

![image-20220528211109794](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528211109794.png)由于pycharm git管理使用不熟练，导致中途两次项目文件被不可逆误删，**血的教训**



## 项目运用到的技术解读

### 导入的模块解读

torch：pytorch框架

**matplotlib**: Python 的 2D绘图库，它以各种硬拷贝格式和跨平台的交互式环境生成出版质量级别的图形。

torchvision:处理图像视频，包含一些常用的数据集、模型、转换函数,本次项目运用到了

transforms：Pytorch中的图像预处理包

PIL （pillow）：是python图片处理的基础库，pillow库中的image对象能够与Numpy ndarry数组实现相互转换

#### 优化器，损失函数

##### 优化器

![image-20220529080502552](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529080502552.png)

当我们使用SGD会把数据拆分后再分批不断放入 NN 中计算. 每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率.

###### lr

学习率

###### momentum

为了抑制SGD的震荡，梯度下降过程可以加入惯性。下坡的时候，如果发现是陡坡，那就利用惯性跑的快一些。

##### 损失函数

![image-20220529081916733](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529081916733.png)



## 结果展示

训练结果：

![image-20220528233048993](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528233048993.png)

![image-20220528232941323](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528232941323.png)

![image-20220528212239675](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528212239675.png)

![image-20220529082109790](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529082109790.png)



![image-20220529082136195](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529082136195.png)



## 遇到的问题

数据标准化时遇到的问题

![image-20220528224101881](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528224101881.png)

处理办法，第一次运行记录问题

![image-20220529072703660](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529072703660.png)

第二次运行处理问题

![image-20220529072720057](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220529072720057.png)

## 其他设备的实现

使用anaconda配置pytorch虚拟环境

建议python版本python=3.7

基础训练集已打包：天翼云盘：https://cloud.189.cn/t/Vv6RN3RrYb22 (访问码:3ngp)

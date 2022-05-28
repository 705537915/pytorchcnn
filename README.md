# 男女分类项目

## 项目实现分为六个阶段

### 环境配置

运行环境：windows10

使用的开发环境：Pycharm

虚拟环境搭建：anaconda

网络框架：pytorch

版本管理工具：git

### 数据集

#### 数据集加载

数据集来源百度

数据集使用到了torch自带的torchvison包

重写Dataset类，使用txt文件加载图片 与图片打标

##### 训练集图片的相对地址与对应的标签写入txt文件

执行的此步骤的python程序为Batch_labeling

图片批量重命名程序rename

数据集预处理使用torchvision的transforms工具包

对数据做了 裁剪， 张量化，以及归一化处理

### 网络搭建

使用的是sequential 方法打包网络的特征提取层以及全连接层

网络是复现Hinton和他的学生Alex Krizhevsky设计的Alexnet

#### Alexnet

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

![image-20220528183505354](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528183505354.png)



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



## 结果展示

![image-20220528234843308](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528234843308.png)

![image-20220528233048993](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528233048993.png)

![image-20220528232941323](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528232941323.png)

![image-20220528212239675](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528212239675.png)







## 遇到的问题

![image-20220528224101881](D:/OneDrive - vhdsr/Typroa/大一下/2/image-20220528224101881.png)


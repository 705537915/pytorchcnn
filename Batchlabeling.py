import os


# 打标签 which 指那个数据集
def batch(which):
    with open(f'F:/AI/CNNtorch/data/{which}/{which}.txt', 'a') as fd:
        man = f'F:/AI/CNNtorch/data/{which}/man'
        woman = f'F:/AI/CNNtorch/data/{which}/woman'
        a = os.listdir(str(man))
        b = os.listdir(str(woman))
        fd.seek(0)  # 指针定位到0
        fd.truncate()  # 清空 文件
        for i in a:
            fd.writelines(f'/man/{i} 0 \n')
        for i in b:
            fd.writelines(f'/woman/{i} 1 \n')
        print(f'{which}打标签完成!')


if __name__ == "__main__":
    if input('测试集打标输入1\n否则为训练集:') == '1':
        batch('test')
    else:
        batch('train')

import numpy as np
import torchvision # pytorch对CV的库
from torchvision import transforms # 对数据操作
from torch.utils import data # 读取小批量数据的函数

from useTool import get_dataloader_workers

def load_data_fashion_mnist(batch_size, resize=None): 
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, 
                            num_workers=get_dataloader_workers()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, 
                            num_workers=get_dataloader_workers()))

def get_fashion_mnist_labels(labels): 
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
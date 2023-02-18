import numpy as np
import torch


from useTool import Accumulator, evaluate_accuracy, accuracy, show_images
from drawPlot import Animator

def train_epoch_ch3(net, train_iter, loss, updater): 
    """训练模型⼀个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module): # 如果使用nn.Module
        net.train() # 开启训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3) # 使用长度为3的迭代器
    for X, y in train_iter: # 扫一遍数据
        # 计算梯度并更新参数
        y_hat = net(X) # 先计算y_hat
        l = loss(y_hat, y) # 交叉熵损失
        if isinstance(updater, torch.optim.Optimizer):
            # 使⽤PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使⽤定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): 
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], 
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


from FashionMnistTool import get_fashion_mnist_labels

def predict_ch3(net, test_iter, n=6): 
    """预测标签"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
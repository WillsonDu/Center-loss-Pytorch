import torch
from center_loss.my_own_code.my_dataset import MNIST
from center_loss.my_own_code.my_center_loss import CenterLoss
from center_loss.my_own_code.my_model import Net
import matplotlib.pyplot as plt
import numpy as np

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'red']

if __name__ == '__main__':

    net = Net()
    mnist = MNIST(batch_size=100)

    is_train_center_loss = True  # 是否训练center_loss
    is_load_pkfile = True  # 是否读取保存的权重文件
    is_save_pkfile = True  # 是否保存权重文件
    is_print_test_accurancy = False  # 是否打印测试数据集的正确率

    optmizer = torch.optim.Adam(net.parameters())

    net_pk_path = "./net/net.pt"
    ct_net_pk_path = "./ct_net/ct_net.pt"

    if is_load_pkfile:
        try:
            net.load_state_dict(torch.load(net_pk_path))
        except:
            pass

    if is_train_center_loss:
        center_loss_fn = CenterLoss(10, 2)

        if is_load_pkfile:
            try:
                center_loss_fn.load_state_dict(torch.load(ct_net_pk_path))
            except:
                pass

        center_loss_weight = 1
        optimizer_center_loss = torch.optim.Adam(center_loss_fn.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()

    for iii in range(100):

        all_features = []
        all_lables = []

        for index, (xs, ys) in enumerate(mnist.train_dataloader):
            # x.shape=(batch,28,28,1)  y.shape=(batch,)
            xs = xs.view(-1, 784)
            features, lables = net(xs)

            loss = loss_fn(lables, ys)  # 实际值和预测值的损失(经过softmax)

            if is_train_center_loss:
                center_loss = center_loss_fn(features, ys)  # center_loss损失
                center_loss = center_loss * center_loss_weight  # center_loss_weight在0到1之间，越大类间距越大
                loss = loss + center_loss

            print(loss)

            # 梯度清零
            optmizer.zero_grad()
            if is_train_center_loss:
                optimizer_center_loss.zero_grad()

            # 反向传播
            loss.backward()

            optmizer.step()
            if is_train_center_loss:
                for param in center_loss_fn.parameters():
                    param.grad.data *= (1. / center_loss_weight)  # 权重回归
                optimizer_center_loss.step()

            all_features.append(features.data.numpy())
            all_lables.append(ys)

        if is_save_pkfile:
            torch.save(net.state_dict(), net_pk_path)
            if is_train_center_loss:
                torch.save(center_loss_fn.state_dict(), ct_net_pk_path)

        np_all_features = np.stack(all_features)
        np_all_lables = np.stack(all_lables)

    #     # 绘图
    #     plt.figure(iii)
    #
    #     for ii in range(10):
    #         plt.scatter(
    #             np_all_features[ii == np_all_lables, 0],
    #             np_all_features[ii == np_all_lables, 1],
    #             c=colors[ii],
    #             s=1
    #         )
    #     plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    #     plt.pause(0.1)
    #     # plt.show()
    #     plt.ioff()  # 释放资源
    # plt.show()  # 不加这句的话，等全部窗体绘制好以后，所有图片窗体都会消失。加上plt.show()以阻塞之

    if is_print_test_accurancy:
        for index, (xs, ys) in enumerate(mnist.test_dataloader):
            xs = xs.view(-1, 784)
            features, labels = net(xs)

            # loss = loss_fn(lables, ys)  # 实际值和预测值的损失(经过softmax)
            #
            # if is_train_center_loss:
            #     center_loss = center_loss_fn(features, ys)  # center_loss损失
            #     center_loss = center_loss * center_loss_weight  # center_loss_weight在0到1之间，越大类间距越大
            #     loss = loss + center_loss

            for i_ in range(len(ys)):
                print("实际值：", ys[i_])
                print("预测值：", torch.argmax(labels[i_], dim=0))
                print("---------------------------")

            break

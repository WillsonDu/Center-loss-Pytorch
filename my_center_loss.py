import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import torch.nn as nn


class CenterLoss(torch.nn.Module):
    """Center loss.

        Reference:
        Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=False):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        # 网络代码(没看懂)

        # batch_size = x.size(0)
        # distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        #           torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        #
        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #
        # dist = []
        # for i in range(batch_size):
        #     value = distmat[i][mask[i]]
        #     value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #     dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()

        # 个人代码
        labels = labels.view(-1)

        # 每个feature隶属于的中心点坐标(因为可能有多个feature同属一个中心点，所以这里会有多个相同的数据)
        center_xy = self.centers.index_select(dim=0, index=labels.long())  # index_select方法只接受LongTensor

        count = torch.histc(labels.float(), bins=10, min=0, max=9)  # 计算从0到9，每个类别的数量 (histc方法只接受FloatTensor)
        ccc = count.index_select(dim=0, index=labels.long())  # 将count按label展开

        value = (x - center_xy) ** 2  # (x1-center_x) ** 2, (y1 - center_y) ** 2
        value = torch.sum(value, 1)  # (x1-center_x) ** 2 + (y1 - center_y) ** 2
        value = torch.sqrt(value)  # sqrt((x1-center_x) ** 2 + (y1 - center_y) ** 2)
        value = value / ccc
        loss = torch.sum(value)
        return loss


if __name__ == '__main__':
    pass

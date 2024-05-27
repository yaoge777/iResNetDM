import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = nn.Parameter(torch.ones(num_classes, 1))
        else:
           
            self.alpha = nn.Parameter(alpha)
        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size, 1]

        num, C = inputs.size()


        # P = F.softmax(inputs, dim=1)  # 计算类别概率
        class_mask = inputs.data.new(num, C).fill_(0)  # 创建零张量作为类别掩码

        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 创建 one-hot 编码的类别掩码

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = nn.Parameter(self.alpha.cuda())

        alpha = self.alpha[ids.data.view(-1)]
        probs = (inputs * class_mask).sum(1).view(-1, 1)  # 计算类别概率和类别掩码的乘积
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p  # 计算 Focal Loss

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

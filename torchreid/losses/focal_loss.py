from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Focal Loss for binary classification
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, use_gpu=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.use_gpu = use_gpu
#         self.ce = nn.CrossEntropyLoss()
#
#     def forward(self, inputs, targets):
#         logp = self.ce(inputs, targets)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp
#         return loss.mean()


# Focal Loss for multi-class classification
def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0).cuda()
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1).cuda())
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum(dim=1).mean()


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = nn.functional.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    return torch.sum(alpha * (modulator * BCLoss)) / torch.sum(labels)


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Loss Based on Effective Number of Samples. In CVPR, 2019.
    https://github.com/richardaecn/class-balanced-loss
    """
    def __init__(self, samples_per_cls, no_of_classes, gamma=2, beta=0.9999, use_gpu=True, loss_type='focal'):
        """
        Compute the Class Balanced Loss between `label` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, label)
        where Loss is one of the standard losses used for Neural Networks.

        :param samples_per_cls: A python list of size [no_of_classes].
        :param no_of_classes: total number of classes. int
        :param gamma: float. Hyperparameter for Focal loss.
        :param beta: float. Hyperparameter for Class balanced loss.
        :param use_gpu:
        """
        super(CBFocalLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.gamma = gamma
        self.beta = beta
        self.use_gpu = use_gpu
        self.loss_type = loss_type

    def forward(self, inputs, targets):
        labels_one_hot = nn.functional.one_hot(targets, self.no_of_classes).float()

        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / torch.sum(weights) * self.no_of_classes
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "sigmoid":
            cb_loss = nn.functional.binary_cross_entropy_with_logits(input=inputs, target=labels_one_hot, weight=weights)
        else:
            cb_loss = focal_loss(labels_one_hot, inputs, weights, self.gamma)
        return cb_loss

from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


class ALSRLoss(nn.Module):
    r""" Adaptive Label Amoothing Regularization Loss.
    
        Reference:
            Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

        With label smoothing, the label :math:`y` for a class is computed by

        .. math::
            \begin{equation}
            (1 - \eps) \times y + \frac{\eps}{K},
            \end{equation}

        where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
        :math:`\eps = 0`, the loss function reduces to the normal cross entropy.

        Args:
            num_classes (int): number of classes.
            eps (float, optional): weight. Default is 0.1.
            use_gpu (bool, optional): whether to use gpu devices. Default is True.
            label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, eps=0.1, alpha=0.2, use_gpu=True, label_smooth=True):
        super(ALSRLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.eps = eps if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_pids * num_vids).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        pids, vids = targets
        log_probs = self.logsoftmax(inputs)
        probs = torch.exp(log_probs)
        ep1s, ep2s = torch.zeros(pids.size(0), 1), torch.zeros(pids.size(0), 1)
        for i in range(pids.size(0)):
            pid, vid = pids[i], vids[i]
            ep1s[i, 1] = torch.sum(probs[i, pid*3:pid*3+3])
            ep2s[i, 1] = probs[i, pid*3+vid]
        ep1s = self.alpha * (1 - ep1s)
        ep2s = self.alpha * (1 - ep2s)
        ones = torch.ones(log_probs.size()) * ep1s / (self.num_classes - 3)
        ones = ones.scatter_(1, (pids * 3).unsqueeze(1).data.cpu(), 0.5 * ep2s)
        ones = ones.scatter_(1, (pids * 3 + 1).unsqueeze(1).data.cpu(), 0.5 * ep2s)
        ones = ones.scatter_(1, (pids * 3 + 2).unsqueeze(1).data.cpu(), 0.5 * ep2s)
        targets = ones.scatter_(1, (pids * 3 + vids).unsqueeze(1).data.cpu(), 1 - ep1s - ep2s)

        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        return (-targets * log_probs).mean(0).sum()

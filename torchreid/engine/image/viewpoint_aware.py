from __future__ import division, print_function, absolute_import

from ... import metrics
from ...engine.engine import Engine
from ...losses import FocalLoss, CrossEntropyLoss, ALSRLoss
from .vat import VATLoss

import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn


class ImageVAReIDEngine(Engine):
    r"""Viewpoint-Aware Loss with Angular Regularization engine for image-reid.

    Ref: Viewpoint-Aware Loss with Angular Regularization for Person Re-Identification. AAAI, 2020.
    https://arxiv.org/pdf/1912.01300v1.pdf

    Args:
        datamanager (DataManager): an instance of ``deepreid.data.ImageDataManager``
            or ``deepreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        weight_f (float, optional): weight for focal loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import deepreid
        datamanager = deepreid.data.ImageDataManager(def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        ans =[]
        window = nums[0:k]
        window.sort()
        median = nums[k-1-k//2] if k%2 == 1 else (nums[k-1-k//2] + nums[k-1-k//2+1]) / 2
        ans.append(median)
        for ind in range(k, len(nums)):
            window.remove(nums[ind-k])
            bisect_left(window, nums[ind])
            median = nums[ind-k//2] if k%2 == 1 else (nums[ind-k//2] + nums[ind-k//2+1]) / 2
            ans.append(median)
        return ans
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = deepreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = deepreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = deepreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = deepreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            arc_margin_y,
            arc_margin_v,
            optimizer,
            gamma=2,
            weight_f=1,
            weight_x=1,
            weight_v=1,
            scheduler=None,
            use_gpu=True,
    ):
        super(ImageVAReIDEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.weight_f = weight_f
        self.weight_x = weight_x
        self.weight_v = weight_v

        self.arc_embed_y = arc_margin_y
        self.arc_embed_v = arc_margin_v
        self.criterion_x = CrossEntropyLoss(num_classes=self.datamanager.num_train_pids,
                                            use_gpu=self.use_gpu,
                                            label_smooth=True)
        self.criterion_f = FocalLoss(gamma=gamma)
        self.criterion_v = ALSRLoss(num_classes=self.datamanager.num_train_pids,
                                    use_gpu=self.use_gpu,
                                    label_smooth=True)
        self.centers_yv = torch.zeros(self.datamanager.num_train_pids, 3, 512)
        self.counts_yv = torch.zeros(self.datamanager.num_train_pids, 3)

    def forward_backward(self, data):
        imgs, pids, vids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            vids = pids.cuda()

        outputs, features = self.model(imgs)
        embeddings_y = self.arc_embed_y(features, pids)
        embeddings_v = self.arc_embed_v(features, pids*3+vids, weight=self.centers_yv.view(-1, 512))

        loss_x = self.compute_loss(self.criterion_x, outputs, pids)
        loss_f = self.compute_loss(self.criterion_f, embeddings_y, pids)
        loss_v = self.compute_loss(self.criterion_v, embeddings_v, (pids, vids))
        loss = self.weight_f * loss_f + self.weight_x * loss_x + self.weight_v * loss_v

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update self.centers_yv & self.counts_yv
        for i in range(pids.size(0)):
            self.counts_yv[pids[i], vids[i]] += 1
            tmp = self.counts_yv[pids[i], vids[i]]
            self.centers_yv[pids[i], vids[i]] = (tmp-1/tmp) * self.centers_yv[pids[i], vids[i]] + 1/tmp * features[i]

        loss_summary = {'loss_x': loss_x.item(),
                        'loss_f': loss_f.item(),
                        'loss_v': loss_v.item(),
                        'acc_x': metrics.accuracy(outputs, pids)[0].item(),
                        'acc_f': metrics.accuracy(embeddings_y, pids)[0].item(),
                        }
        return loss_summary

    def forward(self, imgs, pids):
        indexs = torch.where(pids < self.arc_embed.out_features)
        imgs, pids = imgs[indexs], pids[indexs]
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        if imgs.shape[0] == 0:
            return None, None, None, None

        outputs, features = self.model(imgs)
        embeddings_y = self.arc_embed_y(features, pids)
        embeddings_v = self.arc_embed_v(features, pids)
        loss_x = self.compute_loss(self.criterion_x, outputs, pids).item()
        loss_f = self.compute_loss(self.criterion_f, embeddings_y, pids).item()
        loss_v = self.compute_loss(self.criterion_f, embeddings_v, pids).item()
        acc_x = metrics.accuracy(outputs, pids)[0].item()
        acc_f = metrics.accuracy(embeddings_y, pids)[0].item()
        return loss_x, loss_f, loss_v, acc_x, acc_f

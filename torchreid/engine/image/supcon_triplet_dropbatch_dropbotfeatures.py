from __future__ import absolute_import, print_function, division

import torch
import torch.nn.functional as F

from ..engine import Engine
from ...losses import CrossEntropyLoss, TripletLoss, SupConLoss
from ... import metrics


class ImageSupConTripletDropBatchDropBotFeaturesEngine(Engine):
    r"""Supervised Contrastive Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``deepreid.data.ImageDataManager``
            or ``deepreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torch
        import deepreid
        datamanager = deepreid.data.ImageDataManager(
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

    def __init__(self, datamanager, model, optimizer, metric_loss, margin=0.3,
                 weight_t=1, weight_x=1, weight_db_f=1, weight_db_t=1,
                 weight_db_x=1, weight_b_db_f=1, weight_b_db_t=1, weight_b_db_x=1,
                 scheduler=None, use_gpu=True, label_smooth=True, top_drop_epoch=-1):
        super(ImageSupConTripletDropBatchDropBotFeaturesEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_db_f = weight_db_f
        self.weight_db_t = weight_db_t
        self.weight_db_x = weight_db_x
        self.weight_b_db_f = weight_b_db_f
        self.weight_b_db_t = weight_b_db_t
        self.weight_b_db_x = weight_b_db_x
        self.top_drop_epoch = top_drop_epoch

        # Loss for SupContrastLoss
        self.criterion_s = SupConLoss(temperature=metric_loss['supcon']['temperature'],
                                      base_temperature=metric_loss['supcon']['base_temperature'])
        self.weight_s = metric_loss['supcon']['weight']

        # Loss for global and BatchDrop, incl., TripletLoss and CrossEntropyLoss
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_db_t = TripletLoss(margin=margin)
        self.criterion_db_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_b_db_t = TripletLoss(margin=margin)
        self.criterion_b_db_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data, epoch=0):
        [imgs_v1, imgs_v2, imgs], pids = self._parse_data_for_train(data)
        if self.use_gpu:
            imgs_v1, imgs_v2 = imgs_v1.cuda(), imgs_v2.cuda()
            pids = pids.cuda()

        drop_top = (self.top_drop_epoch != -1) and ((epoch + 1) >= self.top_drop_epoch)
        outputs1, feat1, db_prelogits1, db_feat1, b_db_prelogits1, b_db_feat1 = self.model(imgs_v1, drop_top=drop_top)
        outputs2, feat2, db_prelogits2, db_feat2, b_db_prelogits2, b_db_feat2 = self.model(imgs_v2, drop_top=drop_top)

        # SelfContrast loss for outputs
        noutputs1, noutputs2 = F.normalize(outputs1, dim=1), F.normalize(outputs2, dim=1)
        feature_views = torch.cat([noutputs1.unsqueeze(1), noutputs2.unsqueeze(1)], dim=1)
        loss_s = self._compute_loss(self.criterion_s, feature_views, pids)
        #
        # # SelfContrast loss for db_prelogits1
        # ndb_prelogits1, ndb_prelogits2 = F.normalize(db_prelogits1, dim=1), F.normalize(db_prelogits2, dim=1)
        # feature_views = torch.cat([ndb_prelogits1.unsqueeze(1), ndb_prelogits2.unsqueeze(1)], dim=1)
        # loss_s += self._compute_loss(self.criterion_s, feature_views, pids)
        #
        # # SelfContrast loss for b_db_prelogits1
        # nb_db_prelogits1, nb_db_prelogits2 = F.normalize(b_db_prelogits1, dim=1), F.normalize(b_db_prelogits2, dim=1)
        # feature_views = torch.cat([nb_db_prelogits1.unsqueeze(1), nb_db_prelogits2.unsqueeze(1)], dim=1)
        # loss_s += self._compute_loss(self.criterion_s, feature_views, pids)

        # Global loss
        loss_x = 0.5 * (self._compute_loss(self.criterion_x, outputs1, pids) +
                        self._compute_loss(self.criterion_x, outputs2, pids))
        loss_t = 0.5 * (self._compute_loss(self.criterion_t, feat1, pids) +
                        self._compute_loss(self.criterion_t, feat2, pids))
        # DropBot loss
        loss_db_x = 0.5 * (self._compute_loss(self.criterion_db_x, db_prelogits1, pids) +
                           self._compute_loss(self.criterion_db_x, db_prelogits2, pids))
        loss_db_t = 0.5 * (self._compute_loss(self.criterion_db_t, db_feat1, pids) +
                           self._compute_loss(self.criterion_db_t, db_feat2, pids))
        # Batch DropBot loss
        loss_b_db_x = 0.5 * (self._compute_loss(self.criterion_b_db_x, b_db_prelogits1, pids) +
                             self._compute_loss(self.criterion_b_db_x, b_db_prelogits2, pids))
        loss_b_db_t = 0.5 * (self._compute_loss(self.criterion_b_db_t, b_db_feat1, pids) +
                             self._compute_loss(self.criterion_b_db_t, b_db_feat2, pids))

        loss = self.weight_s * loss_s + \
               self.weight_t * loss_t + self.weight_x * loss_x + \
               self.weight_db_t * loss_db_t + self.weight_db_x * loss_db_x + \
               self.weight_b_db_t * loss_b_db_t + self.weight_b_db_x * loss_b_db_x

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss_s': loss_s.item(),
            'loss_t': loss_t.item(),
            'loss_x': loss_x.item(),
            'loss_db_t': loss_db_t.item(),
            'loss_db_x': loss_db_x.item(),
            'loss_b_db_t': loss_b_db_t.item(),
            'loss_b_db_x': loss_b_db_x.item(),
            'acc_x': metrics.accuracy(outputs1, pids)[0].item()
        }

        return loss_summary

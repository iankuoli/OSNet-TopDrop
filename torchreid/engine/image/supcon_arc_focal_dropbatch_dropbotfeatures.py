from __future__ import absolute_import, print_function, division

import torch
import torch.nn.functional as F

from ..engine import Engine
from ...losses import CrossEntropyLoss, TripletLoss, FocalLoss, CBFocalLoss, LDAMLoss, SupConLoss
from ... import metrics


class ImageSupConArcFocalDropBatchDropBotFeaturesEngine(Engine):
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

    def __init__(self, datamanager, model, optimizer, metric_loss,
                 arc_margin, arc_margin_f, arc_margin_db_f, arc_margin_b_db_f, gamma=2.0,
                 weight_all_f=1, weight_f=1, weight_x=1,
                 weight_db_f=1, weight_db_x=1, weight_b_db_f=1, weight_b_db_x=1,
                 scheduler=None, use_gpu=True, label_smooth=True, top_drop_epoch=-1):
        super(ImageSupConArcFocalDropBatchDropBotFeaturesEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_all_f = weight_all_f
        self.weight_f = weight_f
        self.weight_x = weight_x
        self.weight_db_f = weight_db_f
        self.weight_db_x = weight_db_x
        self.weight_b_db_f = weight_b_db_f
        self.weight_b_db_x = weight_b_db_x
        self.top_drop_epoch = top_drop_epoch

        self.arc = arc_margin
        self.arc_f = arc_margin_f
        self.arc_db_f = arc_margin_db_f
        self.arc_b_db_f = arc_margin_b_db_f

        # Loss for SupContrastLoss
        self.criterion_s = SupConLoss(temperature=metric_loss['supcon']['temperature'],
                                      base_temperature=metric_loss['supcon']['base_temperature'])
        self.weight_s = metric_loss['supcon']['weight']

        # Loss for metric learning based on ArcMargin, incl., CrossEntropy, FocalLoss, CBFocalLoss, LDAMLoss
        if metric_loss['loss'] == 'ce':
            self.criterion_all_f = CrossEntropyLoss(num_classes=self.datamanager.num_train_pids,
                                                    use_gpu=self.use_gpu,
                                                    label_smooth=label_smooth)
        elif metric_loss['loss'] == 'focal':
            self.criterion_all_f = FocalLoss(gamma=metric_loss['gamma'])
        elif metric_loss['loss'] == 'cbfocal' or metric_loss['loss'] == 'ldam':
            samples_per_cls = torch.zeros(datamanager.num_train_pids).cuda()
            for batch_idx, data in enumerate(self.train_loader):
                _, pids = self._parse_data_for_train(data)
                labels_one_hot = torch.sum(F.one_hot(pids, datamanager.num_train_pids).float().cuda(), 0)
                samples_per_cls += labels_one_hot
            if metric_loss['loss'] == 'cbfocal':
                self.criterion_all_f = CBFocalLoss(samples_per_cls=samples_per_cls,
                                                   no_of_classes=datamanager.num_train_pids,
                                                   gamma=metric_loss['gamma'], beta=metric_loss['beta'])
            else:
                self.criterion_all_f = LDAMLoss(cls_num_list=samples_per_cls,
                                                s=metric_loss['s'], max_m=metric_loss['max_m'])
        else:
            exit("Unidentified loss name.")

        # Loss for global and BatchDrop, incl., CrossEntropyLoss and CrossEntropyLoss
        self.criterion_f = FocalLoss(gamma=gamma)
        self.criterion_x = CrossEntropyLoss(num_classes=self.datamanager.num_train_pids,
                                            use_gpu=self.use_gpu,
                                            label_smooth=label_smooth)
        self.criterion_db_f = FocalLoss(gamma=gamma)
        self.criterion_db_x = CrossEntropyLoss(num_classes=self.datamanager.num_train_pids,
                                               use_gpu=self.use_gpu,
                                               label_smooth=label_smooth)
        self.criterion_b_db_f = FocalLoss(gamma=gamma)
        self.criterion_b_db_x = CrossEntropyLoss(num_classes=self.datamanager.num_train_pids,
                                                 use_gpu=self.use_gpu,
                                                 label_smooth=label_smooth)

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

        # Metric learning loss
        embed1 = self.arc(torch.cat((feat1, db_feat1), dim=1), pids),
        embed2 = self.arc(torch.cat((feat2, db_feat2), dim=1), pids)
        loss_all_f = 0.5 * (self._compute_loss(self.criterion_all_f, embed1, pids) +
                            self._compute_loss(self.criterion_all_f, embed2, pids))
        # Global loss
        embed1_f, embed2_f = self.arc_f(feat1, pids), self.arc_f(feat2, pids)
        loss_f = 0.5 * (self._compute_loss(self.criterion_f, embed1_f, pids) +
                        self._compute_loss(self.criterion_f, embed2_f, pids))
        loss_x = 0.5 * (self._compute_loss(self.criterion_x, outputs1, pids) +
                        self._compute_loss(self.criterion_x, outputs2, pids))
        # DropBot loss
        embed1_db_f, embed2_db_f = self.arc_db_f(db_feat1, pids), self.arc_db_f(db_feat2, pids)
        loss_db_f = 0.5 * (self._compute_loss(self.criterion_db_f, embed1_db_f, pids) +
                           self._compute_loss(self.criterion_db_f, embed2_db_f, pids))
        loss_db_x = 0.5 * (self._compute_loss(self.criterion_db_x, db_prelogits1, pids) +
                           self._compute_loss(self.criterion_db_x, db_prelogits2, pids))
        # Batch DropBot loss
        embed1_b_db_f, embed2_b_db_f = self.arc_b_db_f(b_db_feat1, pids), self.arc_b_db_f(b_db_feat2, pids)
        loss_b_db_f = 0.5 * (self._compute_loss(self.criterion_b_db_f, embed1_b_db_f, pids) +
                             self._compute_loss(self.criterion_b_db_f, embed2_b_db_f, pids))
        loss_b_db_x = 0.5 * (self._compute_loss(self.criterion_b_db_x, b_db_prelogits1, pids) +
                             self._compute_loss(self.criterion_b_db_x, b_db_prelogits2, pids))

        loss = self.weight_s * loss_s + self.weight_all_f * loss_all_f + \
               self.weight_f * loss_f + self.weight_x * loss_x + \
               self.weight_db_f * loss_db_f + self.weight_db_x * loss_db_x + \
               self.weight_b_db_f * loss_b_db_f + self.weight_b_db_x * loss_b_db_x

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss_s': loss_s.item(),
            'loss_all_f': loss_all_f.item(),
            'loss_f': loss_f.item(),
            'loss_x': loss_x.item(),
            'loss_db_f': loss_db_f.item(),
            'loss_db_x': loss_db_x.item(),
            'loss_b_db_f': loss_b_db_f.item(),
            'loss_b_db_x': loss_b_db_x.item(),
            'acc_f': metrics.accuracy(embed1, pids)[0].item(),
            'acc_x': metrics.accuracy(outputs1, pids)[0].item()
        }

        return loss_summary

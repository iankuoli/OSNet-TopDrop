from __future__ import absolute_import, print_function, division

from ..engine import Engine
from ...losses import CrossEntropyLoss, TripletLoss
from ... import metrics


class ImageTripletDropBatchDropBotFeaturesEngine(Engine):
    r"""Triplet-loss engine for image-reid.

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

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, weight_db_t=1, weight_db_x=1, weight_b_db_t=1, weight_b_db_x=1, scheduler=None, use_gpu=True,
                 label_smooth=True, top_drop_epoch=-1):
        super(ImageTripletDropBatchDropBotFeaturesEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_db_t = weight_db_t
        self.weight_db_x = weight_db_x
        self.weight_b_db_t = weight_b_db_t
        self.weight_b_db_x = weight_b_db_x
        self.top_drop_epoch = top_drop_epoch

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
        imgs, pids = self._parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        drop_top = (self.top_drop_epoch != -1) and ((epoch + 1) >= self.top_drop_epoch)
        outputs, features, db_prelogits, db_features, b_db_prelogits, b_db_features = self.model(imgs, drop_top=drop_top)
        loss_t = self._compute_loss(self.criterion_t, features, pids)
        loss_x = self._compute_loss(self.criterion_x, outputs, pids)
        loss_db_x = self._compute_loss(self.criterion_db_x, db_prelogits, pids)
        loss_db_t = self._compute_loss(self.criterion_db_t, db_features, pids)
        loss_b_db_x = self._compute_loss(self.criterion_b_db_x, b_db_prelogits, pids)
        loss_b_db_t = self._compute_loss(self.criterion_b_db_t, b_db_features, pids)
        loss = self.weight_t * loss_t + self.weight_x * loss_x + \
               self.weight_db_t * loss_db_t + self.weight_db_x * loss_db_x + \
               self.weight_b_db_t * loss_b_db_t + self.weight_b_db_x * loss_b_db_x

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss_t': loss_t.item(),
            'loss_x': loss_x.item(),
            'loss_db_t': loss_db_t.item(),
            'loss_db_x': loss_db_x.item(),
            'loss_b_db_t': loss_b_db_t.item(),
            'loss_b_db_x': loss_b_db_x.item(),
            'acc_x': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary

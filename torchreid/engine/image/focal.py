from __future__ import division, print_function, absolute_import

from ...engine import Engine
from ...losses import FocalLoss, CrossEntropyLoss
from ... import metrics


class ImageFocalEngine(Engine):
    r"""Focal loss engine for image-reid.

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

    def __init__(self, datamanager, model, optimizer, arc_margin,
                 scheduler=None, use_gpu=True, label_smooth=True, weight_f=1., weight_x=1., gamma=2.):
        super(ImageFocalEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.output_num = datamanager.num_train_pids

        self.weight_f = weight_f
        self.weight_x = weight_x

        self.arc_embed = arc_margin
        self.criterion_f = FocalLoss(gamma=gamma)
        self.criterion_x = CrossEntropyLoss(num_classes=self.datamanager.num_train_pids,
                                            use_gpu=self.use_gpu,
                                            label_smooth=label_smooth)

    def forward_backward(self, data, epoch=0):
        imgs, pids = self._parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)
        embeddings = self.arc_embed(features, pids)
        loss_f = self._compute_loss(self.criterion_f, embeddings, pids)
        loss_x = self._compute_loss(self.criterion_x, outputs, pids)
        loss = self.weight_f * loss_f + self.weight_x * loss_x

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {'loss_f': loss_f.item(),
                        'loss_x': loss_x.item(),
                        'acc_f': metrics.accuracy(embeddings, pids)[0].item(),
                        'acc_x': metrics.accuracy(outputs, pids)[0].item()}
        return loss_summary

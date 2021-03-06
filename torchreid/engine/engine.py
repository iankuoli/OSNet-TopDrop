from __future__ import absolute_import, print_function, division

import os.path as osp
import time
import datetime
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from ..utils import (
    MetricMeter, AverageMeter, open_specified_layers, open_all_layers, visualize_ranked_results,
    save_checkpoint, re_ranking, mkdir_if_missing, visualize_ranked_activation_results,
    visualize_ranked_threshold_activation_results, visualize_ranked_mask_activation_results)
from ..losses import DeepSupervision
from .. import metrics


GRID_SPACING = 10
VECT_HEIGHT = 10


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``deepreid.data.ImageDataManager``
            or ``deepreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader, self.test_loader = self.datamanager.return_dataloaders()
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.max_epoch = 0

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(self, aim_sess, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, test_only=False, print_freq=10,
            dist_metric='euclidean', normalize_feature=False, visrank=False, visrankactiv=False, visrankactivthr=False,
            maskthr=0.7, visrank_topk=10, use_metric_cuhk03=False, ranks=[1, 5, 10, 20], rerank=False, visactmap=False,
            vispartmap=False, visdrop=False, visdroptype='random'):
        """A unified pipeline for training and evaluating a model.
        :param aim_sess: aim recorder
        :param save_dir: directory to save model.
        :param max_epoch: maximum epoch.
        :param start_epoch: (int, optional) starting epoch. Default is 0.
        :param fixbase_epoch: (int, optional) number of epochs to train ``open_layers`` (new layers)
                              while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                              in ``max_epoch``.
        :param open_layers: (str or list, optional) layers (attribute names) open for training.
                            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
                            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                            is only performed at the end of training).
        :param start_eval:
        :param eval_freq:
        :param test_only: (bool, optional) if True, only runs evaluation on test datasets. Default is False.
        :param print_freq: (int, optional) print_frequency. Default is 10.
        :param dist_metric: (str, optional) distance metric used to compute distance matrix between query and gallery.
                            Default is "euclidean".
        :param normalize_feature: (bool, optional) performs L2 normalization on feature vectors before
                                  computing feature distance. Default is False.
        :param visrank: (bool, optional) visualizes ranked results. Default is False. It is recommended to
                        enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                        "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
        :param visrankactiv:
        :param visrankactivthr:
        :param maskthr:
        :param visrank_topk: (int, optional) top-k ranked images to be visualized. Default is 10.
                             use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                             Default is False. This should be enabled when using cuhk03 classic split.
        :param use_metric_cuhk03:
        :param ranks: (list, optional) cmc ranks to be computed. Default is [1, 5, 10, 20].
        :param rerank: (bool, optional) uses person re-ranking (by Zhong et al. CVPR'17).
                       Default is False. This is only enabled when test_only=True.
        :param visactmap: (bool, optional) visualizes activation maps. Default is False.
        :param vispartmap:
        :param visdrop:
        :param visdroptype:
        :return:
        """
        if visrank and not test_only:
            raise ValueError('visrank=True is valid only if test_only=True')

        if visrankactiv and not test_only:
            raise ValueError('visrankactiv=True is valid only if test_only=True')

        if visrankactivthr and not test_only:
            raise ValueError('visrankactivthr=True is valid only if test_only=True')

        if visdrop and not test_only:
            raise ValueError('visdrop=True is valid only if test_only=True')

        if test_only:
            self.test(
                0,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrankactiv=visrankactiv,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                maskthr=maskthr,
                visrankactivthr=visrankactivthr,
                visdrop=visdrop,
                visdroptype=visdroptype
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        if visactmap:
            self.visactmap(self.test_loader, save_dir, self.datamanager.width, self.datamanager.height, print_freq)
            return

        if vispartmap:
            self.vispartmap(self.test_loader, save_dir, self.datamanager.width, self.datamanager.height, print_freq)
            return

        time_start = time.time()
        self.max_epoch = max_epoch
        print('=> Start training')

        for epoch in range(start_epoch, max_epoch):
            losses = self.train(print_freq=print_freq, fixbase_epoch=fixbase_epoch,
                                open_layers=open_layers, epoch=epoch)

            # AIM recorder (acc, loss)
            for key in losses.meters.keys():
                aim_sess.track(losses.meters[key].avg, name=key, epoch=epoch, subset='train')
            
            if (epoch+1) >= start_eval and eval_freq > 0 and (epoch+1) % eval_freq == 0 and (epoch+1) != max_epoch:
                rank1 = self.test(
                    epoch,
                    aim_sess=aim_sess,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrankactiv=visrankactiv,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    rerank=rerank,
                    maskthr=maskthr,
                    visrankactivthr=visrankactivthr
                )
                self._save_checkpoint(epoch, rank1, save_dir)

        if max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                epoch,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrankactiv=visrankactiv,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                maskthr=maskthr,
                visrankactivthr=visrankactivthr
            )
            self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is None:
            self.writer.close()

    def train(self, epoch=0, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.model.train()
        self.two_stepped_transfer_learning(epoch, fixbase_epoch, open_layers)

        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data, epoch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (batch_idx + 1) % print_freq == 0:
                # Estimate remaining time
                nb_this_epoch = num_batches - (batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (epoch + 1)) * num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(
                    '[Epoch][Batch]: [{0}/{1}][{2}/{3}]\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        epoch + 1,
                        self.max_epoch,
                        batch_idx + 1,
                        num_batches,
                        eta=eta_str,
                        losses=losses,
                        lr=self.optimizer.param_groups[0]['lr'],
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[-1]['lr'], n_iter)

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        return losses

    def forward_backward(self, data, epoch=None):
        raise NotImplementedError

    def test(self, epoch, aim_sess=None, dist_metric='euclidean', normalize_feature=False,
             visrank=False, visrankactiv=False, visrank_topk=10, save_dir='', use_metric_cuhk03=False,
             ranks=[1, 5, 10, 20], rerank=False, maskthr=0.7, visrankactivthr=False, visdrop=False, visdroptype='random'):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        targets = list(self.test_loader.keys())
        
        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            queryloader = self.test_loader[name]['query']
            galleryloader = self.test_loader[name]['gallery']
            rank1 = self._evaluate(
                epoch,
                aim_sess=aim_sess,
                dataset_name=name,
                queryloader=queryloader,
                galleryloader=galleryloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrankactiv=visrankactiv,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                maskthr=maskthr,
                visrankactivthr=visrankactivthr,
                visdrop=visdrop,
                visdroptype=visdroptype
            )
        
        return rank1

    @torch.no_grad()
    def _evaluate(self, epoch, aim_sess=None, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False, visrankactiv=False,
                  visrank_topk=10, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                  rerank=False, visrankactivthr=False, maskthr=0.7, visdrop=False, visdroptype='random'):
        batch_time = AverageMeter()

        print('Extracting features from query set ...')
        # Terms: query features, query activations, query person IDs, query camera IDs and image drop masks
        qf, qa, q_pids, q_camids, qm = [], [], [], [], []
        for _, data in enumerate(queryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            activations = self._extract_activations(imgs)
            dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            qa.append(torch.Tensor(activations))
            qm.append(torch.Tensor(dropmask))
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        qm = torch.cat(qm, 0)
        qa = torch.cat(qa, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        # Gallery features, Gallery activations, Gallery person IDs, Gallery camera IDs and Image drop masks
        gf, ga, g_pids, g_camids, gm = [], [], [], [], []
        for _, data in enumerate(galleryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            activations = self._extract_activations(imgs)
            dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            ga.append(torch.Tensor(activations))
            gm.append(torch.Tensor(dropmask))
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        gm = torch.cat(gm, 0)
        ga = torch.cat(ga, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalizing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        # Always show results without re-ranking first
        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))

        # AIM recorder (mAP, Rank-1, Rank-5, Rank-10, Rank-20)
        if aim_sess:
            aim_sess.track(mAP, name='mAP', epoch=epoch, subset='train')
            for r in ranks:
                aim_sess.track(cmc[r-1], name='Rank-{:<3}'.format(r), epoch=epoch, subset='train')

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)
            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                use_metric_cuhk03=use_metric_cuhk03
            )

            print('** Results with Re-Ranking**')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_'+dataset_name),
                topk=visrank_topk
            )
        if visrankactiv:
            visualize_ranked_activation_results(
                distmat,
                qa,
                ga,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrankactiv_'+dataset_name),
                topk=visrank_topk
            )
        if visrankactivthr:
            visualize_ranked_threshold_activation_results(
                distmat,
                qa,
                ga,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrankactivthr_'+dataset_name),
                topk=visrank_topk,
                threshold=maskthr
            )
        if visdrop:
            visualize_ranked_mask_activation_results(
                distmat,
                qa,
                ga,
                qm,
                gm,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visdrop_{}_{}'.format(visdroptype, dataset_name)),
                topk=visrank_topk
            )

        return cmc[0]

    @torch.no_grad()
    def visactmap(self, testloader, save_dir, width, height, print_freq):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        for target in list(testloader.keys()):
            queryloader = testloader[target]['query']
            # original images and activation maps are saved individually
            actmap_dir = osp.join(save_dir, 'actmap_'+target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(queryloader):
                imgs, paths = data[0], data[3]
                if self.use_gpu:
                    imgs = imgs.cuda()
                
                # forward to get convolutional feature maps
                try:
                    outputs = self.model(imgs, return_featuremaps=True)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_featuremaps". '
                                    'Please add return_featuremaps as an input argument to forward(). When '
                                    'return_featuremaps=True, return feature maps only.')
                
                if outputs.dim() != 4:
                    raise ValueError('The model output is supposed to have '
                                     'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs.dim()))
                
                # compute activation maps
                outputs = (outputs**2).sum(1)
                b, h, w = outputs.size()
                outputs = outputs.view(b, h*w)
                outputs = F.normalize(outputs, p=2, dim=1)
                outputs = outputs.view(b, h, w)
                
                if self.use_gpu:
                    imgs, outputs = imgs.cpu(), outputs.cpu()

                for j in range(outputs.size(0)):
                    # get image name
                    path = paths[j]
                    imname = osp.basename(osp.splitext(path)[0])
                    
                    # RGB image
                    img = imgs[j, ...]
                    for t, m, s in zip(img, imagenet_mean, imagenet_std):
                        t.mul_(s).add_(m).clamp_(0, 1)
                    img_np = np.uint8(np.floor(img.numpy() * 255))
                    img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)
                    
                    # activation map
                    am = outputs[j, ...].numpy()
                    am = cv2.resize(am, (width, height))
                    am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                    am = np.uint8(np.floor(am))
                    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
                    
                    # overlapped
                    overlapped = img_np * 0.4 + am * 0.6
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)

                    # save images in a single figure (add white spacing between images)
                    # from left to right: original image, activation map, overlapped image
                    grid_img = 255 * np.ones((height, 3*width+2*GRID_SPACING, 3), dtype=np.uint8)
                    grid_img[:, :width, :] = img_np[:, :, ::-1]
                    grid_img[:, width+GRID_SPACING: 2*width+GRID_SPACING, :] = am
                    grid_img[:, 2*width+2*GRID_SPACING:, :] = overlapped
                    cv2.imwrite(osp.join(actmap_dir, imname+'.jpg'), grid_img)

                if (batch_idx+1) % print_freq == 0:
                    print('- done batch {}/{}'.format(batch_idx+1, len(queryloader)))

    @torch.no_grad()
    def vispartmap(self, testloader, save_dir, width, height, print_freq):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        for target in list(testloader.keys()):
            queryloader = testloader[target]['query']
            # original images and activation maps are saved individually
            actmap_dir = osp.join(save_dir, 'partmap_'+target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing parts activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(queryloader):
                imgs, paths = data[0], data[3]
                if self.use_gpu:
                    imgs = imgs.cuda()

                # forward to get convolutional feature maps
                try:
                    outputs_list = self.model(imgs, return_partmaps=True)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_partmaps". '
                                    'Please add return_partmaps as an input argument to forward(). When '
                                    'return_partmaps=True, return feature maps only.')
                if outputs_list[0][0].dim() != 4:
                    raise ValueError('The model output is supposed to have ' \
                                     'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs_list[0][0].dim()))

                # Print stats between parts and weights
                print("First image stats")
                w = []
                for i, (_, _, _, weights) in enumerate(outputs_list):
                    print("\tpart{} min {} max {}".format(i, torch.min(weights[0, ...]), torch.max(weights[0, ...])))
                    w.append(weights)
                print("Second image stats")
                for i, (_, _, _, weights) in enumerate(outputs_list):
                    print("\tpart{} min {} max {}".format(i, torch.min(weights[1, ...]), torch.max(weights[1, ...])))
                print("Difference")
                for i, (_, _, _, weights) in enumerate(outputs_list):
                    print("\tpart{} min {} max {} mean {}".format(i, torch.min(weights[0, ...] - weights[1, ...]), torch.max(weights[0, ...] - weights[1, ...]), torch.mean(weights[0, ...] - weights[1, ...])))
                print("\tbetween min {} max {} mean {}".format(torch.min(w[0] - w[1]), torch.max(w[0] - w[1]), torch.mean(w[0] - w[1])))

                for part_ind, (outputs, weights, _, _) in enumerate(outputs_list):
                    # compute activation maps
                    b, c, h, w = outputs.size()
                    outputs = (outputs**2).sum(1)
                    outputs = outputs.view(b, h*w)
                    outputs = F.normalize(outputs, p=2, dim=1)
                    outputs = outputs.view(b, h, w)

                    weights = weights.view(b, c)
                    weights = F.normalize(weights, p=2, dim=1)
                    weights = weights.view(b, 1, c)

                    if self.use_gpu:
                        imgs, outputs, weights = imgs.cpu(), outputs.cpu(), weights.cpu()

                    for j in range(outputs.size(0)):
                        # get image name
                        path = paths[j]
                        imname = osp.basename(osp.splitext(path)[0])

                        # RGB image
                        img = imgs[j, ...].clone()
                        for t, m, s in zip(img, imagenet_mean, imagenet_std):
                            t.mul_(s).add_(m).clamp_(0, 1)
                        img_np = np.uint8(np.floor(img.numpy() * 255))
                        img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)
                        
                        # activation map
                        am = outputs[j, ...].numpy()
                        am = cv2.resize(am, (width, height))
                        am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                        am = np.uint8(np.floor(am))
                        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                        # parts activation map
                        pam = weights[j, ...].numpy()
                        pam = np.resize(pam, (VECT_HEIGHT, c))  # expand to create larger vectors for better visualization
                        pam = cv2.resize(pam, (3*width+2*GRID_SPACING, VECT_HEIGHT))
                        pam = 255 * (pam - np.max(pam)) / (np.max(pam) - np.min(pam) + 1e-12)
                        pam = np.uint8(np.floor(pam))
                        pam = cv2.applyColorMap(pam, cv2.COLORMAP_JET)

                        # overlapped
                        overlapped = img_np * 0.4 + am * 0.6
                        overlapped[overlapped>255] = 255
                        overlapped = overlapped.astype(np.uint8)

                        # save images in a single figure (add white spacing between images)
                        # from left to right: original image, activation map, overlapped image
                        grid_img = 255 * np.ones((height + GRID_SPACING + VECT_HEIGHT, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8)
                        grid_img[:height, :width, :] = img_np[:, :, ::-1]
                        grid_img[:height, width+GRID_SPACING: 2*width+GRID_SPACING, :] = am
                        grid_img[:height, 2*width+2*GRID_SPACING:, :] = overlapped
                        grid_img[height + GRID_SPACING:, :, :] = pam

                        cv2.imwrite(osp.join(actmap_dir, imname+'_{}.jpg'.format(part_ind)), grid_img)

                    if (batch_idx+1) % print_freq == 0:
                        print('- done batch {}/{} part {}/{}'.format(batch_idx+1, len(queryloader), part_ind + 1, len(outputs_list)))

    @staticmethod
    def _compute_loss(criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _extract_activations(self, input):
        self.model.eval()
        outputs = self.model(input, return_featuremaps=True)
        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h*w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)
        activations = []
        for j in range(outputs.size(0)):
            # activation map
            am = outputs[j, ...].cpu().numpy()
            am = cv2.resize(am, (self.datamanager.width, self.datamanager.height))
            am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
            activations.append(am)
        return np.array(activations)

    def _extract_drop_masks(self, input, visdrop, visdroptype):
        self.model.eval()
        drop_top = (visdroptype == 'top')
        outputs = self.model(input, drop_top=drop_top, visdrop=visdrop)
        outputs = outputs.mean(1)
        masks = []
        for j in range(outputs.size(0)):
            # Drop masks
            dm = outputs[j, ...].cpu().numpy()
            dm = cv2.resize(dm, (self.datamanager.width, self.datamanager.height))
            masks.append(dm)
        return np.array(masks)

    @staticmethod
    def _parse_data_for_train(data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    @staticmethod
    def _parse_data_for_eval(data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)

    def two_stepped_transfer_learning(self, epoch, fixbase_epoch, open_layers, model=None):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
            open_specified_layers(model, open_layers)
        else:
            open_all_layers(model)

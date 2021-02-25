from __future__ import absolute_import, division

__all__ = ['cmnet_norm_botdropfeat_doubot', 'cmnet_norm_neck_botdropfeat_doubot',
           'cmnet_norm_neck', 'cmnet_norm', 'cmnet_norm_neck_doubot', 'cmnet_norm_doubot',
           'cmnodropnet_norm', 'cmnodropnet_norm_neck']

import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50_ls, resnext50_32x4d, resnext101_32x8d
from .osnet_ain import osnet_ain_x1_0
from .gen_mean_pool import GeM

from torchvision.models.resnet import Bottleneck
import random


# Batch dropblock
class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            if visdrop:
                return mask
            x = x * mask
        return x


class BatchDropTop(nn.Module):
    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
        self.h_ratio = h_ratio
    
    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h*w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda: mask = mask.cuda()
            if visdrop:
                return mask
            x = x * mask
        return x


class BatchFeatureErase_Basic(nn.Module):
    def __init__(self, channels, h_ratio=0.33, w_ratio=1.):
        super(BatchFeatureErase_Basic, self).__init__()
        self.drop_batch_bottleneck = Bottleneck(channels, 512)
        self.drop_batch_drop = BatchDrop(h_ratio, w_ratio)

    def forward(self, x):
        x = self.drop_batch_bottleneck(x)
        x = self.drop_batch_drop(x)
        return x


class BatchFeatureErase_Top(nn.Module):
    def __init__(self, channels, h_ratio=0.33, w_ratio=1., double_bottleneck = False):
        super(BatchFeatureErase_Top, self).__init__()
        if double_bottleneck:
            self.drop_batch_bottleneck = nn.Sequential(
                Bottleneck(channels, 512),
                Bottleneck(channels, 512)
            )
        else:
            self.drop_batch_bottleneck = Bottleneck(channels, 512)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=False, bottleneck_features = False, visdrop=False):
        features = self.drop_batch_bottleneck(x)
        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x


class TopBDNet(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', neck=False,
                 drop_height_ratio=0.33, drop_width_ratio=1.0,
                 double_bottleneck=False, drop_bottleneck_features=False, backbone='resnet',
                 **kwargs):
        super(TopBDNet, self).__init__()
        self.loss = loss
        self.drop_bottleneck_features = drop_bottleneck_features
        if neck:
            self.bottleneck_global = nn.BatchNorm1d(512)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_db = nn.BatchNorm1d(1024)
            self.bottleneck_db.bias.requires_grad_(False)  # no shift
            self.bottleneck_drop_bottleneck_features = nn.BatchNorm1d(2048)
            self.bottleneck_drop_bottleneck_features.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck_global = None
            self.bottleneck_db = None
            self.bottleneck_drop_bottleneck_features = None

        self.reduction_global = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.reduction_db = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool_global = GeM()
        self.avgpool_drop = GeM()
        self.classifier_global = nn.Linear(512, num_classes)
        self.classifier_db = nn.Linear(1024, num_classes)
        self.batch_drop = BatchFeatureErase_Top(2048, drop_height_ratio, drop_width_ratio, double_bottleneck)
        if self.drop_bottleneck_features:
            self.classifier_drop_bottleneck = nn.Linear(2048, num_classes)
        else:
            self.classifier_drop_bottleneck = None
        self._init_params()

        if backbone == 'osnet_ain':
            resnet = osnet_ain_x1_0(num_classes, pretrained=True)
            self.base = nn.Sequential(*list(resnet.children())[:-4] + [nn.Conv2d(512, 2048, 2, stride=2)])
        else:
            if backbone == 'resnext50':
                resnet = resnext50_32x4d(num_classes, pretrained=True)
            elif backbone == 'resnext101':
                resnet = resnext101_32x8d(num_classes, pretrained=True)
            else:
                resnet = resnet50_ls(num_classes, pretrained=True)
            self.base = nn.Sequential(*list(resnet.children())[:-2])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False, drop_top=False, visdrop=False):
        x = self.base(x)
        if return_featuremaps:
            return x
        if visdrop:  # Return dropmask
            drop_mask = self.batch_drop(x, drop_top=drop_top, visdrop=visdrop)
            return drop_mask

        if self.drop_bottleneck_features:
            drop_x, t_drop_bottleneck_features = self.batch_drop(x, drop_top=drop_top, bottleneck_features=True)
            t_drop_bottleneck_features = self.avgpool_drop(t_drop_bottleneck_features).view(t_drop_bottleneck_features.size()[:2])
            t_drop_bottleneck_features = F.normalize(t_drop_bottleneck_features)
            if self.bottleneck_drop_bottleneck_features:
                x_drop_bottleneck_features = self.bottleneck_drop_bottleneck_features(t_drop_bottleneck_features)
            else:
                x_drop_bottleneck_features = t_drop_bottleneck_features
            x_drop_bottleneck_features = self.classifier_drop_bottleneck(x_drop_bottleneck_features)
        else:
            drop_x = self.batch_drop(x, drop_top=drop_top)
            x_drop_bottleneck_features, t_drop_bottleneck_features = None, None

        # Global
        x = self.avgpool_global(x)
        t_x = self.reduction_global(x)
        t_x = t_x.view(t_x.size()[:2])
        x_x = self.bottleneck_global(t_x) if self.bottleneck_global else t_x
        x_x = F.normalize(x_x)
        x_prelogits = self.classifier_global(x_x)

        # Dropbatch
        drop_x = self.maxpool(drop_x).view(drop_x.size()[:2])
        t_drop_x = self.reduction_db(drop_x)
        x_drop_x = self.bottleneck_db(t_drop_x) if self.bottleneck_db else t_drop_x
        x_drop_x = F.normalize(x_drop_x)
        x_drop_prelogits = self.classifier_db(x_drop_x)

        if not self.training:
            return torch.cat((x_x, x_drop_x), dim=1)

        if self.loss == 'triplet_dropbatch':
            return x_prelogits, t_x, x_drop_prelogits, t_drop_x
        if 'dropbot' in self.loss:
            return x_prelogits, t_x, x_drop_prelogits, t_drop_x, x_drop_bottleneck_features, t_drop_bottleneck_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


# Top bdnet
def cmnet_norm_botdropfeat_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = TopBDNet(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        **kwargs
    )
    return model


def cmnet_norm_neck_botdropfeat_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = TopBDNet(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        **kwargs
    )
    return model


# Top bdnet without third stream
def cmnet_norm_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = TopBDNet(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=False,
        double_bottleneck=True,
        **kwargs
    )
    return model


def cmnet_norm_neck_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = TopBDNet(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=False,
        double_bottleneck=True,
        **kwargs
    )
    return model


# Batch dropblock
def cmnet_norm(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = TopBDNet(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=False,
        double_bottleneck=False,
        **kwargs
    )
    return model


# Batch dropblock
def cmnet_norm_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = TopBDNet(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=False,
        double_bottleneck=False,
        **kwargs
    )
    return model


class NoDropNet(nn.Module):
    """
    only global and regularization streams
    """
    def __init__(self, num_classes=0, loss='softmax', neck=False, drop_height_ratio=0.33, drop_width_ratio=1.0,
                 backbone='resnet50', **kwargs):
        super(NoDropNet, self).__init__()
        self.loss = loss
        if neck:
            self.bottleneck_global = nn.BatchNorm1d(512)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_regularization = nn.BatchNorm1d(2048)
            self.bottleneck_regularization.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck_global = None
            self.bottleneck_regularization = None

        self.reduction_global = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.reduction_regularization = nn.Sequential(
                Bottleneck(2048, 512),
                Bottleneck(2048, 512)
            )
        self.avgpool_global = GeM()
        self.avgpool_regularization = GeM()
        self.classifier_global = nn.Linear(512, num_classes)
        self.classifier_regularization = nn.Linear(2048, num_classes)
        self._init_params()

        if backbone == 'resnext50':
            resnet = resnext50_32x4d(num_classes, pretrained=True)
        else:
            resnet = resnet50_ls(num_classes, pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False, drop_top=False):
        x = self.base(x)
        if return_featuremaps:
            return x

        # Regularization
        reg_x = self.reduction_regularization(x)
        t_reg = self.avgpool_regularization(reg_x).view(reg_x.size(0), -1)
        if self.bottleneck_regularization:
            x_reg = self.bottleneck_regularization(t_reg)
        else:
            x_reg = t_reg
        x_reg_prelogits = self.classifier_regularization(x_reg)

        # Global
        x = self.avgpool_global(x)
        t_x = self.reduction_global(x)
        t_x = t_x.view(t_x.size(0), -1)
        if self.bottleneck_global:
            x_x = self.bottleneck_global(t_x)
        else:
            x_x = t_x
        x_prelogits = self.classifier_global(x_x)

        if not self.training:
            return torch.cat((x_x, x_reg), dim=1)

        if self.loss == 'triplet_dropbatch':
            return x_prelogits, t_x, x_reg_prelogits, t_reg
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


# Top bdnet without second stream
def cmnodropnet_norm(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = NoDropNet(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model


def cmnodropnet_norm_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = NoDropNet(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

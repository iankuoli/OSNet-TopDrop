from __future__ import absolute_import

from .resnet import *
from .resnetmid import *
from .senet import *
from .densenet import *
from .inceptionresnetv2 import *
from .inceptionv4 import *
from .xception import *

from .nasnet import *
from .mobilenetv2 import *
from .shufflenet import *
from .squeezenet import *
from .shufflenetv2 import *

from .mudeep import *
from .hacnn import *
from .pcb import *
from .mlfn import *
from .osnet import *
from .osnet_ain import *
from .osnet_ain_lambda import *
from .plr_osnet import *
from .plr_osnet_ain import *
from .bdnet import *
from .cloudmile_net import *
from .cloudmile_net_norm import *


__model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet50_ls': resnet50_ls,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': inceptionresnetv2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2_x1_0': mobilenetv2_x1_0,
    'mobilenetv2_x1_4': mobilenetv2_x1_4,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'bdnet_neck': bdnet_neck,
    'bdnet': bdnet,
    'top_bdnet_neck_doubot': top_bdnet_neck_doubot,
    'top_bdnet_doubot': top_bdnet_doubot,
    'top_bdnet_botdropfeat_doubot': top_bdnet_botdropfeat_doubot,
    'top_bdnet_neck_botdropfeat_doubot': top_bdnet_neck_botdropfeat_doubot,
    'nodropnet_neck': nodropnet_neck,
    'nodropnet': nodropnet,
    'cmnet_neck_doubot': cmnet_neck_doubot,
    'cmnet_doubot': cmnet_doubot,
    'cmnet_botdropfeat_doubot': cmnet_botdropfeat_doubot,
    'cmnet_neck_botdropfeat_doubot': cmnet_neck_botdropfeat_doubot,
    'cmnodropnet_neck': cmnodropnet_neck,
    'cmnodropnet': cmnodropnet,
    'cmnet_norm_neck_doubot': cmnet_norm_neck_doubot,
    'cmnet_norm_doubot': cmnet_norm_doubot,
    'cmnet_norm_botdropfeat_doubot': cmnet_norm_botdropfeat_doubot,
    'cmnet_norm_neck_botdropfeat_doubot': cmnet_norm_neck_botdropfeat_doubot,
    'cmnodropnet_norm_neck': cmnodropnet_norm_neck,
    'cmnodropnet_norm': cmnodropnet_norm
}


def show_avai_models():
    """Displays available models.

    Examples::
        >> from deepreid import models
        >> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True, backbone='resnet50'):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >> from deepreid import models
        >> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        backbone=backbone,
    )
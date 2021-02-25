import sys
import os
import os.path as osp
import time
import argparse

import torch
import torch.nn as nn
from aim import Session

import torchreid
from torchreid.engine.image.arc_margin import ArcMarginProduct
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)
from script.default_config import (
    get_default_config, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler, arc_margins=None):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet_dropbatch':
            engine = torchreid.engine.ImageTripletDropBatchEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet_dropbatch_dropbotfeatures':
            engine = torchreid.engine.ImageTripletDropBatchDropBotFeaturesEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'supcon_triplet_dropbatch_dropbotfeatures':
            metric_loss = {'supcon': {'temperature': cfg.loss.supcon.temperature,
                                      'base_temperature': cfg.loss.supcon.base_temperature,
                                      'weight': cfg.loss.supcon.weight_s}}
            engine = torchreid.engine.ImageSupConTripletDropBatchDropBotFeaturesEngine(
                datamanager,
                model,
                optimizer,
                metric_loss,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name in {'arc_focal_triplet_dropbatch_dropbotfeatures',
                               'supcon_arc_focal_triplet_dropbatch_dropbotfeatures',
                               'arc_ce_triplet_dropbatch_dropbotfeatures',
                               'supcon_arc_ce_triplet_dropbatch_dropbotfeatures',
                               'arc_cbfocal_triplet_dropbatch_dropbotfeatures',
                               'supcon_arc_cbfocal_triplet_dropbatch_dropbotfeatures',
                               'arc_ldam_triplet_dropbatch_dropbotfeatures',
                               'supcon_arc_ldam_triplet_dropbatch_dropbotfeatures'}:
            metric_loss = {}
            if 'arc_focal' in cfg.loss.name:
                metric_loss = {'loss': 'focal',
                               'gamma': cfg.loss.focal.gamma}
            elif 'arc_ce' in cfg.loss.name:
                metric_loss = {'loss': 'ce'}
            elif 'arc_cbfocal' in cfg.loss.name:
                metric_loss = {'loss': 'cbfocal',
                               'gamma': cfg.loss.cbfocal.gamma,
                               'beta': cfg.loss.cbfocal.beta}
            elif 'arc_ldam' in cfg.loss.name:
                metric_loss = {'loss': 'ldam',
                               's': cfg.loss.ldam.s,
                               'max_m': cfg.loss.ldam.max_m}
            else:
                exit("Unidentified loss name!!!")

            args = {'margin': cfg.loss.triplet.margin,
                    'weight_f': cfg.loss.focal.weight_f,
                    'weight_t': cfg.loss.triplet.weight_t,
                    'weight_x': cfg.loss.triplet.weight_x,
                    'weight_db_t': cfg.loss.dropbatch.weight_db_t,
                    'weight_db_x': cfg.loss.dropbatch.weight_db_x,
                    'weight_b_db_t': cfg.loss.dropbatch.weight_b_db_t,
                    'weight_b_db_x': cfg.loss.dropbatch.weight_b_db_x,
                    'top_drop_epoch': cfg.loss.dropbatch.top_drop_epoch,
                    'scheduler': scheduler,
                    'use_gpu': cfg.use_gpu,
                    'label_smooth': cfg.loss.softmax.label_smooth}

            if 'supcon' in cfg.loss.name:
                metric_loss['supcon'] = {'temperature': cfg.loss.supcon.temperature,
                                         'base_temperature': cfg.loss.supcon.base_temperature,
                                         'weight': cfg.loss.supcon.weight_s}
                engine = torchreid.engine.ImageSupConArcTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    metric_loss,
                    arc_margin=arc_margins['arc_margin'],
                    **args
                )
            else:
                engine = torchreid.engine.ImageArcTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    metric_loss,
                    arc_margin=arc_margins['arc_margin'],
                    **args
                )
        elif cfg.loss.name in {'arc_focal_focal_dropbatch_dropbotfeatures',
                               'supcon_arc_focal_focal_dropbatch_dropbotfeatures',
                               'arc_ce_focal_dropbatch_dropbotfeatures',
                               'supcon_arc_ce_focal_dropbatch_dropbotfeatures',
                               'arc_cbfocal_focal_dropbatch_dropbotfeatures',
                               'supcon_arc_cbfocal_focal_dropbatch_dropbotfeatures',
                               'arc_ldam_focal_dropbatch_dropbotfeatures',
                               'supcon_arc_ldam_focal_dropbatch_dropbotfeatures'}:
            metric_loss = {}
            if 'arc_focal' in cfg.loss.name:
                metric_loss = {'loss': 'focal',
                               'gamma': cfg.loss.focal.gamma}
            elif 'arc_ce' in cfg.loss.name:
                metric_loss = {'loss': 'ce'}
            elif 'arc_cbfocal' in cfg.loss.name:
                metric_loss = {'loss': 'cbfocal',
                               'gamma': cfg.loss.cbfocal.gamma,
                               'beta': cfg.loss.cbfocal.beta}
            elif 'arc_ldam' in cfg.loss.name:
                metric_loss = {'loss': 'ldam',
                               's': cfg.loss.ldam.s,
                               'max_m': cfg.loss.ldam.max_m}
            else:
                exit("Unidentified loss name!!!")

            args = {'arc_margin': arc_margins['arc_margin'],
                    'arc_margin_f': arc_margins['arc_margin_f'],
                    'arc_margin_db_f': arc_margins['arc_margin_db_f'],
                    'arc_margin_b_db_f': arc_margins['arc_margin_b_db_f'],
                    'gamma': cfg.loss.focal.gamma,
                    'weight_all_f': cfg.loss.focal.weight_all_f,
                    'weight_f': cfg.loss.focal.weight_f,
                    'weight_x': cfg.loss.triplet.weight_x,
                    'weight_db_f': cfg.loss.dropbatch.weight_db_t,
                    'weight_db_x': cfg.loss.dropbatch.weight_db_x,
                    'weight_b_db_f': cfg.loss.dropbatch.weight_b_db_t,
                    'weight_b_db_x': cfg.loss.dropbatch.weight_b_db_x,
                    'top_drop_epoch': cfg.loss.dropbatch.top_drop_epoch,
                    'scheduler': scheduler,
                    'use_gpu': cfg.use_gpu,
                    'label_smooth': cfg.loss.softmax.label_smooth}

            if 'supcon' in cfg.loss.name:
                metric_loss['supcon'] = {'temperature': cfg.loss.supcon.temperature,
                                         'base_temperature': cfg.loss.supcon.base_temperature,
                                         'weight': cfg.loss.supcon.weight_s}
                engine = torchreid.engine.ImageSupConArcFocalDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    metric_loss,
                    **args
                )
            else:
                engine = torchreid.engine.ImageArcFocalDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    metric_loss,
                    **args
                )
        elif cfg.loss.name == 'triplet':
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        else:
            exit("Can not identified the loss name")
    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('-s', '--sources', type=str, nargs='+', help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+', help='target datasets (delimited by space)')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation')
    parser.add_argument('--root', type=str, default='', help='path to data root')
    parser.add_argument('--gpu-devices', type=str, default='',)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    if cfg.use_gpu and args.gpu_devices:
        # if gpu_devices is not specified, all available gpus will be used
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    datamanager = build_datamanager(cfg)
    
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        backbone=cfg.model.backbone
    )
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    if 'arc' in cfg.loss.name:
        feat_num = 512
        if 'neck_botdropfeat_doubot' in cfg.model.name:
            feat_num += 1024
        arc_margins = {'arc_margin': ArcMarginProduct(in_features=feat_num, out_features=datamanager.num_train_pids,
                                                      s=cfg.loss.arc.s, m=cfg.loss.arc.m,
                                                      easy_margin=True, use_gpu=cfg.use_gpu)
                       }
        if 'focal_dropbatch' in cfg.loss.name:
            arc_margins['arc_margin_f'] = ArcMarginProduct(in_features=512,
                                                           out_features=datamanager.num_train_pids,
                                                           s=cfg.loss.arc.s, m=cfg.loss.arc.m,
                                                           easy_margin=True, use_gpu=cfg.use_gpu)
            arc_margins['arc_margin_db_f'] = ArcMarginProduct(in_features=1024,
                                                              out_features=datamanager.num_train_pids,
                                                              s=cfg.loss.arc.s, m=cfg.loss.arc.m,
                                                              easy_margin=True, use_gpu=cfg.use_gpu)
            arc_margins['arc_margin_b_db_f'] = ArcMarginProduct(in_features=2048,
                                                                out_features=datamanager.num_train_pids,
                                                                s=cfg.loss.arc.s, m=cfg.loss.arc.m,
                                                                easy_margin=True, use_gpu=cfg.use_gpu)
    else:
        arc_margins = None

    optimizer = torchreid.optim.build_optimizer(model, arc_margins=arc_margins, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        args.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    aim_sess = Session(experiment=cfg.model.name)
    aim_sess.set_params({
        'model_name': cfg.model.name,
        'num_epochs': cfg.train.max_epoch,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'learning_rate': cfg.train.lr,
        'loss': cfg.loss.name,
        'arc_s': cfg.loss.arc.s,
        'arc_m': cfg.loss.arc.m,
        'weight_f': cfg.loss.focal.weight_f,
        'weight_x': cfg.loss.focal.weight_x,
        'optim': cfg.train.optim,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'dist_metric': cfg.test.dist_metric,
        'attention_dim_k': cfg.lambdalayer.dim_k if 'Lambda' in cfg.model.name else -1,
        'attention_heads': cfg.lambdalayer.heads if 'Lambda' in cfg.model.name else -1,
        'attention_dim_u': cfg.lambdalayer.dim_u if 'Lambda' in cfg.model.name else -1
    }, name='hparams')

    engine = build_engine(cfg, datamanager, model, optimizer, scheduler, arc_margins=arc_margins)
    engine.run(aim_sess, **engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()

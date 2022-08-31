"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
Note: Our implementation is different from ADDA paper in several respects. We do not use separate networks for
source and target domain, nor fix classifier head. Besides, we do not adopt asymmetric objective loss function
of the feature extractor.
"""
import random
import time
import warnings
import copy
import argparse
import shutil
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
from tllib.alignment.adda import ImageClassifier
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_requires_grad(net, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all = sum(p.numel() for p in model.parameters())
    print('All parameters: {}'.format(all))
    print('Trainable parameters: {}'.format(trainable))

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, None, None)
    
    train_source_loader, train_target_loader = None, None
    if args.weighted_sample:
        source_weight = utils.make_weight_for_balanced_classes(train_source_dataset.imgs, len(train_source_dataset.classes))
        source_weight = torch.DoubleTensor(source_weight)
        source_sampler = torch.utils.data.sampler.WeightedRandomSampler(source_weight, len(source_weight))

        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.workers, drop_last=True, sampler=source_sampler)
    else:
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers, drop_last=True)


    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    train_source_iter = ForeverDataIterator(train_source_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None


    if args.phase == 'train':
        # first pretrain the classifier wish source data
        print("Pretraining the model on source domain.")
        args.pretrain = logger.get_checkpoint_path('pretrain')
        pretrain_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                         pool_layer=pool_layer, finetune=not args.scratch).to(device)
        pretrain_model.backbone.requires_grad_(False)
        count_parameters(pretrain_model)

        if args.distributed_training:
            pretrain_model = nn.DataParallel(pretrain_model)

        pretrain_optimizer = Adam(pretrain_model.get_parameters(), args.pretrain_lr,
                                 weight_decay=1e-4)
        pretrain_lr_scheduler = LambdaLR(pretrain_optimizer,
                                         lambda x: args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (
                                             -args.lr_decay))

        # start pretraining
        best = 0
        for epoch in range(args.pretrain_epochs):
            print("lr:", pretrain_lr_scheduler.get_lr())
            # pretrain for one epoch
            utils.empirical_risk_minimization(train_source_iter, pretrain_model, pretrain_optimizer,
                                              pretrain_lr_scheduler, epoch, args,
                                              device)
            # validate to show pretrain process
            acc = utils.validate(val_loader, pretrain_model, args, device)
            if acc > best:
                best = acc
                torch.save(pretrain_model.state_dict(), os.path.join(os.path.split(args.pretrain)[0], '{}.pth'.format(str(epoch).zfill(3))))

        torch.save(pretrain_model.state_dict(), os.path.join(os.path.split(args.pretrain)[0], 'last.pth'))
        print("Pretraining process is done.")
    
    if args.phase == 'test':
        utils.validate(val_loader, pretrain_model, args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADDA for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretrain checkpoint for classification model')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate of the classifier', dest='lr')
    parser.add_argument('--pretrain-lr', default=0.001, type=float, help='initial pretrain learning rate')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrain-epochs', default=3, type=int, metavar='N',
                        help='number of total epochs (pretrain) to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='source',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--weighted-sample', action='store_true')
    parser.add_argument('--distributed-training', action='store_true')
    args = parser.parse_args()
    main(args)

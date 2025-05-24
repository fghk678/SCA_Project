"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import sys
import logging
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from utils.load_data import load_data
from utils.select_model_dataset import load_model_dataset
from utils.model_utils import validate
from utils.tllib.modules.domain_discriminator import DomainDiscriminator
from utils.tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from utils.tllib.utils.data import ForeverDataIterator
from utils.tllib.utils.metric import accuracy
from utils.tllib.utils.meter import AverageMeter, ProgressMeter
from utils.tllib.utils.logger import CompleteLogger
from utils.tllib.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
# dataset parameters
parser.add_argument('data_path', metavar='DIR',
                    help='root path of dataset')
parser.add_argument('--data_name', type=str, default="OfficeHome", help='dataset')
parser.add_argument('--data_folder_name', type=str, default="OfficeHome", help='dataset')
parser.add_argument('--source', type=str, default="Ar", help='source domain')
parser.add_argument('--target', type=str, default="Cl", help='target domain')
parser.add_argument('--train-resizing', type=str, default='default')
parser.add_argument('--val-resizing', type=str, default='default')
parser.add_argument('--resize-size', type=int, default=224,
                    help='the image size after resizing')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--no-hflip', action='store_true',
                    help='no random horizontal flipping during training')
parser.add_argument('--normalize', type=int, default=1, help='mean normalize for our formulation.')
parser.add_argument('--norm-mean', type=float, nargs='+',
                    default=(0.485, 0.456, 0.406), help='normalization mean')
parser.add_argument('--norm-std', type=float, nargs='+',
                    default=(0.229, 0.224, 0.225), help='normalization std')
# model parameters
parser.add_argument('--model_type', type=str, default="resnet resnet50 2048", 
                    help='pretrained feature extractor (resnet, resnet50, 2048)')

# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=utils.get_model_names(),
#                     help='backbone architecture: ' +
#                             ' | '.join(utils.get_model_names()) +
#                             ' (default: resnet18)')
parser.add_argument('--bottleneck-dim', default=256, type=int,
                    help='Dimension of bottleneck')
parser.add_argument('--no-pool', action='store_true',
                    help='no pool layer after the feature extractor.')
parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
parser.add_argument('--trade-off', default=1., type=float,
                    help='the trade-off hyper-parameter for transfer loss')
# training parameters
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                    help='Number of iterations per epoch')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--per-class-eval', action='store_true',
                    help='whether output per-class accuracy during evaluation')
parser.add_argument("--log", type=str, default='dann',
                    help="Where to save logs, checkpoints and debugging images.")
parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                    help="When phase is 'test', only test the model."
                            "When phase is 'analysis', only analysis the model.")
parser.add_argument('--log_file', type=str, default="", help='log file path, default is auto generated')

args = parser.parse_args()

sys.path.append(args.data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置日志
if not args.log_file:
    os.makedirs("logs", exist_ok=True)
    # timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print(args.model_type)
    model_name = args.model_type.split(" ")[1]
    log_file = f"logs/dann/{model_name}/{args.data_name}/{args.source}_{args.target}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    args.log_file = log_file

# 配置日志记录

logger = CompleteLogger(args.log_file, args.phase)

model_type = [i.strip().replace(",", "") for i in args.model_type.split(" ") if i != "" ]
model_type[-1] = int(model_type[-1])
# 记录实验配置
logger.logger.write(f"开始 {args.source} -> {args.target} 实验，种子: {args.seed}, GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}\n")
logger.logger.write(f"配置: 架构={args.model_type[1]}, bottleneck维度={args.bottleneck_dim}, trade-off={args.trade_off}\n")
logger.logger.write(f"学习率: lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}\n")



if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    # logger.logger.write("Seed selected deterministic training.\n")
    # warnings.warn('You have chosen to seed training. '
    #                 'This will turn on the CUDNN deterministic setting, '
    #                 'which can slow down your training considerably! '
    #                 'You may see unexpected behavior when restarting '
    #                 'from checkpoints.\n')

cudnn.benchmark = True
logger.logger.write(f"Given {model_type}...loading models and datasets\n")

feature, feature_size, train_source_loader, train_target_loader, num_classes, args.class_names = load_model_dataset(model_type, args, device)

#Validation set and covariance matrix loading
v1, v2, y1, y2 = load_data(model_type, device, feature, augment_factor=1, 
                           train_source_loader=train_source_loader, train_target_loader= train_target_loader, 
                           feature_size=feature_size, args=args, max_num_samples=None)

#Feature mean=0
if args.normalize:
    v1 = v1 - v1.mean(axis=0)
    v2 = v2 - v2.mean(axis=0)

y1 = y1.type(torch.LongTensor)
y2 = y2.type(torch.LongTensor)
logger.logger.write("Finished loading datasets and models\n")

# 创建验证和测试数据加载器
val_loader = DataLoader(
    train_target_loader.dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)
test_loader = DataLoader(
    train_target_loader.dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)

train_source_iter = ForeverDataIterator(train_source_loader)
train_target_iter = ForeverDataIterator(train_target_loader)

# create model
logger.logger.write("=> using model '{}'\n".format(args.model_type[1]))

classifier = ImageClassifier(feature, num_classes, bottleneck_dim=args.bottleneck_dim,
                             finetune=not args.scratch, in_features=feature_size).to(device)
domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

# define optimizer and lr scheduler
optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

# define loss function
domain_adv = DomainAdversarialLoss(domain_discri).to(device)

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        
            
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.logger.write(f"Epoch {epoch}, batch {i} loss {loss.item():.6f} cls_loss {cls_loss.item():.6f} "
                      f"transfer_loss {transfer_loss.item():.6f} cls_acc {cls_acc.item():.3f} "
                      f"domain_acc {domain_acc.item():.3f}\n")

if __name__ == '__main__':
  
    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        logger.logger.write("Saving t-SNE to {}\n".format(tSNE_filename))
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        logger.logger.write("A-distance = {}\n".format(A_distance))

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args, device)
        logger.logger.write("Test accuracy: {}\n".format(acc1))

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        logger.logger.write("Current learning rate: {}\n".format(lr_scheduler.get_last_lr()[0]))
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
                lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    logger.logger.write("Best accuracy: {:3.4f}\n".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args, device)
    logger.logger.write("Test accuracy: {:3.4f}\n".format(acc1))

    logger.close()

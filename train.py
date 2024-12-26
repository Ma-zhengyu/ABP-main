# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
# torch.backends.cudnn.enabled = False
from torch.utils.tensorboard import SummaryWriter
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.BCL_loss import BalSCL


def print_memory_usage():
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def accuracy_from_labels(predictions, targets):
    correct = predictions.eq(targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total * 100.0
    return accuracy


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments



def compute_py(train_loader, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    value_list = list(label_freq.values())
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(args.device)
    return value_list, label_freq_array


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def compute_adjustment(train_loader, tro, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def compute_adjustment_by_py(py, tro, args):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(args.device)
    return adjustments


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'smallimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet_proto','wideresnet', 'resnext', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=250000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    parser.add_argument('--num-max', default=500, type=int,
                        help='the max number of the labelled data')
    parser.add_argument('--num-max-u', default=4000, type=int,
                        help='the max number of the unlabeled data')
    parser.add_argument('--imb-ratio-label', default=1, type=int,
                        help='the imbalanced ratio of the labelled data')
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int,
                        help='the imbalanced ratio of the unlabeled data')
    parser.add_argument('--flag-reverse-LT', default=0, type=int,
                        choices=[0,1, 2, 3,4],
                        help='0 same;1 reverse; 2 random; 3 center; 4 off-center')

    parser.add_argument('--ema-mu', default=0.99, type=float,
                        help='mu when ema')

    parser.add_argument('--tau1', default=1, type=float,
                        help='tau for head1 consistency')
    parser.add_argument('--tau12', default=2, type=float,
                        help='tau for head2 consistency')
    parser.add_argument('--tau2', default=2, type=float,
                        help='tau for head2 balanced CE loss')
    parser.add_argument('--ema-u', default=0.9, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')

    parser.add_argument('--est-epoch', default=6, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')
    
    args = parser.parse_args()
    global best_acc
    global best_acc_b

    
    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'wideresnet_proto':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'resnet':
            import models.resnet_ori as models
            model = models.ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.dataset_name = 'cifar-10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'wideresnet_proto':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.dataset_name = 'cifar-100'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.dataset_name = 'stl-10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'svhn':
        args.num_classes = 10
        args.dataset_name = 'svhn'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'smallimagenet':
        args.num_classes = 127
        if args.img_size == 32:
            args.dataset_name = 'imagenet32'
        elif args.img_size == 64:
            args.dataset_name = 'imagenet64'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, 'datasets/'+args.dataset_name)

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    
    args.est_step = 0
    
    cls_num_list, args.py_con = compute_py(labeled_trainloader, args)
    args.u_py = torch.ones(args.num_classes) / args.num_classes
    args.u_py = args.u_py.to(args.device)


    args.u_py2 = torch.zeros(args.num_classes)
    args.u_py2 = args.u_py2.to(args.device)


    args.use_rate = torch.ones(args.num_classes)
    args.use_rate = args.use_rate.to(args.device)



    args.adjustment_expert_x = compute_adjustment_by_py(args.py_con, args.tau2, args)
    args.adjustment_balance = compute_adjustment_by_py(args.py_con, args.tau2, args)
    args.adjustment_expert_x2 = compute_adjustment_by_py(args.u_py, 0, args)
    args.adjustment_expert_u = compute_adjustment_by_py(args.u_py, 0, args)


    args.taumin = 0
    args.taumax = args.tau1

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = 'FixMatch-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(['Top1_b_acc', 'Top5_b_acc', 'Best_Top1_b_acc', 'Best_Top1_b2_acc','Best_Top1_joint_acc','Best_Top1_sum_acc'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0



    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)
    args.logger.close()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_b
    test_accs = []
    avg_time = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        max_count = args.use_rate.max()
        args.use_rate = args.use_rate / max_count
        # 计算每个类的阈值，等于对应位置的概率值乘以固定阈值
        class_thresholds = args.use_rate * args.threshold
        print_memory_usage()
        print('current epoch: ', epoch+1)
        print('平衡分支：')
        print(args.u_py)
        print('专家分支：')
        print(args.u_py2)
        print('利用率标准化之后：')
        print(args.use_rate)
        print('每个类的阈值分别是：')
        print(class_thresholds)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)


            try:
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)

            mask_l = (u_real != -2)
            mask_l = mask_l.cuda()

            data_time.update(time.time() - end)
            batch_size = targets_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3*args.mu+1).to(args.device)
            
            targets_x = targets_x.to(args.device)

            logits_feat = model(inputs)

            logits_b = model.classify1(logits_feat)
            logits_b = de_interleave(logits_b, 3 * args.mu + 1)
            logits_x_b = logits_b[:batch_size]
            logits_u_w_b, logits_u_s_b, logits_u_s1_b = logits_b[batch_size:].chunk(3)
            del logits_b
            Lx_b = F.cross_entropy(logits_x_b+args.adjustment_balance, targets_x, reduction='mean')

            logits_expert = model.classify2(logits_feat)
            logits_expert = de_interleave(logits_expert, 3*args.mu+1)
            logits_x_expert = logits_expert[:batch_size]
            logits_u_w_expert, logits_u_s_expert, logits_u_s1_expert = logits_expert[batch_size:].chunk(3)
            del logits_expert
            Lx_b2 = (F.cross_entropy(logits_x_expert+args.adjustment_expert_x-args.adjustment_expert_x2, targets_x, reduction='mean')).mean()

            pseudo_label_b = torch.softmax(logits_u_w_b.detach() / args.T, dim=-1)
            pseudo_label_expert = torch.softmax(logits_u_w_expert.detach() / args.T, dim=-1)
            
            max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)     
            max_probs_expert, targets_u_expert = torch.max(pseudo_label_expert, dim=-1)

            mask_b = max_probs_b.ge(args.threshold) 
            mask_expert = max_probs_expert.ge(args.threshold)

            all_counts_b = torch.bincount(targets_u_b[mask_l*mask_b], minlength=args.num_classes).detach()
            all_counts_expert = torch.bincount(targets_u_expert[mask_l*mask_expert], minlength=args.num_classes).detach()

            now_mask = torch.zeros(args.num_classes)
            now_mask = now_mask.to(args.device)

            now_mask2 = torch.zeros(args.num_classes)
            now_mask2 = now_mask2.to(args.device)


            if epoch > args.est_epoch:

                args.est_step = args.est_step + 1

                if all_counts_b.sum() > 0  and all_counts_expert.sum() > 0:
                    now_mask = all_counts_b / all_counts_b.sum()  
                    now_mask2 = all_counts_expert / all_counts_expert.sum()

                    
                    args.u_py = args.ema_u * args.u_py + (1-args.ema_u) * now_mask
                    args.u_py2 = args.ema_u * args.u_py2 + (1-args.ema_u) * now_mask2
                    args.adjustment_expert_x2 = compute_adjustment_by_py(args.u_py, args.tau1, args)
                    args.adjustment_expert_u = compute_adjustment_by_py(args.u_py, args.tau2, args)

             
                logits_u_s_b_twice = torch.cat([logits_u_s_b+args.adjustment_expert_u, logits_u_s1_b+args.adjustment_expert_u], dim=0).cuda()
                logits_u_s_b2_twice = torch.cat([logits_u_s_expert, logits_u_s1_expert], dim=0).cuda()
                targets_u_twice = torch.cat([targets_u_expert, targets_u_expert], dim=0).cuda() 
                mask_twice = torch.cat([mask_expert, mask_expert], dim=0).cuda()


                Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_twice, reduction='none') * mask_twice).mean()
                Lu_b2 = (F.cross_entropy(logits_u_s_b2_twice, targets_u_twice, reduction='none') * mask_twice).mean()

               
                loss =Lx_b + Lx_b2 + Lu_b  + Lu_b2
            else:
                loss =Lx_b+Lx_b2

            
            loss.backward()
            
            losses.update(loss.item())
            losses_x.update(Lx_b.item()+Lx_b2.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_acc_b, test_top5_acc_b,test_acc_b1, top1_joint, top1_sum = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_b, epoch)


            is_best = test_acc_b > best_acc_b

  
            best_acc_b = max(test_acc_b, best_acc_b)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch+1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'best_acc': best_acc_b,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                
                }, is_best, args.out, epoch_p=epoch+1)

            test_accs.append(test_acc_b)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc_b))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc_b, test_top5_acc_b, best_acc_b,test_acc_b1, top1_joint, top1_sum])
    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()


    top1_b = AverageMeter()
    top5_b = AverageMeter()
    top1_b2 = AverageMeter()

    top1_joint = AverageMeter()
    top1_sum = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_feat = model(inputs)

            outputs_b= model.classify1(outputs_feat)
            outputs_b2 = model.classify2(outputs_feat)

            loss = F.cross_entropy(outputs_b, targets)



            prob_b = torch.softmax(outputs_b, dim=1)  # Softmax for the second classifier
            prob_b2 = torch.softmax(outputs_b2, dim=1)  # Softmax for the third classifier
            


            prob_b_max, pred_b = torch.max(prob_b, dim=1)  # Get the max probability and predicted class
            prob_b2_max, pred_b2 = torch.max(prob_b2, dim=1)  # Get the max probability and predicted class


            max_prob, final_pred_idx = torch.max(torch.stack([prob_b_max, prob_b2_max], dim=1), dim=1)


            final_preds = torch.stack([pred_b, pred_b2], dim=1)
            final_pred_labels = final_preds.gather(1, final_pred_idx.view(-1, 1))

            total_prob =prob_b + prob_b2

            

            prec1_b, prec5_b = accuracy(outputs_b, targets, topk=(1, 5))
            prec1_b2, prec5_b2 = accuracy(outputs_b2, targets, topk=(1, 5))
            prec1_sum, prec5_sum = accuracy(total_prob, targets, topk=(1, 5))

            joint_prec1 = accuracy_from_labels(final_pred_labels.view(-1), targets)

            

            top1_b.update(prec1_b.item(), inputs.shape[0])
            top5_b.update(prec5_b.item(), inputs.shape[0])
            top1_b2.update(prec1_b2.item(), inputs.shape[0])
            top1_joint.update(joint_prec1, inputs.shape[0])
            top1_sum.update(prec1_sum.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1_b.avg))
    logger.info("top-5 acc: {:.2f}".format(top5_b.avg))

    return top1_b.avg, top5_b.avg, top1_b2.avg,top1_joint.avg,top1_sum.avg



if __name__ == '__main__':
    main()

# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
import setGPU
import argparse
import os
import torch
from torch.nn import CrossEntropyLoss, BCELoss, Sigmoid
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import time
import datetime
import math
from train_utils import AverageMeter, accuracy, init_logfile, log, multi_accuracy
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch AWA Training')
parser.add_argument('--dataset', type=str, default = "AWA", choices=DATASETS)
parser.add_argument('--arch', type=str, default = "resnet50",choices=ARCHITECTURES)
parser.add_argument('--attribute', default=0, type=int, metavar='N',
                    help='index for attribute')
parser.add_argument('--outdir', type=str, default = "attribute_models/", help='folder to save model and training log)')
parser.add_argument('--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=20,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.50, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--weight', default=2, type=int, metavar='N',
                    help='the loss weight for tp/tn')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

def main(attribute):
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train', attribute = attribute)
    test_dataset = get_dataset(args.dataset, 'test', attribute = attribute)
    pin_memory = (args.dataset == "imagenet")
    
    ## balanced training ##
    samples_weights = train_dataset.samples_weights()
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights) , replacement=True)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch,
                              sampler = sampler, num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset, classes = 1)

    logfilename = os.path.join(args.outdir, 'attribute_%d_log.txt' % attribute)
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttest loss\ttrain acc(multi)\ttest acc(multi)")

    criterion = BCELoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)
        after = time.time()
        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}||\t{}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_last_lr()[0], train_loss, test_loss, train_acc, test_acc))
        
        scheduler.step()
        torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'attribute_%d.pth.tar'% (attribute)))

def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = [AverageMeter() ]
    end = time.time()

    # switch to train mode
    model.train()
    global wei_tgt
    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        outputs = model(inputs)
        m = Sigmoid()
        confidence = m(outputs)
        weight = torch.ones(targets.shape)
        weight[targets==1] = wei_tgt
        criterion = BCELoss(weight = weight).cuda()
        loss = criterion(confidence, targets)
        # measure accuracy and record loss
        pred = confidence.data.round()
        acc1 = multi_accuracy(pred, targets)
        losses.update(loss.item(), inputs.size(0))
        for j in range(len(top1)):
            top1[j].update(acc1[j].item(), inputs.size(0),pred[:,j], targets[:,j] )
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        multi_acc_log = ''
        for k in range(len(top1)):
            multi_acc_log += '%s_acc/TP/TN: %.2f/%.2f/%.2f; ' % (f'attribute_{attribute}', top1[k].avg,top1[k].avg_tp,top1[k].avg_tn)
            
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            for k in range(len(top1)):
                print('%s_acc/TP/TN: %.2f (%.2f)/%.2f/%.2f; ' % (f'attribute_{attribute}', top1[k].val, top1[k].avg,top1[k].avg_tp,top1[k].avg_tn))
            print('------------------------------------------')
    return (losses.avg, multi_acc_log)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = [AverageMeter() ]
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs)
            m = Sigmoid()
            confidence = m(outputs)
            loss = criterion(confidence, targets)
            # measure accuracy and record loss
            pred = confidence.data.round()
            acc1 = multi_accuracy(pred, targets)
            losses.update(loss.item(), inputs.size(0))
            for j in range(len(top1)):
                top1[j].update(acc1[j].item(), inputs.size(0),pred[:,j], targets[:,j] )
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            multi_acc_log = ''
            for k in range(len(top1)):
                multi_acc_log += '%s_acc/TP/TN: %.2f/%.2f/%.2f; ' % (f'attribute_{attribute}', top1[k].avg,top1[k].avg_tp,top1[k].avg_tn)

            if i % args.print_freq == 0 or i == len(loader)-1 :
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for k in range(len(top1)):
                    print('%s_acc/TP/TN: %.2f (%.2f)/%.2f/%.2f; ' % (f'attribute_{attribute}', top1[k].val, top1[k].avg,top1[k].avg_tp,top1[k].avg_tn))
                print('------------------------------------------')
        return (losses.avg, multi_acc_log)


if __name__ == "__main__":
    wei_tgt = args.weight
    attribute = args.attribute
    main(attribute)

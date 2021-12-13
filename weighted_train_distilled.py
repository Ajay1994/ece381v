#python weighted_train_distilled.py  --swa
import argparse
import os
import random
import shutil
import time
import sys

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from utils.prepare_data import get_data_models
from utils.misc import save_checkpoint, AverageMeter, moving_average, bn_update

from progress.bar import Bar as Bar
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

# ############################### Dataset ###############################
parser.add_argument('-d', '--dataset', default='Lesion', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# ############################### Optimization Option ###############################
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test_batch', default=64, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 35, 55, 70], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')


# ############################### Checkpoints ###############################
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--distill', default=['resnet18/model_best.pth.tar', 'resnet50/model_best.pth.tar', 'mobilenet/model_best.pth.tar', 'densenet/model_best.pth.tar'], type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


# ############################### Architecture ###############################
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', help='model architecture: ' + ' (default: resnet18)')
parser.add_argument('--activation', type=str, default="relu", help='Activation Function to use')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')


# ############################### Misc ###############################
parser.add_argument('--manualSeed', type=int, default=5094, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--save_dir', default='weighted_distillationt10_mobilenet', type=str)

# Knowledge distillation parameters
parser.add_argument('--temperature', default=10, type=float, help='temperature of KD')
parser.add_argument('--alpha', default=0.9, type=float, help='ratio for KL loss')

# ############################### Device Option ###############################
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

########################## SWA setting ##########################
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=15, metavar='N', help='SWA start epoch number (default: 55)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N', help='SWA model collection frequency/cycle length in epochs (default: 1)')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    
def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)

    # ############################### Dataset ###############################
    print('==> Preparing dataset %s' % args.dataset)
    dataloaders = get_data_models(args)
    img_batch, label_batch = next(iter(dataloaders["train"]))
    print("Data Loaded: {} | {}".format(img_batch.shape, label_batch.shape))
    
    # ############################### Model ###############################
    if args.arch == "resnet18":
        model = torchvision.models.resnet18()
        swa_model = torchvision.models.resnet18()
    elif args.arch == "resnet34":
        model = torchvision.models.resnet34()
    elif args.arch == "resnet50":
        model = torchvision.models.resnet50()
    elif args.arch == "densenet":
        model = torchvision.models.densenet121()
        num_fltr = model.classifier.in_features
        model.classifier = nn.Linear(num_fltr, label_batch.shape[1])
        model = torch.nn.DataParallel(model)
    
        swa_model = torchvision.models.densenet121()
        num_fltr = swa_model.classifier.in_features
        swa_model.classifier = nn.Linear(num_fltr, label_batch.shape[1])
        swa_model = torch.nn.DataParallel(swa_model)
    elif args.arch == "mobilenet":
        model = torchvision.models.mobilenet_v2()
        num_fltr = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_fltr, label_batch.shape[1])
        model = torch.nn.DataParallel(model)
    
        swa_model = torchvision.models.mobilenet_v2()
        num_fltr = swa_model.classifier[1].in_features
        swa_model.classifier[1] = nn.Linear(num_fltr, label_batch.shape[1])
        swa_model = torch.nn.DataParallel(swa_model)
        

#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, label_batch.shape[1])
#     model = torch.nn.DataParallel(model)
    
#     num_ftrs = swa_model.fc.in_features
#     swa_model.fc = nn.Linear(num_ftrs, label_batch.shape[1])
#     swa_model = torch.nn.DataParallel(swa_model)
    
    model_ref1 = torchvision.models.resnet18()
    num_ftrs = model_ref1.fc.in_features
    model_ref1.fc = nn.Linear(num_ftrs, label_batch.shape[1])
    model_ref1 = torch.nn.DataParallel(model_ref1)
    
    model_ref2 = torchvision.models.resnet50()
    num_ftrs = model_ref2.fc.in_features
    model_ref2.fc = nn.Linear(num_ftrs, label_batch.shape[1])
    model_ref2 = torch.nn.DataParallel(model_ref2)
    
    model_ref3 = torchvision.models.mobilenet_v2()
    num_fltr = model_ref3.classifier[1].in_features
    model_ref3.classifier[1] = nn.Linear(num_fltr, label_batch.shape[1])
    model_ref3 = torch.nn.DataParallel(model_ref3)
    
    model_ref4 = torchvision.models.densenet121()
    num_fltr = model_ref4.classifier.in_features
    model_ref4.classifier = nn.Linear(num_fltr, label_batch.shape[1])
    model_ref4 = torch.nn.DataParallel(model_ref4)
    
    model.cuda()
    swa_model.cuda()
    model_ref1.cuda()
    model_ref2.cuda()
    model_ref3.cuda()
    model_ref4.cuda()
    
    
    # ############################### Optimizer and Loss ###############################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
   
    # ############################### Resume ###############################
    title = args.dataset + "-" + args.arch
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    if args.distill:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.distill[0]), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.distill[0])
        model_ref1.load_state_dict(checkpoint['state_dict'])
        
        
        assert os.path.isfile(args.distill[1]), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.distill[1])
        model_ref2.load_state_dict(checkpoint['state_dict'])
        
        assert os.path.isfile(args.distill[2]), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.distill[2])
        model_ref3.load_state_dict(checkpoint['state_dict'])
        
        assert os.path.isfile(args.distill[3]), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.distill[3])
        model_ref4.load_state_dict(checkpoint['state_dict'])
        
        
    # evaluate with random initialization parameters
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    
    # save random initialization parameters
    save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')
    
    # ############################### Train and val ###############################
    best_f1 = 0.0
    fopen = open(args.save_dir + "/log_dir.txt", "w")
    fopen2 = open(args.save_dir + "/logger.txt", "w")
    swa_n = 0  
    
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        score = ""
        
        precision, recall, f1_score, micro_f1, macro_f1, accuracy  = train(dataloaders["train"], model, model_ref1, model_ref2, model_ref3, model_ref4, criterion, optimizer, epoch, use_cuda)
        print(" Train : Precision : {:.3f} || Recall : {:.3f} || F1-Score : {:.3f} || Micro F1-Score : {:.3f} || Macro F1-Score : {:.3f} || Accuracy : {:.3f}".format(precision, recall, f1_score, micro_f1, macro_f1, accuracy))
        score += str(f1_score) + "\t"
        
        if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:
            print("SWA Started")
            # SWA
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(dataloaders["train"], swa_model)
        
        precision, recall, f1_score, micro_f1, macro_f1, accuracy = test(dataloaders["val"], swa_model, criterion, epoch, use_cuda)
        print(" Valid : Precision : {:.3f} || Recall : {:.3f} || F1-Score : {:.3f} || Micro F1-Score : {:.3f} || Macro F1-Score : {:.3f} || Accuracy : {:.3f}".format(precision, recall, f1_score, micro_f1, macro_f1, accuracy))
        score += str(f1_score) + "\t ||"
        
        precision, recall, f1_score, micro_f1, macro_f1, accuracy = test(dataloaders["test"], swa_model, criterion, epoch, use_cuda)
        print(" Test : Precision : {:.3f} || Recall : {:.3f} || F1-Score : {:.3f} || Micro F1-Score : {:.3f} || Macro F1-Score : {:.3f} || Accuracy : {:.3f}".format(precision, recall, f1_score, micro_f1, macro_f1, accuracy))
        score += str(f1_score) + " - " + str(micro_f1) + " - " + str(macro_f1) + " - " + str(accuracy) + "\n"
        fopen2.write(score)
        fopen2.flush()
        
        fopen.write(" Test : Precision : {:.3f} || Recall : {:.3f} || F1-Score : {:.3f} || Micro F1-Score : {:.3f} || Macro F1-Score : {:.3f} || Accuracy : {:.3f}\n".format(precision, recall, f1_score, micro_f1, macro_f1, accuracy))
        fopen.flush()
        
        if f1_score > best_f1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': swa_model.state_dict(),
                'f1_score': f1_score,
                'prec': precision,
                'recall': recall,
                'optimizer' : optimizer.state_dict(),
            }, True, checkpoint=args.save_dir)
            best_f1 = f1_score
            
        
    
    # ################################### test ###################################
    print('Load best model ...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("-"*100)
    precision, recall, f1_score, micro_f1, macro_f1, accuracy = test(dataloaders["test"], swa_model, criterion, epoch, use_cuda)
    print("\n Test : Precision : {:.3f} || Recall : {:.3f} || F1-Score : {:.3f} || Micro F1-Score : {:.3f} || Macro F1-Score : {:.3f} || Accuracy : {:.3f}".format(precision, recall, f1_score, micro_f1, macro_f1, accuracy))
    
    fopen.write("-"*50)
    fopen.write("\n Test : Precision : {:.3f} || Recall : {:.3f} || F1-Score : {:.3f} || Micro F1-Score : {:.3f} || Macro F1-Score : {:.3f} || Accuracy : {:.3f}".format(precision, recall, f1_score, micro_f1, macro_f1, accuracy))
    fopen.close()
    
def train(trainloader, model, model_ref1,  model_ref2, model_ref3, model_ref4,  criterion, optimizer, epoch, use_cuda):
    model.train()
    model_ref1.eval()
    model_ref2.eval()
    model_ref3.eval()
    model_ref4.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    end = time.time()
    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        targets = torch.max(targets, 1)[1]
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # get teacher model outputs
        with torch.no_grad():
            ref1_logit = model_ref1(inputs)
            ref2_logit = model_ref2(inputs)
            ref3_logit = model_ref3(inputs)
            ref4_logit = model_ref4(inputs)
        
        outputs = model(inputs)
        
#         loss = criterion(outputs, targets.data)
        loss = loss_fn_kd(outputs, targets.data, ref1_logit, ref2_logit, ref3_logit, ref4_logit)
        losses.update(loss.item(), inputs.size(0))
        
        gt = torch.cat((gt, targets.data))
        pred = torch.cat((pred, outputs.data.topk(1)[1].squeeze(1)), 0)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
#     print(gt.shape, pred.shape)
#     print(precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = None)[2])
    scores = precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = "weighted", zero_division = 0)
    micro_scores = precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = "micro", zero_division = 0)
    macro_scores = precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = "macro", zero_division = 0)
    return scores[0], scores[1], scores[2], micro_scores[2], macro_scores[2], accuracy_score(gt.cpu(), pred.cpu())
        
def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)
            targets = torch.max(targets, 1)[1]

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            
            gt = torch.cat((gt, targets.data))
            pred = torch.cat((pred, outputs.data.topk(1)[1].squeeze(1)), 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg
                        )
            bar.next()
    bar.finish()
    scores = precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = "weighted", zero_division = 0)
    micro_scores = precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = "micro", zero_division = 0)
    macro_scores = precision_recall_fscore_support(gt.cpu(), pred.cpu(), average = "macro", zero_division = 0)
    return scores[0], scores[1], scores[2], micro_scores[2], macro_scores[2], accuracy_score(gt.cpu(), pred.cpu())

def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss

def loss_fn_kd(outputs, labels, ref_logit1, ref_logit2, ref_logit3, ref_logit4):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! 
    """
    alpha = args.alpha
    T = args.temperature
    kd_weight = [0.7, 0.2, 0.09, 0.01]
    
    kd_weight_shuffle = random.sample(kd_weight, len(kd_weight))    
    
    _q = F.log_softmax(outputs/T, dim=1)
    _t1 = F.softmax(ref_logit1/T, dim=1)
    _t2 = F.softmax(ref_logit2/T, dim=1)
    _t3 = F.softmax(ref_logit3/T, dim=1)
    _t4 = F.softmax(ref_logit4/T, dim=1)
    CE = F.cross_entropy(outputs, labels)
    
    _t = [_t1, _t2, _t3, _t4]
    
    KD_loss = 0.0
    for i in range(0, len(kd_weight_shuffle)):
        KD_loss += nn.KLDivLoss()(_q, _t[i]) * (alpha * T * T * kd_weight_shuffle[i])
        
    KD_loss  +=  CE * (1. - alpha)

    return KD_loss
    
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
if __name__ == '__main__':
    main()

    
    

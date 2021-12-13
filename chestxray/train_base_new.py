#python train_base.py --arch resnet18 --save_dir resnet18/
import argparse
import os
import random
import shutil
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import roc_auc_score

from utils.prepare_data import get_data_models
from utils.misc import save_checkpoint, AverageMeter

from progress.bar import Bar as Bar
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support

# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

# ############################### Dataset ###############################
parser.add_argument('-d', '--dataset', default='NIH ChestX-rays', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# ############################### Optimization Option ###############################
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=512, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test_batch', default=256, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 15, 25], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')


# ############################### Checkpoints ###############################
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


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
parser.add_argument('--save_dir', default='chest_resent18/', type=str)


# ############################### Device Option ###############################
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

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
    index_batch, img_batch, label_batch = next(iter(dataloaders["train"]))
    print("Data Loaded: {} | {}".format(img_batch.shape, label_batch.shape))
    
#     sys.exit(0)
    # ############################### Model ###############################
    if args.arch == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, label_batch.shape[1])
    elif args.arch == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, label_batch.shape[1])
    
    model = torch.nn.DataParallel(model)
    model.cuda()
   
    # ############################### Optimizer and Loss ###############################
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    
    # ############################### Resume ###############################
    title = args.dataset + "-" + args.arch
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_f1 = checkpoint['f1_score']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Best F1-Score : {}".format(best_f1)) # Valid : Precision : 0.7257768869054372 || Recall : 0.7616580310880829 || F1-Score : 0.7397434352609674
        
    # evaluate with random initialization parameters
    if args.evaluate:
        print('\nEvaluation only')
        precision, recall, f1_score = test(dataloaders["val"], model, criterion, -1, use_cuda)
        print(" Valid : Precision : {} || Recall : {} || F1-Score : {} ".format(precision, recall, f1_score))
        return

    # save random initialization parameters
    save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')
    
    # ############################### Train and val ###############################
    best_auc = 0.0
    fopen = open(args.save_dir + "log_dir.txt", "w")
    fopen2 = open(args.save_dir + "logger.txt", "w")
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        score = ""
        
        mean_AUC, AUCs  = train(dataloaders["train"], model, criterion, optimizer, epoch, use_cuda)
        print(" Train :  Mean AUC : {} || AUCs : {} ".format(mean_AUC, AUCs))
        score += str(mean_AUC) + "\t"
        
        mean_AUC, AUCs = test(dataloaders["val"], model, criterion, epoch, use_cuda)
        print(" Valid :  Mean AUC : {} || AUCs : {} ".format(mean_AUC, AUCs))
        score += str(mean_AUC) + "\t"
        
        mean_AUC, AUCs = test(dataloaders["test"], model, criterion, epoch, use_cuda)
        print(" Test :  Mean AUC : {} || AUCs : {} ".format(mean_AUC, AUCs))
        score += str(mean_AUC) + "\n"
        
        if mean_AUC > best_auc:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'auc_score': mean_AUC,
                'optimizer' : optimizer.state_dict(),
            }, True, checkpoint=args.save_dir)
            best_auc = mean_AUC
            
        
        fopen2.write(score)
        fopen2.flush()
        
        fopen.write(" Test : Mean AUC : {:.3f} || AUCs : {} \n".format(mean_AUC, AUCs))
        fopen.flush()
    # ################################### test ###################################
    print('Load best model ...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("-"*100)
    mean_AUC, AUCs = test(dataloaders["test"], model, criterion, epoch, use_cuda)
    print(" Test :  Mean AUC : {} || AUCs : {} ".format(mean_AUC, AUCs))
    fopen.write("-"*50)
    fopen.write("Best Test :  Mean AUC : {} || AUCs : {} ".format(mean_AUC, AUCs))
    
    
    fopen.close()
    fopen2.close()
    
def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    end = time.time()
    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (index, inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        
            
        outputs = model(inputs)
        gt = torch.cat((gt, targets.data), 0)
        pred = torch.cat((pred, outputs.data), 0)

        
        loss = criterion(outputs, targets.data)
        losses.update(loss.item(), inputs.size(0))
        
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
    
    mean_auc, AUCs = compute_AUCs(gt, pred)
    return mean_auc, AUCs
        
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
        for batch_idx, (index, inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            gt = torch.cat((gt, targets.data), 0)
            pred = torch.cat((pred, outputs.data), 0)
            
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
   
            
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
    
    
    mean_auc, AUCs = compute_AUCs(gt, pred)
    return mean_auc, AUCs
      
def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(14):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return np.array(AUROCs).mean(), AUROCs
    
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
if __name__ == '__main__':
    main()

    
    

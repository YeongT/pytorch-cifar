'''Train CIFAR10 with PyTorch.'''
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.backends.cudnn as cudnn
except ImportError:
    print('\nPyTorch가 설치되어 있지 않습니다.\n')
    print('설치 방법: README.md의 "Environment & Quick Start" 섹션을 참고하세요. 예:')
    print('  CPU-only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu')
    print('  CUDA 11.7 예시: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117')
    raise SystemExit(1)

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=16, type=int, help='training batch size (default 16 for 3GB GPUs)')
parser.add_argument('--test_batch', default=100, type=int, help='test batch size')
parser.add_argument('--num_workers', default=2, type=int, help='DataLoader num_workers (Windows: keep small)')
parser.add_argument('--accumulation_steps', default=1, type=int, help='gradient accumulation steps to simulate larger batch')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision (CUDA only)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    pin_memory=(device=='cuda'), persistent_workers=(args.num_workers>0))

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
    pin_memory=(device=='cuda'), persistent_workers=(args.num_workers>0))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')r
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# sync scheduler T_max with total epochs (use args.epochs to avoid NameError)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Maximum number of epochs
num_epochs = args.epochs

# Setup AMP scaler if requested and CUDA available
scaler = None
use_amp = args.use_amp and device == 'cuda'
if use_amp:
    scaler = torch.cuda.amp.GradScaler()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # forward under autocast when using AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            raw_loss = criterion(outputs, targets)
            loss = raw_loss / max(1, args.accumulation_steps)

        # backward and step with scaler if AMP enabled
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimizer step when enough accumulation steps have been seen
        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(trainloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_loss += raw_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # use autocast in eval for speed/memory when AMP enabled
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if start_epoch >= num_epochs:
    print('start_epoch (%d) >= num_epochs (%d). Nothing to do.' % (start_epoch, num_epochs))
else:
    for epoch in range(start_epoch, num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()

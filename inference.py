"""Inference and quick visualization for trained CIFAR-10 models.

Usage examples (from project root):
  python inference.py --checkpoint ./checkpoint/ckpt.pth --num-images 8
  python inference.py --checkpoint ./checkpoint/ckpt.pth --num-images 16 --save-grid results.png
  python inference.py --checkpoint ./checkpoint/ckpt.pth --onnx-export model.onnx

This script will:
 - load the checkpoint (default ./checkpoint/ckpt.pth)
 - build the model (default SimpleDLA; use --model to change)
 - run inference on a small random subset of the CIFAR-10 test set
 - show a grid with predicted / true labels and confidences
 - optionally export the model to ONNX

"""

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print('\nPyTorch가 설치되어 있지 않습니다. README.md를 참고하여 설치하세요.')
    raise SystemExit(1)

import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision
import torchvision.transforms as transforms

import models as models_module


parser = argparse.ArgumentParser(description='Inference & visualization for CIFAR-10 model')
parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth', help='path to checkpoint')
parser.add_argument('--model', default='SimpleDLA', help='model class name from models (default: SimpleDLA)')
parser.add_argument('--num-images', type=int, default=8, help='number of test images to display')
parser.add_argument('--save-grid', default=None, help='optional: save the image grid to this path')
parser.add_argument('--onnx-export', default=None, help='optional: export the model to ONNX at this path')
parser.add_argument('--device', default=None, help='device to use (cuda/cpu), default auto')
args = parser.parse_args()

# device
if args.device:
    device = args.device
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model class
if not hasattr(models_module, args.model):
    print(f"모델 클래스 '{args.model}'를 찾을 수 없습니다. models.py에서 사용 가능한 이름을 확인하세요.")
    raise SystemExit(1)
ModelClass = getattr(models_module, args.model)

# prepare model
net = ModelClass()
net = net.to(device)
net.eval()

# load checkpoint
if not os.path.isfile(args.checkpoint):
    print(f"체크포인트 '{args.checkpoint}'를 찾을 수 없습니다.")
    raise SystemExit(1)

ckpt = torch.load(args.checkpoint, map_location=device)
if 'net' in ckpt:
    state_dict = ckpt['net']
else:
    state_dict = ckpt

# DataParallel로 저장된 모델의 'module.' 접두사 제거
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 'module.' 제거
    else:
        new_state_dict[k] = v

try:
    net.load_state_dict(new_state_dict)
except Exception as e:
    print('체크포인트 로딩에 실패했습니다:', e)
    raise SystemExit(1)

print('모델 로드 완료, device=', device)

# dataset & transform (same as training)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# pick random subset
num = max(1, min(args.num_images, len(testset)))
indices = np.random.choice(len(testset), num, replace=False)
sub = torch.utils.data.Subset(testset, indices)
loader = torch.utils.data.DataLoader(sub, batch_size=num, shuffle=False, num_workers=0)

images, targets = next(iter(loader))
images = images.to(device)

with torch.no_grad():
    logits = net(images)
    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)

# helper to unnormalize for display
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

def unnormalize(img_tensor):
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img

# class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# plot grid
cols = min(4, num)
rows = math.ceil(num / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])

for i in range(rows*cols):
    r = i // cols
    c = i % cols
    ax = axes[r, c]
    ax.axis('off')
    if i < num:
        img = unnormalize(images[i])
        ax.imshow(img)
        true = classes[targets[i].item()]
        pred = classes[preds[i].item()]
        conf = confs[i].item()
        ax.set_title(f'P: {pred} ({conf:.2f})\nT: {true}')
    else:
        ax.set_visible(False)

plt.tight_layout()

if args.save_grid:
    outdir = os.path.dirname(args.save_grid)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(args.save_grid)
    print('결과 이미지 저장:', args.save_grid)
else:
    print('화면에 결과를 표시합니다. 창을 닫으면 스크립트가 종료됩니다.')
    plt.show()

# optional ONNX export
if args.onnx_export:
    dummy = torch.randn(1,3,32,32, device=device)
    net.eval()
    try:
        torch.onnx.export(net, dummy, args.onnx_export, input_names=['input'], output_names=['output'], opset_version=11)
        print('ONNX로 저장됨:', args.onnx_export)
    except Exception as e:
        print('ONNX export 실패:', e)

print('완료.')

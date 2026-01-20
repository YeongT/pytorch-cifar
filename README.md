# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |


---

Environment & Quick Start

1) (권장) Python 가상환경 생성 및 활성화
   - Windows (cmd):
     python -m venv .venv
     .venv\Scripts\activate

2) PyTorch 설치
   - PyTorch는 CUDA 버전(또는 CPU 전용)에 따라 설치 명령이 다릅니다. 정확한 명령은 공식 설치 페이지(https://pytorch.org/get-started/locally/)에서 확인하세요.

   - 예: CPU 전용 (간단)
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   - 예: CUDA 11.7 (일반적인 조합 — 사용자의 GPU 드라이버/환경과 확인하세요)
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

   - 예: CUDA 12.1
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   주의: 위 URL은 예시이며, 사용자의 시스템(드라이버 버전, CUDA 툴킷 설치 여부)에 따라 적절한 wheel을 선택해야 합니다. 가능하면 https://pytorch.org/get-started/locally/에서 권장 명령을 그대로 복사해서 사용하세요.

3) (선택) 종속 패키지 설치
   pip install -r requirements.txt

4) 학습 실행 예시
   - 기본(10 epoch, batch_size=16):
     python main.py

   - 커스텀 예시 (예: 유효 배치 64를 위한 accumulation 사용)
     python main.py --epochs 10 --batch_size 16 --accumulation_steps 4 --num_workers 2

   - 단일 에폭(테스트용)
     python main.py --epochs 1 --batch_size 8 --test_batch 8 --num_workers 0

추가 안내
- 메모리가 적은 GPU(예: GTX 1060 3GB)에서는 기본 batch_size를 16으로 설정했습니다. OOM 발생 시 --batch_size를 더 작게(예: 8) 하거나 --accumulation_steps로 유효 배치를 늘리세요.
- Windows에서는 DataLoader의 --num_workers를 0~4 사이에서 실험하세요. (spawn 오버헤드 때문)
- Mixed precision(AMP)을 적용하면 메모리와 속도에 도움이 될 수 있습니다. 원하시면 프로젝트에 AMP 통합을 도와드리겠습니다.

문제 발생 시
- `ModuleNotFoundError: No module named 'torch'`가 나오면 PyTorch가 설치되지 않은 것입니다. 위 설치 단계 참고하세요.
- 그 밖의 에러가 발생하면 에러 로그를 붙여 알려주시면 더 자세히 도와드리겠습니다.

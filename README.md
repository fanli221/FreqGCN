# FreqGCN
#### FreqGCN: A Simple Yet Effective GCN-based Diffusion Model for Stochastic Human Motion Prediction
## 🛠 Setup
### 1. Python/Conda Environment

```
mkdir ./checkpoints
mkdir ./data
mkdir ./inference
mkdir ./results
conda create -n freqgcn python=3.8
conda activate freqgcn
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirement.txt
```

### 2. Datasets

**Datasets for [Human3.6M](http://vision.imar.ro/human3.6m/description.php), [HumanEva-I](http://humaneva.is.tue.mpg.de/) 
For Human3.6M and HumanEva-I, we adopt the data preprocessing from [GSPS](https://github.com/wei-mao-2019/gsps).

### 3. Pretrained Models

We provide the pretrained models for all three datasets [here](https://drive.google.com/drive/folders/16iPASM7pnYEixBXaVFnp2pGbjgg-Ppxq?usp=sharing). 

## 🔎 Evaluation
Evaluate on Human3.6M:
```
python main.py --cfg h36m --mode eval --ckpt ./ckeckpoints/h36m_ckpt.pt
```
Evaluate on HumanEva-I:
```
python main.py --cfg humaneva --mode eval --ckpt ./ckeckpoints/humaneva_ckpt.pt
```

## ⏳ Training

```
python main.py --cfg h36m --mode train
```
```
python main.py --cfg humaneva --mode train
```

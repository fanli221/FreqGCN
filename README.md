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

We adopt the data preprocessing from [GSPS](https://github.com/wei-mao-2019/gsps), which you can refer to [here](https://drive.google.com/drive/folders/1sb1n9l0Na5EqtapDVShOJJ-v6o-GZrIJ) and download all files into the `./data` directory.

### 3. Pretrained Models

We provide the pretrained models for all three datasets [here](https://drive.google.com/drive/folders/1YgYcUaAtIz5-RZyg8yz7ZCJW2-g4ZKT1?usp=sharing). The pretrained model need to be put in the `./checkpoints` directory.

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
Train on Human3.6M:
```
python main.py --cfg h36m --mode train
```
Train on HumanEva-I:
```
python main.py --cfg humaneva --mode train
```

## 🌹 Acknowledgment
We thank [HumanMAC](https://github.com/LinghaoChan/HumanMAC), [Transfusion](https://github.com/sibotian96/TransFusion), [Comusion](https://github.com/jsun57/CoMusion/) for making their code publicly available.

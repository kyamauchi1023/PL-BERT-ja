# PL-BERT-ja
A repository of Japanese Phoneme-Level BERT

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone git@github.com:kyamauchi1023/PL-BERT-ja.git
cd PL-BERT-ja
```
3. Create a new environment (recommended):
```bash
pipenv install --python 3.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Preprocessing
```bash
python preprocess.py
```
```bash
pgrep python | xargs kill -9
```

## Trianing
```bash
python train.py
```
```
tensorboard --logdir logs
```

## Finetuning
Please check [this README](alvpredictor/README.md)


## References
- [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)
- [tomaarsen/TTSTextNormalization](https://github.com/tomaarsen/TTSTextNormalization)
- [https://arxiv.org/abs/2301.08810](https://arxiv.org/abs/2301.08810)
- [https://github.com/yl4579/PL-BERT](https://github.com/yl4579/PL-BERT)

# PL-BERT-ja
A repository of Japanese [Phoneme-Level BERT](https://arxiv.org/abs/2301.08810)

This repository enables [https://github.com/yl4579/PL-BERT](https://github.com/yl4579/PL-BERT) to be pre-trained with the [Japanese wikipedia corpus](https://dumps.wikimedia.org).

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

## Trianing
```bash
python train.py
```
```
tensorboard --logdir logs
```

## Pre-trained model
Pre-trained model weights are available from [plb-ja_10000000-steps](https://drive.google.com/file/d/1TRazSiBGs1NnBvPLe1V96MUb5gTs2IcO/view?usp=sharing).

The downloaded zip file contains the following files:
- `10000000.pth.tar`: 10000000 steps pre-trained weights
- `config.yml`: contains the config used for pre-training
- `token_maps.pkl`: a list of the correspondence between the tokens and the ids created by the pre-processing
- `token_maps.txt`: `token_maps.pkl` made into a text file


## References
- [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)
- [tomaarsen/TTSTextNormalization](https://github.com/tomaarsen/TTSTextNormalization)
- [https://arxiv.org/abs/2301.08810](https://arxiv.org/abs/2301.08810)
- [https://github.com/yl4579/PL-BERT](https://github.com/yl4579/PL-BERT)
- [Japanese wikipedia corpus](https://dumps.wikimedia.org)

# ALP Predictor

```shell
$ pipenv shell
```

## 学習
```shell
$ python train_apm.py -p alvpredictor/config/JSUT/preprocess.yaml -t alvpredictor/config/JSUT/train.yaml --restore_epoch 0

$ python train_apm.py -p alvpredictor/config/osaka/preprocess.yaml -t alvpredictor/config/osaka/train.yaml --restore_epoch 1
```
```shell
$ tensorboard --logdir alvpredictor/output/log/JSUT
$ tensorboard --logdir alvpredictor/output/log/osaka
```

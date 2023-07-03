# ALP Predictor

```shell
$ pipenv shell
```

## 学習
```shell
$ python train_apm.py -p alvpredictor/config/JSUT/preprocess.yaml -m alvpredictor/config/JSUT/model.yaml -t alvpredictor/config/JSUT/train.yaml --restore_epoch 0

$ python train_apm.py -p alvpredictor/config/osaka/preprocess.yaml -m alvpredictor/config/osaka/model.yaml -t alvpredictor/config/osaka/train.yaml --restore_epoch 0
```
```shell
$ tensorboard --logdir alvpredictor/output/log/JSUT
$ tensorboard --logdir alvpredictor/output/log/osaka
```
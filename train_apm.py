import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import AlbertConfig, AlbertModel
from model import MultiTaskModel

from alvpredictor.dataset import Dataset
from alvpredictor.model import ALVPredictor
from alvpredictor.utils.tools import get_mask_from_lengths
from evaluate_apm import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config, bert_config = configs

    # Get dataset
    dataset = Dataset(
        "train", preprocess_config
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True
    )

    # prepare pre-trained model
    albert_base_configuration = AlbertConfig(**bert_config['model_params'])
    bert_ = AlbertModel(albert_base_configuration)
    num_vocab = 30924
    bert = MultiTaskModel(
        bert_,
        num_vocab=num_vocab,
        num_tokens=bert_config['model_params']['vocab_size'],
        hidden_size=bert_config['model_params']['hidden_size']
    )
    
    # load pre-trained weight
    load = True
    try:
        files = os.listdir(bert_config['log_dir'])
        ckpts = []
        for f in files:
            if f.endswith(".pth.tar"):
                ckpts.append(f)

        iters = [int(f.split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
        iters = sorted(iters)[-1]
        check_path = os.path.join(bert_config['log_dir'], "{}.pth.tar".format(iters))
    except:
        load = False
    
    if load:
        print(f"loading from {check_path} ...")
        checkpoint = torch.load(check_path)
        bert.load_state_dict(checkpoint['model'], strict=False)

    # get model
    num_class = preprocess_config["preprocessing"]["num_class"]
    model = ALVPredictor(
        bert,
        num_class=num_class,
        hidden_size=bert_config['model_params']['hidden_size']
    ).to(device)

    # get optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=train_config["optimizer"]["lr"])

    # load restore weight
    if args.restore_epoch:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_epoch),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    # freeze weight
    # for param in model.encoder.parameters():
    #     param.require_grad = False
    # print(model)
    # return

    # prepare loss func
    Loss = nn.CrossEntropyLoss().to(device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    epoch = args.restore_epoch + 1
    total_epoch = train_config["epoch"]["total_epoch"]
    save_epoch = train_config["epoch"]["save_epoch"]

    inner_bar = tqdm(total=total_epoch, desc="Epoch {}".format(epoch), position=1)
    while True:
        epoch_loss = 0
        batch_num = 0
        acc = 0
        num_phoneme = 0
        for batchs in loader:
            texts = batchs["text"].to(device)
            alvs = batchs["alv"].unsqueeze(-1).to(device)
            text_lens = batchs["text_lens"].to(device)

            max_src_len = len(texts[0])
            src_masks = get_mask_from_lengths(text_lens, max_src_len).unsqueeze(-1)

            # Forward
            outputs = model(texts, src_masks)
            src_masks = ~src_masks
            # Cal Loss
            loss = Loss(outputs.masked_select(src_masks).view(-1, num_class).float(), alvs.masked_select(src_masks).long())
            # Backward
            loss.backward()
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Cal Accuuracy
            with torch.no_grad():
                epoch_loss += loss.item()
                batch_num += 1
                preds = outputs.masked_select(src_masks).view(-1, num_class).argmax(dim=-1)
                acc += torch.sum(preds == alvs.masked_select(src_masks))
                num_phoneme += len(preds)

        epoch_loss = epoch_loss / batch_num
        accuracy = acc / num_phoneme

        if epoch % save_epoch == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(
                    train_config["path"]["ckpt_path"],
                    "{}.pth.tar".format(epoch),
                ),
            )

        # train loss
        message1 = "Epoch {}/{}, ".format(epoch, total_epoch)
        message2 = f"Train Loss: {epoch_loss}, Train Accuracy: {accuracy}"
        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
            f.write(message1 + message2 + "\n")
        inner_bar.write(message1 + message2)
        train_logger.add_scalar("Train Loss", epoch_loss, epoch)
        train_logger.add_scalar("Train Accuracy", accuracy, epoch)

        # validation loss
        model.eval()
        message = evaluate(model, epoch, configs, val_logger)
        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
            f.write(message + "\n")
        inner_bar.write(message)
        model.train()

        if epoch == total_epoch:
            quit()
        epoch += 1
        inner_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "-b", "--bert_config", type=str, default="Configs/config.yml", help="path to bert config yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    bert_config = yaml.load(open(args.bert_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config, bert_config)

    main(args, configs)

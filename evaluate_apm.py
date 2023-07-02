import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alvpredictor.dataset import Dataset
from alvpredictor.utils.tools import get_mask_from_lengths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, epoch, configs, logger=None):
    preprocess_config, model_config, train_config, bert_config = configs

    # Get dataset
    dataset = Dataset(
        "val", preprocess_config
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    num_class = preprocess_config["preprocessing"]["num_class"]

    # Get loss function
    Loss = nn.CrossEntropyLoss()

    # Evaluation
    epoch_loss = 0
    acc = 0
    num_phoneme = 0
    for batchs in loader:
        texts = batchs["text"].to(device)
        alvs = batchs["alv"].unsqueeze(-1).to(device)
        text_lens = batchs["text_lens"].to(device)

        max_src_len = len(texts[0])
        src_masks = get_mask_from_lengths(text_lens, max_src_len).unsqueeze(-1)
        with torch.no_grad():
            # Forward
            outputs = model(texts, src_masks)
            src_masks = ~src_masks
            # Cal Loss
            loss = Loss(outputs.masked_select(src_masks).view(-1, num_class).float(), alvs.masked_select(src_masks).long())
            epoch_loss += loss.item()

            preds = outputs.masked_select(src_masks).view(-1, num_class).argmax(dim=-1)
            acc += torch.sum(preds == alvs.masked_select(src_masks))
            num_phoneme += len(preds)


    epoch_loss = epoch_loss / len(dataset)
    accuracy = acc / num_phoneme

    message = f"Validation Epoch {epoch}, Val Loss: {epoch_loss}, Val Accuracy: {accuracy}"

    if logger is not None:
        logger.add_scalar("Val Loss", epoch_loss, epoch)
        logger.add_scalar("Val Accuracy", accuracy, epoch)

    return message

import os
import os.path as osp
import pickle
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import LoggerType
from datasets import load_from_disk
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from transformers import AlbertConfig, AlbertModel
from transformers import BertJapaneseTokenizer
import yaml

from dataloader import build_dataloader
from model import MultiTaskModel
from utils import length_to_mask, scan_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    curr_steps = 0
    
    dataset = load_from_disk(config["data_folder"])

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    log_for_tensorboard = 'logs'
    if not osp.exists(log_for_tensorboard): os.makedirs(log_for_tensorboard, exist_ok=True)
    train_logger = SummaryWriter(log_for_tensorboard)
    
    batch_size = config["batch_size"]
    train_loader = build_dataloader(dataset, 
                                    batch_size=batch_size, 
                                    num_workers=2,
                                    dataset_config=config['dataset_params'],
                                    device=device)

    albert_base_configuration = AlbertConfig(**config['model_params'])
    
    bert = AlbertModel(albert_base_configuration).to(device)
    bert = MultiTaskModel(bert, 
                          num_vocab=max([m['token'] for m in token_maps.values()]), 
                          num_tokens=config['model_params']['vocab_size'],
                          hidden_size=config['model_params']['hidden_size']).to(device)
    
    load = True
    try:
        files = os.listdir(log_dir)
        ckpts = []
        for f in os.listdir(log_dir):
            if f.startswith("step_"): ckpts.append(f)

        iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
        iters = sorted(iters)[-1]
    except:
        iters = 0
        load = False
    
    optimizer = AdamW(bert.parameters(), lr=1e-4)
    
    accelerator = Accelerator(mixed_precision=config['mixed_precision'], split_batches=True, kwargs_handlers=[ddp_kwargs])
    
    if load:
        checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
        state_dict = checkpoint['net']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        bert.load_state_dict(new_state_dict, strict=False)
        
        accelerator.print('Checkpoint loaded.')
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    bert, optimizer, train_loader = accelerator.prepare(
        bert, optimizer, train_loader
    )

    accelerator.print('Start training...')

    running_loss = 0
    
    for _, batch in enumerate(train_loader):
        curr_steps += 1
        
        words, labels, phonemes, input_lengths, masked_indices = batch
        words, labels, phonemes = words.to(device), labels.to(device), phonemes.to(device)
        text_mask = length_to_mask(torch.Tensor(input_lengths)).to(device)
        
        tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())
        
        loss_vocab = 0
        for _s2s_pred, _text_input, _text_length, _masked_indices in zip(words_pred, words, input_lengths, masked_indices):
            loss_vocab += criterion(_s2s_pred[:_text_length], 
                                        _text_input[:_text_length])
        loss_vocab /= words.size(0)
        
        loss_token = 0
        sizes = 0
        for _s2s_pred, _text_input, _text_length, _masked_indices in zip(tokens_pred, labels, input_lengths, masked_indices):
            if len(_masked_indices) > 0:
                _text_input = _text_input[:_text_length][_masked_indices]
                loss_tmp = criterion(_s2s_pred[:_text_length][_masked_indices], 
                                            _text_input[:_text_length]) 
                loss_token += loss_tmp
                sizes += 1
        loss_token /= sizes

        loss = loss_vocab + loss_token

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        running_loss += loss.item()

        iters = iters + 1
        if (iters+1)%log_interval == 0:
            accelerator.print ('Step [%d/%d], Loss: %.5f, Vocab Loss: %.5f, Token Loss: %.5f'
                    %(iters+1, num_steps, running_loss / log_interval, loss_vocab, loss_token))
            running_loss = 0
            train_logger.add_scalar("Total Loss", running_loss / log_interval, iters+1)
            train_logger.add_scalar("Vocab Loss", loss_vocab, iters+1)
            train_logger.add_scalar("Token Loss", loss_token, iters+1)
            
        if (iters+1)%save_interval == 0:
            accelerator.print('Saving..')

            state = {
                'net':  bert.state_dict(),
                'step': iters,
                'optimizer': optimizer.state_dict(),
            }

            accelerator.save(state, log_dir + '/step_' + str(iters + 1) + '.t7')

        if curr_steps > num_steps:
            return 


if __name__ == '__main__':
    config_path = "Configs/config.yml" # you can change it to anything else
    config = yaml.safe_load(open(config_path))

    with open(config['dataset_params']['token_maps'], 'rb') as handle:
        token_maps = pickle.load(handle)

    tokenizer = BertJapaneseTokenizer.from_pretrained(config['dataset_params']['tokenizer'])

    criterion = nn.CrossEntropyLoss().to(device) # F0 loss (regression)

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    loss_train_record = list([])
    loss_test_record = list([])

    num_steps = config['num_steps']
    log_interval = config['log_interval']
    save_interval = config['save_interval']

    train()
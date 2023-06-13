from concurrent.futures import TimeoutError
import os
from pebble import ProcessPool
import pickle

import datasets
from datasets import load_from_disk, concatenate_datasets
import pathlib
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer
import yaml

from simple_loader import FilePathDataset, build_dataloader
from phonemize import phonemize

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_shard(i):
    directory = root_directory + "/shard_" + str(i)
    if os.path.exists(directory):
        print("Shard %d already exists!" % i)
        return
    print('Processing shard %d ...' % i)
    shard = dataset.shard(num_shards=num_shards, index=i)
    processed_dataset = shard.map(lambda t: phonemize(t['text'], tokenizer), remove_columns=['text'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    processed_dataset.save_to_disk(directory)


if __name__ == '__main__':
    ##### config #####
    config_path = "Configs/config.yml" # you can change it to anything else
    config = yaml.safe_load(open(config_path))

    ##### set tokenizer #####
    tokenizer = BertJapaneseTokenizer.from_pretrained(config['dataset_params']['tokenizer'])

    ##### download dataset #####
    # comment out the following line in hogehoge/datasets/wikipedia/wikipedia.py
    # | "Distribute" >> beam.transforms.Reshuffle()
    datasets.config.DOWNLOADED_DATASETS_PATH = pathlib.Path("/media/yamauchi/2084DFA884DF7EAA/dataset/wikipedia-ja")
    dataset = datasets.load_dataset(
        'wikipedia', language="ja", date="20230601", beam_runner="DirectRunner",
        cache_dir="/media/yamauchi/2084DFA884DF7EAA/dataset/wikipedia-ja/.cache"
    )
    dataset = dataset['train']

    ##### make shards #####
    root_directory = "./wiki_phoneme"
    num_shards = 50000
    max_workers = 20 # change this to the number of CPU cores your machine has 
    with ProcessPool(max_workers=max_workers) as pool:
        pool.map(process_shard, range(num_shards), timeout=60)

    ##### correct shards #####
    output = [dI for dI in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory,dI))]
    datasets = []
    for o in output:
        directory = root_directory + "/" + o
        try:
            shard = load_from_disk(directory)
            datasets.append(shard)
            print("%s loaded" % o)
        except:
            continue
    dataset = concatenate_datasets(datasets)
    dataset.save_to_disk(config['data_folder'])
    print('Dataset saved to %s' % config['data_folder'])

    ##### Remove unneccessary tokens from the pre-trained tokenizer #####
    dataset = load_from_disk(config['data_folder'])
    file_data = FilePathDataset(dataset)
    loader = build_dataloader(file_data, num_workers=20, batch_size=128, device=device)

    special_token = config['dataset_params']['word_separator']

    unique_index = [special_token]
    for _, batch in enumerate(tqdm(loader)):
        unique_index.extend(batch["input_ids"])
        unique_index = list(set(unique_index))

    token_maps = {}
    for t in tqdm(unique_index):
        word = tokenizer.decode([t])
        token_maps[t] = {'word': word, 'token': unique_index.index(t)}

    with open(config['dataset_params']['token_maps'], 'wb') as handle:
        pickle.dump(token_maps, handle)
    print('Token mapper saved to %s' % config['dataset_params']['token_maps'])

    # print(token_maps)

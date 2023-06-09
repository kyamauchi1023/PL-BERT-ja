from concurrent.futures import TimeoutError
import os
from pebble import ProcessPool

import datasets
import pathlib
import phonemizer
from transformers import BertJapaneseTokenizer
import yaml

from phonemize import phonemize


def process_shard(i):
    directory = root_directory + "/shard_" + str(i)
    if os.path.exists(directory):
        print("Shard %d already exists!" % i)
        return
    print('Processing shard %d ...' % i)
    shard = dataset.shard(num_shards=num_shards, index=i)
    processed_dataset = shard.map(lambda t: phonemize(t['text'], global_phonemizer, tokenizer), remove_columns=['text'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    processed_dataset.save_to_disk(directory)


if __name__ == '__main__':
    # config
    config_path = "Configs/config.yml" # you can change it to anything else
    config = yaml.safe_load(open(config_path))

    # set tokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained(config['dataset_params']['tokenizer'])

    # download dataset
    datasets.config.DOWNLOADED_DATASETS_PATH = pathlib.Path("/media/yamauchi/2084DFA884DF7EAA/dataset/wikipedia-ja")
    dataset = datasets.load_dataset(
        'wikipedia', language="ja", date="20210120", beam_runner="DirectRunner",
        cache_dir="/media/yamauchi/2084DFA884DF7EAA/dataset/wikipedia-ja/.cache"
    )
    dataset = dataset['train']

    # # comment out Line 523 in datasets/datasets/wikipedia/wikipedia.py
    # # | "Distribute" >> beam.transforms.Reshuffle() 
    # datasets.config.DOWNLOADED_DATASETS_PATH = pathlib.Path("/media/yamauchi/2084DFA884DF7EAA/dataset/wikipedia-ja")
    # dataset = datasets.load_dataset(
    #     "./wikipedia.py", language="ja", date="20210120", beam_runner="DirectRunner",
    #     cache_dir="/media/yamauchi/2084DFA884DF7EAA/dataset/wikipedia-ja/.cache"
    # )
    # dataset = dataset['train']

    # preprocess
    root_directory = "./wiki_phoneme"
    num_shards = 50000
    max_workers = 24 # change this to the number of CPU cores your machine has 
    with ProcessPool(max_workers=max_workers) as pool:
        pool.map(process_shard, range(num_shards), timeout=60)

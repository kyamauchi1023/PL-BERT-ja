import string

import pyopenjtalk
import unicodedata

from convert_label import openjtalk2julius


def global_phonemize(text: str):
    phonemes = pyopenjtalk.g2p(text).split(' ')
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return phonemes


def phonemize(text, tokenizer):
    text = unicodedata.normalize("NFKC", text)
    words = tokenizer.tokenize(text)
    input_ids_ = tokenizer.convert_tokens_to_ids(words)
    
    phonemes = []
    input_ids = []
    for i in range(len(words)):
        word = words[i]
        input_id = input_ids_[i]
        phoneme = global_phonemize(word.replace('#', ''))
        if len(phoneme) != 0:
            phonemes.append(''.join(phoneme))
            input_ids.append(input_id)
        
    assert len(input_ids) == len(phonemes)
    return {'input_ids' : input_ids, 'phonemes': phonemes}
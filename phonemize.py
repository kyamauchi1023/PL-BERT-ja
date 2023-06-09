import pyopenjtalk
import unicodedata

from prepare_tg_accent import pp_symbols
from convert_label import openjtalk2julius


def global_phonemize(text: str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phonemes = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return phonemes


def phonemize(text, tokenizer):
    text = unicodedata.normalize("NFKC", text)
    words = tokenizer.tokenize(text)
    
    phonemes = [global_phonemize([word], strip=True)[0] if word not in string.punctuation else word for word in words]
    input_ids = [tokenizer.encode(word)[0] for word in words]
        
    assert len(input_ids) == len(phonemes)
    return {'input_ids' : input_ids, 'phonemes': phonemes}
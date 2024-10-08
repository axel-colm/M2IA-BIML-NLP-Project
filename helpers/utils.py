from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import one_hot
from typing import List, Tuple, Union, Iterable
from torch import LongTensor
from torchtext.vocab import Vocab
import torch

def load_file(file_path: str) -> Tuple[List]:

    texts = []
    emotions = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        text, emotion = line.strip().split(";")
        texts.append(text)
        emotions.append(emotion)

    return texts, emotions

def create_vocab(all_lines: List[str]) -> LongTensor:
    def yield_lines():
        for line in all_lines:
            yield line.split()     
    vocab = build_vocab_from_iterator(yield_lines(), specials=["unk"])
    vocab.set_default_index(vocab['unk'])
    return vocab

def get_one_hot(voc, keys: Union[str, Iterable], max_length=64) -> LongTensor:
    if isinstance(keys, str):
        keys = [keys]
    indices = voc(keys)
    if len(indices) > max_length:
        indices = indices[:max_length]
    if len(indices) < max_length:
        padding = [0] * (max_length - len(indices))
        indices = indices + padding
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    one_hot_tensor = one_hot(indices_tensor, num_classes=len(voc)) 
    return one_hot_tensor

import torch
import numpy as np
from numpy.typing import NDArray
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import one_hot
from typing import List, Tuple, Union, Iterable
from torch import LongTensor, Tensor
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, Dataset
from scipy.sparse._csr import csr_matrix

def load_file(file_path: str) -> Tuple[List[str], List[str]]:
    texts = []
    emotions = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        text, emotion = line.strip().split(";")
        texts.append(text)
        emotions.append(emotion)

    return texts, emotions

def create_vocab(sentences: List[str]) -> Vocab:
    def yield_lines():
        for sentence in sentences:
            yield sentence.split()

    vocab = build_vocab_from_iterator(yield_lines(), specials=["unk"])
    vocab.set_default_index(vocab['unk'])
    return vocab

def get_one_hot(vocab: Vocab, keys: Union[str, Iterable], max_length=64) -> LongTensor:
    if isinstance(keys, str):
        keys = [keys]

    indices = vocab(keys)
    if len(indices) > max_length:
        indices = indices[:max_length]
    if len(indices) < max_length:
        padding = [0] * (max_length - len(indices))
        indices = indices + padding

    indices_tensor = torch.tensor(indices, dtype=torch.long)
    one_hot_tensor = one_hot(indices_tensor, num_classes=len(vocab))
    return one_hot_tensor


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: NDArray):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch: List[Tuple[str, NDArray]], vocab: Vocab, max_length: int = 64) -> Tuple[Tensor, Tensor]:
    texts, labels = zip(*batch)
    one_hot_batch = torch.stack([get_one_hot(vocab, text.split(), max_length) for text in texts]).to(torch.float32)
    one_hot_batch = one_hot_batch.transpose(0, 1)
    labels_batch = torch.from_numpy(np.array(labels)).to(torch.long)
    return one_hot_batch, labels_batch


def create_loader(sentences: List[str], labels: NDArray,vocab: Vocab, batch_size: int, shuffle=False, max_length=64) -> DataLoader:
    dataset = TextDataset(sentences, labels)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda batch: collate_fn(batch, vocab, max_length)
    )
    return loader

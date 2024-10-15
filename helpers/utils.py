import torch
import numpy as np
from numpy.typing import NDArray
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import one_hot
from typing import List, Tuple, Union, Iterable
from torch import LongTensor, Tensor
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

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

def get_one_hot(vocab: Vocab, keys: Union[str, Iterable], max_length=24) -> LongTensor:
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

def collate_fn(batch: List[Tuple[str, NDArray]], vocab: Vocab, max_length: int = 24) -> Tuple[Tensor, Tensor]:
    texts, labels = zip(*batch)
    one_hot_batch = torch.stack([get_one_hot(vocab, text.split(), max_length) for text in texts]).to(torch.float32)
    one_hot_batch = one_hot_batch.permute(dims=(1,0,2))
    labels_batch = torch.from_numpy(np.array(labels)).to(torch.long)
    return one_hot_batch, labels_batch


def create_loader(sentences: List[str], labels: NDArray,vocab: Vocab, batch_size: int, shuffle=False, max_length=24) -> DataLoader:
    dataset = TextDataset(sentences, labels)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda batch: collate_fn(batch, vocab, max_length)
    )
    return loader

def calculate_mean_length(texts):
    total_length = sum(len(sentence.split()) for sentence in texts)
    mean_length = total_length / len(texts) if texts else 0
    return mean_length

def plot_train_statistics(history):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_accuracy = history['train_acc']
    val_accuracy = history['val_acc']
    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(epochs, train_loss, label='Training Loss', color='blue')
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='orange')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, train_accuracy, label='Training Accuracy', color='green')
    axes[1].plot(epochs, val_accuracy, label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()
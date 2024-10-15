import torch.backends
from helpers.utils import load_file, create_vocab, create_loader, calculate_mean_length
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rnn import RNN

if __name__ == "__main__":

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_texts, train_emotions = load_file('dataset/train.txt')
    val_texts, val_emotions = load_file('dataset/val.txt')
    test_texts, test_emotions = load_file('dataset/test.txt')
    le = LabelEncoder()
    train_emotions = le.fit_transform(train_emotions)
    val_emotions = le.transform(val_emotions)
    test_emotions = le.transform(test_emotions)

    # Create vocab
    all_texts = []
    all_texts.extend(train_texts)
    all_texts.extend(val_texts)
    vocab = create_vocab(all_texts)

    # Get mean sentence length for max_length argument
    mean_length = calculate_mean_length(all_texts)
    max_length = math.ceil(mean_length)

    # Create DataLoader
    batch_size = 96
    train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True, max_length=max_length)
    val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, max_length=max_length)
    test_loader = create_loader(test_texts, test_emotions, vocab, batch_size=1, max_length=max_length)

    # Initialsize model
    rnn = RNN(len(vocab),128,len(np.unique(train_emotions)))
    # history = rnn.fit(train_loader, val_loader, 10, lr=0.001)
    rnn.evaluate(test_loader)


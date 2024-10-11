import torch.backends
from helpers.utils import load_file, create_vocab, create_loader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rnn import RNN

if __name__ == "__main__":

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

    # Create DataLoader
    batch_size = 8
    train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True)
    val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, shuffle=False)
    test_loader = create_loader(test_texts, test_emotions, vocab, batch_size=1, shuffle=False)

    # Initialsize model
    rnn = RNN(len(vocab),512,len(np.unique(train_emotions)))
    history = rnn.fit(train_loader, val_loader, 10, lr=0.005)
    acc, loss = rnn.evaluate(test_loader)

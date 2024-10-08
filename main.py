from helpers.utils import load_file, create_vocab, get_one_hot
import numpy as np

if __name__ == "__main__":
    texts, emotions = load_file("dataset/train.txt")
    vocab = create_vocab(texts)
    encodings = []
    for text in texts:
        encodings.append(get_one_hot(vocab, text))
    print(np.array(encodings).shape)
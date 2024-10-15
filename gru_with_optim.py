from helpers.utils import load_file, create_vocab, create_loader, calculate_mean_length, plot_train_statistics
from helpers.optimisation import retrieve_class_weights, WeightedNLLLoss, get_word_distribution, plot_k_least_frequent, clean_text
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from custom_implemenations.gru import CustomGRU

if __name__ == "__main__":

    # Load the files
    train_texts, train_emotions = load_file('dataset/train.txt')
    val_texts, val_emotions = load_file('dataset/val.txt')
    test_texts, test_emotions = load_file('dataset/test.txt')

    # Encode labels
    le = LabelEncoder()
    train_emotions = le.fit_transform(train_emotions)
    val_emotions = le.transform(val_emotions)
    test_emotions = le.transform(test_emotions)

    # Compute class weights
    class_weights = retrieve_class_weights(train_emotions,np.unique(train_emotions))
    
    # Preprocess the texts 
    freq_dist = get_word_distribution(train_texts)
    plot_k_least_frequent(freq_dist, 20)

    rare_words = [word.lower() for word in freq_dist.keys() if freq_dist[word] == 1]
    train_texts = clean_text(train_texts, rare_words)
    val_texts = clean_text(val_texts, rare_words)
    test_texts = clean_text(test_texts, rare_words)

    # Create vocab
    vocab = create_vocab(train_texts)

    # Get mean sentence length for max_length argument
    mean_length = calculate_mean_length(train_texts)
    max_length = math.ceil(mean_length)

    # Create DataLoader
    batch_size = 96
    train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True, max_length=max_length)
    val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, max_length=max_length)
    test_loader = create_loader(test_texts, test_emotions, vocab, batch_size=1, max_length=max_length)

    # Train the model
    loss_func = WeightedNLLLoss(class_weights=class_weights)
    lstm = CustomGRU(len(vocab), 128, len(np.unique(train_emotions)))
    history = lstm.fit(train_loader, val_loader, 10, lr=0.001, loss_func=loss_func)
    plot_train_statistics(history)
    lstm.evaluate(test_loader, loss_func=loss_func)
    lstm.save("gru_with_optim.pth")
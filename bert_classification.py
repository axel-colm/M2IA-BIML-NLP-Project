from helpers.utils import load_file, plot_train_statistics
from helpers.optimisation import retrieve_class_weights, get_word_distribution, plot_k_least_frequent, clean_text
import numpy as np
from sklearn.preprocessing import LabelEncoder
from helpers.bert_utils import WeightedCrossEntropyLoss, TextClassificationDataset, DistilBERTClassifier, train, evaluate
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch


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
    rare_words = [word.lower() for word in freq_dist.keys() if freq_dist[word] == 1]
    train_texts = clean_text(train_texts, rare_words)
    val_texts = clean_text(val_texts, rare_words)
    test_texts = clean_text(test_texts, rare_words)

    # Loss function
    criterion = WeightedCrossEntropyLoss(class_weights)

    # Prepare the model
    bert_model_name = 'distilbert/distilbert-base-uncased'
    num_classes = len(np.unique(train_emotions))
    max_length = 24
    batch_size = 96
    num_epochs = 10
    learning_rate = 2e-5

    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_emotions, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_emotions, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_emotions, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = DistilBERTClassifier(bert_model_name, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    history = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs)
    plot_train_statistics(history)

    evaluate(model, test_dataloader, criterion)
    torch.save(model.state_dict(), "models/distilbert_classifier.pth")
    
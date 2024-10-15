import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download('wordnet')

def get_word_distribution(all_texts):
    all_texts = " ".join(text for text in all_texts)
    tokens = [word.lower() for word in all_texts.split()]
    freq_dist = nltk.FreqDist(tokens)
    return freq_dist

def plot_k_least_frequent(freq_dist, threshold):
    counts = np.array(list(freq_dist.values()))
    frequency_count_dict = {}
    for count in counts:
        if count in frequency_count_dict:
            frequency_count_dict[count] += 1
        else:
            frequency_count_dict[count] = 1
    filtered_dict = {k: v for k, v in frequency_count_dict.items() if k <= threshold}
    frequencies = list(filtered_dict.keys())
    word_counts = list(filtered_dict.values())
    plt.figure(figsize=(12, 6))
    plt.bar(frequencies, word_counts, color='skyblue', edgecolor='black')
    plt.title(f'Distribution de la fréquence des mots')
    plt.xlabel('Fréquence des mots')
    plt.ylabel('Nombre de mots')
    plt.xticks(frequencies)
    plt.grid(axis='y')
    plt.show()

def remove_rare_words(text, rare_words):
    after_rare_words = [word.lower() for word in text.split() if word not in rare_words]
    text_post_rare_words = " ".join(word for word in after_rare_words)
    return text_post_rare_words

def clean_text(texts, rare_words):
    lemmatizer = WordNetLemmatizer()
    cleaned_texts = []
    for text in texts:
        # Suppression de ponctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(f"[{string.punctuation}]", "", text)
        # Suppression des mots rares
        text = remove_rare_words(text, rare_words)
        # Suppression des stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in text.split() if word not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        text = " ".join(lemmatized_tokens)
        cleaned_texts.append(text)
    return cleaned_texts

def retrieve_class_weights(labels, unique_labels):
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights

class WeightedNLLLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedNLLLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, outputs, labels):
        ce_loss = nn.NLLLoss(weight=self.class_weights)(outputs, labels)
        return ce_loss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import torch

nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download('wordnet')

def get_word_distribution(all_texts):
    tfidf_vectorizer = TfidfVectorizer()
    all_texts = " ".join(text for text in all_texts)
    tokens = [word for word in all_texts.split()]
    freq_dist = nltk.FreqDist(tokens)
    return freq_dist

def remove_rare_words(text, rare_words):
    after_rare_words = [word for word in text.split() if word not in rare_words]
    text_post_rare_words = " ".join(word for word in after_rare_words)
    return text_post_rare_words

def clean_text(text, rare_words):
    # Suppression de ponctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    # Suppression des mots rares
    text = remove_rare_words(text, rare_words)
    # Suppression des stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    text = " ".join(token for token in tokens)
    return text

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
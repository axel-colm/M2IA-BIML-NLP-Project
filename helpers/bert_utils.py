import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    
class DistilBERTClassifier(torch.nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, outputs, labels):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)(outputs, labels)
        return ce_loss
    
import torch

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=3):
    previous_val_loss = float('inf')
    epochs_with_no_improvement = 0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_correct += (torch.argmax(outputs, 1) == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (torch.argmax(outputs, 1) == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")


        if previous_val_loss <= avg_val_loss:
            epochs_with_no_improvement += 1
        else:
            epochs_with_no_improvement = 0

        previous_val_loss = avg_val_loss

        if epochs_with_no_improvement == patience:
            break

    return history

def evaluate(model, test_loader, criterion):
    correct = []
    actual_labels = []
    total_loss = 0.0 
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct.extend(torch.argmax(outputs, 1).cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(actual_labels, correct)
    print(f"Accuracy Score: {accuracy:.4f} - Loss: {avg_loss:.4f}")
    print(f"Classification report:\n {classification_report(actual_labels, correct)}")
    ConfusionMatrixDisplay.from_predictions(actual_labels, correct)
    plt.show()
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = F.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def fit(self, train_loader, val_loader, epochs, lr, loss_func=nn.NLLLoss()):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            self.train()
            for x, t in train_loader:
                optimizer.zero_grad()
                hidden = self.initHidden(x.size(1))
                for i in range(x.size(0)):
                    output, hidden =self(x[i], hidden)
                loss = loss_func(output, t)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (torch.argmax(output, 1) == t).sum().item()
                train_total += t.size(0)
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total
            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, t in val_loader:
                    hidden = self.initHidden(x.size(1))
                    for i in range(x.size(0)):
                        output, hidden = self(x[i], hidden)
                    loss = loss_func(output, t)
                    val_loss += loss.item()
                    val_correct += (torch.argmax(output, 1) == t).sum().item()
                    val_total += t.size(0)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_correct / val_total
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_acc'].append(avg_val_acc)
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        return history

    def evaluate(self, test_loader, loss_func=nn.NLLLoss()):
        correct = []
        actual_labels = []
        
        self.eval()
        
        with torch.no_grad():
            total_loss = 0
            for x, t in test_loader:
                hidden = self.initHidden(x.size(1))
                for i in range(x.size(0)):
                    output, hidden = self(x[i], hidden)
                loss = loss_func(output, t)
                total_loss += loss.item()
                correct.extend(torch.argmax(output, 1).cpu().numpy())
                actual_labels.extend(t.cpu().numpy())
            avg_loss = total_loss / len(test_loader)
        print(f"Accuracy Score: {accuracy_score(actual_labels, correct)} - Loss: {avg_loss}")
        print(f"Classification report\n {classification_report(actual_labels, correct)}")
        ConfusionMatrixDisplay.from_predictions(actual_labels, correct)
        plt.show()


    def save(self, path):
        RNN.save_model(self, path)
        
    @staticmethod
    def save_model(model: 'RNN', path):
        path = os.path.join('models', path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
    
    @staticmethod
    def load(path, input_size, hidden_size, output_size) -> 'RNN':
        if not os.path.exists(path):
            path = os.path.join('models', path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
        model = RNN(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(path))
        return model
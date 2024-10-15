import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Parameters for GRU gates
        self.Wz = nn.Parameter(torch.Tensor(input_size, hidden_size))  # Update gate
        self.Uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))

        self.Wr = nn.Parameter(torch.Tensor(input_size, hidden_size))  # Reset gate
        self.Ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))

        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size))   # New gate
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

        # Classification layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """Assumes x is of shape (sequence_length, batch_size, feature)"""
        seq_len, bs, _ = x.size()
        h_t = torch.zeros(bs, self.hidden_size).to(x.device)

        for t in range(seq_len):
            x_t = x[t]  # Shape: (batch_size, input_size)

            # GRU gate calculations
            z_t = torch.sigmoid(x_t @ self.Wz + h_t @ self.Uz + self.bz)  # Update gate
            r_t = torch.sigmoid(x_t @ self.Wr + h_t @ self.Ur + self.br)  # Reset gate
            h_hat_t = torch.tanh(x_t @ self.W + (r_t * h_t) @ self.U + self.b)  # New gate

            # Update hidden state
            h_t = (1 - z_t) * h_t + z_t * h_hat_t  # Combining previous and new hidden state

        output = self.fc(h_t)  # Use the last hidden state for classification
        output = self.softmax(output)
        return output

    def fit(self, train_loader, val_loader, epochs, lr, loss_func=nn.NLLLoss(), patience=3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            self.train()
            for x, t in train_loader:
                optimizer.zero_grad()
                output = self(x)
                loss = loss_func(output, t)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (torch.argmax(output, 1) == t).sum().item()
                train_total += t.size(0)
                
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, t in val_loader:
                    output = self(x)
                    loss = loss_func(output, t)
                    val_loss += loss.item()
                    val_correct += (torch.argmax(output, 1) == t).sum().item()
                    val_total += t.size(0)
                    
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_correct / val_total
            
            # Logging history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_acc'].append(avg_val_acc)
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break
        
        return history

    def evaluate(self, test_loader, loss_func=nn.NLLLoss()):
        correct = []
        actual_labels = []
        
        self.eval()
        
        with torch.no_grad():
            total_loss = 0
            for x, t in test_loader:
                output = self(x)
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
        CustomGRU.save_model(self, path)

    def load(self, path):
        CustomGRU.load(path, self.input_size, self.hidden_size, self.output_size)
        
    @staticmethod
    def save_model(model: 'CustomGRU', path):
        path = os.path.join('models', path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
    
    @staticmethod
    def load(path, input_size, hidden_size, output_size) -> 'CustomGRU':
        if not os.path.exists(path):
            path = os.path.join('models', path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
        model = CustomGRU(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(path))
        return model
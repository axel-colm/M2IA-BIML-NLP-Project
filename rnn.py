import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input to hidden layer (i2h) and hidden to hidden layer (h2h)
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
        # Hidden to output layer (h2o)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        sequence_length = x.size(0)
        
        # Manually unroll the sequence
        for i in range(sequence_length):
            hidden = torch.tanh(self.i2h(x[i]) + self.h2h(hidden))
        
        # Output layer after the last hidden state
        output = self.h2o(hidden)
        output = F.log_softmax(output, dim=1)
        
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)  # Initialize hidden state as zeros

    def fit(self, train_loader, val_loader, epochs, lr):
        loss_func = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            self.train()
            for x, t in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                
                hidden = self.initHidden(batch_size=x.size(1))
                
                # Forward pass through the network
                y, hidden = self(x, hidden)
                
                # Compute loss
                loss = loss_func(y, t)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                # Accumulate the loss and accuracy
                train_loss += loss.item()
                train_correct += (torch.argmax(y, 1) == t).sum().item()
                train_total += t.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total
            
            # Validation loop
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, t in val_loader:
                    hidden = self.initHidden(batch_size=x.size(1))
                    y, hidden = self(x, hidden)
                    loss = loss_func(y, t)
                    val_loss += loss.item()
                    val_correct += (torch.argmax(y, 1) == t).sum().item()
                    val_total += t.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_correct / val_total
            
            # Record history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_acc'].append(avg_val_acc)
            
            # Print epoch stats
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        return history

    def evaluate(self, test_loader):
        loss_func = nn.NLLLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.eval()
        
        with torch.no_grad():
            for x, t in test_loader:
                hidden = self.initHidden(batch_size=x.size(1))
                y, hidden = self(x, hidden)
                loss = loss_func(y, t)
                total_loss += loss.item()
                correct += (torch.argmax(y, 1) == t).sum().item()
                total += t.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return accuracy, avg_loss

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
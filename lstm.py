import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.fc = nn.Linear(hidden_sz, hidden_sz)  # Add a fully connected layer
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        output = self.fc(hidden_seq)  # Apply the fully connected layer
        return output, (h_t, c_t)
    
    
    def fit(self, train_loader, val_loader, epochs, lr):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for idx, (x, t) in enumerate(train_loader):
                print(f"Batch {idx+1}/{len(train_loader)}", end="\r")
                hidden = (torch.zeros(x.size(0), self.hidden_size).to(x.device),
                            torch.zeros(x.size(0), self.hidden_size).to(x.device))
                optimizer.zero_grad()
                output, hidden = self(x, hidden)
                loss = loss_func(output[:, -1, :], t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            acc, val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")

    def evaluate(self, test_loader):
        loss_func = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.eval()
        
        with torch.no_grad():
            for x, t in test_loader:
                hidden = (torch.zeros(x.size(0), self.hidden_size).to(x.device),
                            torch.zeros(x.size(0), self.hidden_size).to(x.device))
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                output, hidden = self(x, hidden)
                loss = loss_func(output[:, -1, :], t)  # Use the last output for classification
                total_loss += loss.item()
                correct += (torch.argmax(output[:, -1, :], 1) == t).sum().item()
                total += t.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return accuracy, avg_loss

        
    
if __name__ == "__main__":
    from helpers.utils import load_file, create_vocab, create_loader, calculate_mean_length
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

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

    # Get mean sentence length for max_length argument
    mean_length = calculate_mean_length(all_texts)
    max_length = math.ceil(mean_length)

    # Create DataLoader
    batch_size = 96
    train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True, max_length=max_length)
    val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, max_length=max_length)
    test_loader = create_loader(test_texts, test_emotions, vocab, batch_size=1, max_length=max_length)

    
    lstm = LSTM(len(vocab), len(np.unique(train_emotions)))
    history = lstm.fit(train_loader, val_loader, 10, lr=0.005)
    acc, loss = lstm.evaluate(test_loader)
    
    
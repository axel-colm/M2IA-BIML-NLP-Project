import os
import torch.nn as nn
import torch.nn.functional as F
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Combined input and hidden
        hidden = F.relu(self.i2h(x) + self.h2h(hidden))
        # Output layer from hidden
        output = self.h2o(hidden)
        output = F.log_softmax(output, dim=1)
        # Return output and hidden state
        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
    
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


if __name__ == "__main__":
    model = RNN(10, 10, 10)
    model.save('test.pth')  # Save model to models/test.pth
    model = RNN.load('test.pth', 10, 10, 10)  # Load model from models/test.pth
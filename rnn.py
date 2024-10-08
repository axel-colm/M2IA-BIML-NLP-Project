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
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
        
    

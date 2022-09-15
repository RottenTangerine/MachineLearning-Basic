import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, output_size)
        self.nonlinear = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)

        hidden = self.input2hidden(combined)
        hidden = self.nonlinear(hidden)
        output = self.input2output(combined)
        output = self.nonlinear(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


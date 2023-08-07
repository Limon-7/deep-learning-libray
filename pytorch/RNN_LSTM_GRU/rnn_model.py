import torch
from utils import config
# create RNN layer:
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length,  num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first = True )
        self.fc = torch.nn.Linear(hidden_size*sequence_length, num_classes)
    
    def forward(self,x):
        ho = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.DEVICE)

        # forward prop
        out, _ =self.rnn(x, ho)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
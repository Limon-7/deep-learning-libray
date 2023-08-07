import torch
from utils import config
# create LSTM layer:
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length,  num_classes=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first = True )
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self,x):
        ho = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.DEVICE)
        co = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.DEVICE)


        # forward prop
        out, _ =self.lstm(x, (ho,co))
        out = self.fc(out[:, -1, :]) # all_batch, last_hidden_state, all_features; lossing information-> taking the most relevant information
        return out
import torch.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(CustomLSTM, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embed_dim1 = 128
        self.embed_dim2 = 64

        self.lstm1 = nn.LSTM(self.seq_len, self.embed_dim1)

        self.lstm2 = nn.LSTM(self.embed_dim1, self.embed_dim2)
        
        self.output_layer = nn.Linear(self.embed_dim2, self.pred_len)
        
    def forward(self, batch_x):
        x, _ = self.lstm1(batch_x.squeeze(-1))
        x = nn.functional.tanh(x)
        x, _ = self.lstm2(x)
        x = nn.functional.tanh(x)
        return self.output_layer(x).unsqueeze(-1)
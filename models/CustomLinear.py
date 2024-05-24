import torch.nn as nn


class CustomLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(CustomLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embed_dim = 1000

        self.input_layer = nn.Linear(self.seq_len, self.embed_dim)
        self.relu1 = nn.ReLU()
        self.output_layer = nn.Linear(self.embed_dim, self.pred_len)

    def forward(self, batch_x):
        print(batch_x.shape)
        x = self.relu1(self.input_layer(batch_x.squeeze(-1)))
        return self.output_layer(x).unsqueeze(-1)
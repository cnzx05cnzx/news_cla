import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, emb_len, emb_dim, out_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(emb_len, emb_dim)
        self.lstm = nn.LSTM(emb_dim, out_dim, batch_first=True, dropout=0.5, bidirectional=True, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(2 * out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        # 初始输入格式为(length, batch_size)
        out = self.embedding(x)
        # (length, batch_size, emb) -> (batch_size, length , emb )

        out = torch.transpose(out, 0, 1)

        out, (h, c) = self.lstm(out)
        out = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        out = self.linear(out)

        return out

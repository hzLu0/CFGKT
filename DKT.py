import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import one_hot


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, n_skills,device):
        super(DKT, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.n_skills = n_skills
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.qa_embedding = nn.Embedding(self.n_skills * 2 + 2, self.input_dim, padding_idx=self.n_skills*2+1)
        self.fc = nn.Linear(self.hidden_dim, self.n_skills)

    def forward(self, x, cshft):
        x = self.qa_embedding(x)
        out,hn = self.rnn(x)
        res = self.fc(out)
        x = torch.sigmoid(res)
        final = (x * one_hot(cshft.long(), self.n_skills+1)[:,:,:-1]).sum(-1)
        return final.squeeze(-1)
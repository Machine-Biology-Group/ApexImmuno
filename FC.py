import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable





class FC_model(nn.Module):
    def __init__(self, input_dim, h_dim):
        super().__init__()


        self.dim_emb = input_dim
        self.dim_h = 1024#h_dim
        self.dropout = 0.1


        self.fc2 = nn.Linear(input_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, int(h_dim / 2))
        self.fc4 = nn.Linear(int(h_dim / 2), int(h_dim / 4))
        self.fc5 = nn.Linear(int(h_dim / 4), 1)

        self.ln1 = nn.LayerNorm(h_dim)#nn.Dropout(0.2)
        self.ln2 = nn.LayerNorm(int(h_dim / 2))#nn.Dropout(0.2)
        self.ln3 = nn.LayerNorm(int(h_dim / 4))#nn.Dropout(0.2)

        self.dp1 = nn.Dropout(0.1)#nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.1)#nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.1)#nn.Dropout(0.2)

    def forward(self, x):


        out = self.dp1(F.leaky_relu((self.fc2(x))))
        out = self.dp2(F.leaky_relu((self.fc3(out))))
        out = self.dp3(F.leaky_relu((self.fc4(out))))
        out = self.fc5(out)

        return out


    def predict(self, x):

        return F.sigmoid(self.forward(x))

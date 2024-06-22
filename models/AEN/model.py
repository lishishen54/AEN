import torch.nn as nn
import torch
import math
from torch.autograd import Variable



class AEN(nn.Module):
    def __init__(self, d_model, outdim):
        super(AEN, self).__init__()
        self.d_model = d_model
        self.outdim = outdim
        self.q_linear = nn.Linear(self.d_model, outdim).cuda()

        self.k_linear = nn.Linear(self.d_model, outdim).cuda()
        self.norm_k = nn.LayerNorm(outdim).cuda()

        self.v_linear = nn.Linear(self.d_model, outdim).cuda()

        self.class_softmax = torch.nn.Softmax(dim=1)

    def forward(self, support_set, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]

        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        q_prototype = queries_vs


        if self.training:
            support_set_ks = self.norm_k(self.k_linear(support_set))
            queries_qs = self.norm_k(self.q_linear(queries))
            affinity_m = self.class_softmax(torch.matmul(queries_qs, torch.transpose(support_set_ks, -2, -1))/pow(self.outdim, 0.5))
            class_prototype = torch.matmul(affinity_m, support_set_vs)
            return q_prototype, class_prototype
        else:
            return q_prototype, support_set_vs

 
   


# thank you jojonki
# and https://arxiv.org/abs/1703.03130
import torch
from torch import nn
from torch.nn import functional as F

from utils import DataOperat

class StructuredSelfAttention(nn.Module):
    def __init__(self, hidden_size, n_classes, attn_hops, mlp_d, mlp_hidden, device='cpu'):
        super().__init__()
        initrange = 0.1
        self.Ws1 = torch.Tensor(1, mlp_d, 2*hidden_size).to(device)
        self.Ws2 = torch.Tensor(1, attn_hops, mlp_d).to(device)
        self.Ws1.uniform_(-initrange, initrange)
        self.Ws2.uniform_(-initrange, initrange)

        self.fc1 = nn.Linear(attn_hops * 2*hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, n_classes)

    def forward(self, hidden_sequence):
        '''
        :params hidden_sequence: (batch, sentence, 2*hidden)

        :return out: (batch, n_class)
        '''
        batch_size = hidden_sequence.size(0)
        sentence_len = hidden_sequence.size(1)
        hs_T = torch.transpose(hidden_sequence, 2, 1) # (batch, 2*hidden, sentence)

        # A(batch, mlp_d, sentence) = Ws1(batch, mlp_d, 2*hidden) * hs_T(batch, 2*hidden, sentence)
        A = torch.tanh(torch.bmm(self.Ws1.repeat(batch_size, 1, 1), hs_T))
        A = torch.bmm(self.Ws2.repeat(batch_size, 1, 1), A) # (batch, attn_hops, sentence)
        A = F.softmax(A.view(-1, sentence_len), dim=0).view(batch_size, -1, sentence_len) # batchNormalization
        M = torch.bmm(A, hidden_sequence) # (batch, attn_hops, 2*hidden)

        out = self.fc1(M.view(batch_size, -1)) # (batch, mlp_hidden)
        out = F.relu(out)
        out = self.fc2(out) # (batch, n_class)
        out = F.log_softmax(out, dim=1)
        return out, A

class ClassifyRNN(nn.Module):
    def __init__(
            self, 
            num_ids, 
            emb_size, 
            hidden_size, 
            n_classes, 
            padding_idx=0,
            attn_hops=32,
            mlp_d=384,
            mlp_hidden=512,
            device='cpu'
        ):
        '''
        :param num_ids: number of ids
        :param emb_size: free size?
        :param hidden_size: free size?
        :param padding_idx: int or None
        '''
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_ids, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.self_attention = StructuredSelfAttention(
            hidden_size=hidden_size, 
            n_classes=n_classes,
            attn_hops=attn_hops,
            mlp_d=mlp_d,
            mlp_hidden=mlp_hidden,
            device=device
            )
        
    def forward(self, input_data):
        '''
        :param input_data: (batch, sentence)

        :return output: (sentence, batch, 2*hidden)
        '''
        embedded = self.embedding(input_data) # (batch, sentence, emb)
        embedded = F.leaky_relu(embedded) # or relu or None
        hidden_sequence, _ = self.lstm(embedded) # (batch, sentence, 2*hidden)
        out, A = self.self_attention(hidden_sequence)
        return out, A

if __name__ == '__main__':
    # test code (usuary not use)
    num_ids = 16
    emb_size = 8
    hidden_size = 32
    n_classes = 4
    test_encoder = ClassifyRNN(num_ids, emb_size, hidden_size, n_classes)
    gen_batch = DataOperat.gen_batch_tensor('data/Arithmetic_pad_x.csv', 50)
    sample_data = next(gen_batch)
    print(sample_data)
    print(sample_data.size())
    out, A = test_encoder(sample_data)
    print(out)
    print(out.size())

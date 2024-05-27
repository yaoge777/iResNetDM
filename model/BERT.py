import torch
from torch import nn
import math
import torch_geometric.nn as pyg_nn


class InputEmbedding(nn.Module):
    
    def __init__(self, word_num, d_model):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(word_num, d_model)
        nn.init.normal_(self.emb.weight.data, mean=0.0, std=1.0)

    def forward(self,x):
        return self.emb(x) * math.sqrt(self.d_model)
  

class PositionEmbedding(nn.Module):
    
    def __init__(self, seq_len, d_model, device=torch.device('cuda')):
        super().__init__()
        self.pe = torch.ones(seq_len, d_model, device=device)
        d = torch.arange(d_model, dtype=torch.float, device=device)
        d = torch.pow(10000, d*2/math.sqrt(d_model))
        for i in range(seq_len):
            self.pe[i, 0::2] = torch.sin(i/d[0::2])
            self.pe[i, 1::2] = torch.cos(i/d[1::2])
    
    def forward(self, x):
        return x + self.pe


class K_mer_aggregate(nn.Module):

    def __init__(self,kmer, in_dim, dropout=0.1):
        super().__init__()

        self.layers = []

        self._add_layers(kmer, in_dim, dropout)
        self.d_model = in_dim

    def _add_layers(self, kmer, in_dim, dropout):

        for k in kmer:

            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=k, padding=(k-1)//2),
                    nn.BatchNorm1d(in_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        
        self.layers = nn.ModuleList(self.layers)

        self.conct = nn.Sequential(
            nn.Conv2d(in_channels=len(kmer), out_channels=1, kernel_size=1),
            nn.LayerNorm(in_dim)
        )

    def forward(self, x):
        output_list = []

        x = x.permute(0, 2, 1)

        for layer in self.layers:
            output = layer(x)
            output_list.append(output.permute(0, 2, 1).unsqueeze(dim=1))


        # for i, output in enumerate(output_list):
        #     if output.size(2) < max_height:
        #         output_list[i] = F.pad(output, (0, 0, 0, max_height - output.size(2)), 'constant', 0)

        # x = torch.cat(output_list, dim = 1).view(x.size(0), max_height, -1)
        x = torch.cat(output_list, dim=1)
        return x, self.conct(x).squeeze(1)
        # return output_list[0], output_list[1]







class ResNet(nn.Module):

    def __init__(self, kernel_size, d_model, dropout=0.1):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model * 2,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(d_model*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_model * 2, out_channels=d_model,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(d_model)
        )

    def forward(self, x):

        return x + self.seq(x.permute(0, 2, 1)).permute(0, 2, 1)


class ResNetBlock(nn.Module):

    def __init__(self, kmer, kernel_size, d_model, layers, mode='train_test'):
        super().__init__()

        self.encoder = K_mer_aggregate(kmer, d_model)
        # self.gcn = GraphConvNet(d_model*2, d_model * 4)
        self.resnet = nn.ModuleList([ResNet(kernel_size, d_model) for _ in range(layers)])
        self.activations = []
        self.mode = mode
        if mode == 'test':
            self.register_hooks()
    def forward_hook(self, module, input, output):
        self.activations.append(output)  # 使用类的成员变量来存储激活

    def register_hooks(self):
        for layer in self.resnet:
            layer.register_forward_hook(self.forward_hook)


    def forward(self, x):

        cts, emb = self.encoder(x)
        # x, input_data = self.gcn(x)
        # self.activations.append(x)
        if self.mode == 'test':
            self.activations.append(emb)
        x = emb
        for layer in self.resnet:
            x = layer(x)
        
        return cts, x, self.activations, emb
        

class MultiHeadAtten(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.head = args.heads_num
        self.dk = args.d_model // args.heads_num
        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)

    @staticmethod
    def attention(q, k, v, d_model):
        atten = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d_model)
        atten = atten.softmax(dim=-1)
        value = torch.matmul(atten, v)
        return atten, value
    
    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)        
        v = self.wv(x)
        q = q.view(batch, seq, self.head, self.dk).transpose(2,1)
        k = k.view(batch, seq, self.head, self.dk).transpose(2,1)
        v = v.view(batch, seq, self.head, self.dk).transpose(2,1)
        atten, value = MultiHeadAtten.attention(q,k,v, self.d_model)
        atten = atten.transpose(2, 1).contiguous().view(batch, self.head, seq, seq)
        value = value.transpose(2, 1).contiguous().view(batch, seq, self.d_model)
        return atten, self.wo(value)


class LSTM_FeedForward(nn.Module):

    def __init__(self, d_model, hidden, lstm_layers, dropout):
        super().__init__()
        dp = 0 if lstm_layers == 1 else dropout
        self.lstm = nn.LSTM(d_model, hidden, batch_first=True, bidirectional=True, num_layers=lstm_layers, dropout=dp)
        self.seq = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, d_model),
        ) 

    def forward(self, x):
        x = self.lstm(x)[0]
        return self.seq(x)

class Linear_FeedForward(nn.Module):
    
    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        d_model = d_model
        hidden = hidden
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model)
        )
    
    def forward(self, x):
        return self.seq(x)


class ResidualConnection(nn.Module):
    
    def __init__(self,d_model, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)
        
    def forward(self,x, sublayer):

        value = sublayer(self.norm(x))

        if isinstance(value, tuple):
            atten, v = value[0], value[1]
            return atten, x + self.dp(v)
        
        else:
            return x + self.dp(value)


class Block(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.attn = MultiHeadAtten(config)
        self.ff = None
        if config.ff == 'linear':
            self.ff = Linear_FeedForward(config.d_model, config.hidden, config.dropout)
        elif config.ff == 'lstm':
            self.ff = LSTM_FeedForward(config.d_model, config.hidden, config.lstm_layers, config.dropout)
        self.block = nn.ModuleList([ResidualConnection(config.d_model, config.dropout) for _ in range(2)])
    
    def forward(self, x):
        atten, x = self.block[0](x, self.attn)
        x = self.block[1](x, self.ff)
        return atten, x


class Classification_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*4, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cls = nn.Linear(input_dim, output_dim)

        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        x = self.norm(self.seq(x))

        return x, self.cls(x).softmax(dim=-1)


class Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.emb = InputEmbedding(config.word_num, config.d_model)
        # self.densenet = DenseNet(kmer=config.kmer, d_model=config.d_model,
        #                        growth_rate=config.growth_rate, block_num=config.densenet_block_num,
        #                        layer_num=config.densenet_layer_num, num_init_features=len(config.kmer),
        #                         kernel_size=config.kernel_size)
        # self.pb = PositionEmbedding(seq_len=42, d_model=config.d_model)
        self.resnet = ResNetBlock(kmer=config.kmer, kernel_size=config.kernel_size, d_model=config.d_model,
                                  layers=config.resnet_layer, mode=config.mode)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers_num)])
        self.norm = nn.LayerNorm(config.d_model)
        self.prj = Classification_layer(config.d_model, config.num_class)
        self.rep = config.rep

    def forward(self, x):


        x = self.emb(x)
        cts, x, activations, emb = self.resnet(x)

        emb_rep = emb[:, 0]
        resnet_rep = x[:, 0]
        for layer in self.layers:
            atn_matrix, x = layer(x)


        if self.rep == 'mean':
            rep = self.norm(torch.mean(x, dim=2))

        elif self.rep == 'max':
            rep = self.norm(torch.max(x, dim=2)[0])

        else:
            rep = self.norm(x[:, 0])

            if self.rep != 'default':
                print('the rep value is wrong')

        final_rep, x = self.prj(rep)
        return (x, activations, atn_matrix,
                [emb_rep.cpu().detach().numpy(),
                 resnet_rep.cpu().detach().numpy(),
                 rep.cpu().detach().numpy(),
                 final_rep.cpu().detach().numpy()], cts)


class Embedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb = InputEmbedding(config.word_num, config.d_model)

    def forward(self, x):
        return self.emb(x)

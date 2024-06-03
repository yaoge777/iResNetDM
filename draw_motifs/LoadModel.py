from Model import Encoder, Embedding
import torch

class Config:

    def __init__(self, word_num=5, d_model=256, kmer=[1, 3],
                 kernel_size=7, resnet_layer=5, layers_num=2, num_class=5,
                 dropout=0, hidden=256):
        self.word_num = word_num
        self.d_model = d_model
        self.kmer = kmer
        self.kernel_size = kernel_size
        self.resnet_layer = resnet_layer
        self.layers_num = layers_num
        self.num_class = num_class
        self.dropout = dropout
        self.heads_num = 8
        self.ff = 'linear'
        self.hidden = hidden
        self.rep = 'default'
        self.mode = 'train_test'
def load_model(config, param_path, device='cuda'):
    model = Encoder(config)
    emb_model = Embedding(config)
    pretrained_dict = torch.load(param_path, map_location=torch.device(device))
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)
    model.to(device)

    new_model_dict = emb_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    emb_model.load_state_dict(new_model_dict)
    emb_model.to(device)

    return model, emb_model

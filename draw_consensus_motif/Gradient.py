import numpy as np
import torch
from utils_cm import cal_consensus_motif
from LoadModel import Config, load_model


def collect_data(path, target):
    seq = []
    with open(path, 'r') as f:
        lines = f.readlines()[1:]

        for line in lines:
            l = line.split('\t')
            label = int(l[1]) - 1
            if label == target:
                seq.append((l[0]))
    return seq

def collect_from_text(path):

    seq = []
    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            seq.append(line.strip())
    return seq

def collect_from_fasta(path, max_sequence=3000):
    seq = []
    cur = 0
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('>') and line.strip():
                seq.append(line.strip())
                cur += 1
                if cur >= max_sequence:
                    break
    return seq
def tokenize(seqs, device='cuda'):

    dic = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
    data = []
    for seq in seqs:
        data.append([0])
        for s in seq:
            data[-1].append(dic[s])
    return torch.tensor(data, dtype=torch.long, device=device)


def calculate_outputs_and_gradients(inputs, model, index):
    # do the pre-processing
    predict_idx = None
    gradients = []
    n_steps = len(inputs)
    model.eval()
    for i in range(n_steps-1, -1, -1):
        input = inputs[i].unsqueeze(0)
        input.requires_grad = True
        input.retain_grad()
        output, _, _, _ = model(input)
        # clear grad
        model.zero_grad()
        if i == n_steps - 1 and output[0][index].item() < 0.25:
            return None
        output[0][index].backward(retain_graph=True)
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients

# integrated gradients
def integrated_gradients(inputs, model, emb_model, predict_and_gradients, baseline, index, steps=50, cuda=True):
    device = 'cuda' if cuda else 'cpu'
    # scale inputs and compute gradients
    data = emb_model.emb(inputs.unsqueeze(0))
    # baseline = emb_model.emb(torch.zeros_like(inputs.unsqueeze(0)))
    # baseline = baseline.detach()
    data = data.detach()
    if baseline is None:
        baseline = torch.zeros_like(data, dtype=torch.float, device=device)
    scale_factors = torch.arange(0, steps + 1, dtype=torch.float32, device=device) / steps

    # 将 scale factors 的形状从 [steps+1] 调整为 [steps+1, 1, 1] 以匹配 data 和 baseline 的形状
    scale_factors = scale_factors.unsqueeze(-1).unsqueeze(-1)
    scaled_inputs = baseline + scale_factors * (data - baseline)
    grads = predict_and_gradients(scaled_inputs, model, index)
    if grads is None:
        return None
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.expand_dims(avg_grads, axis=0)
    data = data.cpu().numpy()
    baseline = baseline.cpu().numpy()
    integrated_grad = (data - baseline) * avg_grads
    return integrated_grad


class ModelWithSeparateEmbedding(torch.nn.Module):
    def __init__(self, original_model):
        super(ModelWithSeparateEmbedding, self).__init__()
        # 将原始模型的embedding层作为一个独立的子模块
        self.emb = original_model.emb

        # 创建一个新的Sequential模块，包含原始模型中除了embedding层之外的所有层
        self.rest_of_model = torch.nn.Sequential(
            *list(original_model.children())[1:]  # 假设embedding是第一个层
        )

    def forward(self, x):

        # 然后通过模型的其余部分
        x = self.rest_of_model(x)
        return x


def calculate_ig(data_path, config, param_path, target, load_path=None):
    if load_path is not None:

        ig_scores = np.loadtxt(load_path[0], delimiter=',')

        valid_sequence = np.load(load_path[1])

        return ig_scores, valid_sequence

    model, emb_model = load_model(config, param_path)
    seqs = np.array(collect_from_fasta(data_path))
    tokenized_seqs = tokenize(seqs)
    ig_scores = []
    valid_sequence = []
    for i, seq in enumerate(tokenized_seqs):
        integrated_grad = integrated_gradients(seq, model, emb_model, calculate_outputs_and_gradients,
                                               index=target, baseline=None)
        if integrated_grad is None:
            continue
        ig_scores.append(np.sum(integrated_grad, axis=-1)[0])
        valid_sequence.append(seqs[i])
        print(f'{i}: {ig_scores[-1]}')
    return ig_scores, np.array(valid_sequence)

def find_motif(folder, valid_sequence, ig_scores, s_i, e_i, save):
    if save:
        np.savetxt(f'{folder}/ig_score_{folder}.txt', ig_scores, delimiter=',')
        np.save(f'{folder}/valid_sequence.npy', valid_sequence)
    motif, score = cal_consensus_motif(valid_sequence, np.array(ig_scores)[:, s_i:e_i], s_i, e_i)
    for i, v in enumerate(motif):
        np.savetxt(f'{folder}/{i}_motif.txt', motif[i], delimiter=',')
    np.savetxt(f'{folder}/score.txt', score, delimiter=',')


data_path = '../data/DNA_MS/cdhit_cleaned/split_fasta/6mA/C.equisetifolia/6mA_C.equisetifolia_combined_pos.fasta'
target = 2
param_path ='Resnet_atten_5_2_256 _FL_newset[1, 3]merBERT, ACC[0.756].pt'
config = Config()
folder_path = '4mC_CE'
load_path = ['4mC_CE/ig_score_4mC_CE.txt', '4mC_CE/valid_sequence.npy']
# load_path = None
save = True if load_path is None else False
s_i = 16
e_i = 27

ig_scores, valid_sequence = calculate_ig(data_path=data_path, config=config, param_path=param_path, target=target,
                                            load_path=load_path)

find_motif(folder=folder_path, valid_sequence=valid_sequence, ig_scores=ig_scores, s_i=s_i, e_i=e_i, save=save)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Constrastive_loss(nn.Module):
    
    def __init__(self, temperature, d_model, kmer_num):
      super().__init__()
      self.temperature = temperature
      self.kmer_num = kmer_num
      self.prj = nn.Sequential(
         nn.AdaptiveAvgPool2d((1, d_model)),
         nn.ReLU(),
         nn.Linear(d_model, d_model)
        )
    def forward(self, rep):

      #  rep: [B, 2, d]
      rep = self.prj(rep).squeeze(dim = 2)
      B, _, d = rep.shape
      rep = rep.view(self.kmer_num * B, d)


      # norm, then calculate the cosine distance for any pairs
      rep = F.normalize(rep, dim=1)
      sim_matrix = torch.mm(rep, rep.t()) / self.temperature

      # reduce the max for stability
      sim_matrix_max, _ = torch.max(sim_matrix, dim = 1, keepdim = True)
      sim_matrix = sim_matrix - sim_matrix_max.detach()


      # add mask for self pair to be -inf
      mask = torch.eye(self.kmer_num * B, device = rep.device).bool()
      sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))


      exp_sim_matrix = sim_matrix.exp()
      logits = exp_sim_matrix / exp_sim_matrix.sum(dim=1, keepdim=True)

      labels = torch.arange(1, self.kmer_num * B, self.kmer_num, device=rep.device).view(B, 1)
      labels = torch.cat((labels, labels-1), dim = 1).view(self.kmer_num * B)
      
      # Compute the contrastive loss
      positive_logits = logits[range(self.kmer_num * B), labels]

      loss = -torch.log(positive_logits + 1e-9).mean()

      return loss
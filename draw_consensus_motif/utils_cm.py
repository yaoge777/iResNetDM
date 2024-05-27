import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# def read_file(path, target):
#     seq = []
#     with open(path, 'r') as f:
#         lines = f.readlines()[1:]
#
#         for line in lines:
#             l = line.split('\t')
#             label = int(l[1]) - 1
#             if label == target:
#                 seq.append(l[0][15:31])
#     return seq

def to_onehot(seq):
    length = len(seq)
    out = np.zeros((4,length))
    for i in range(length):
        if seq[i] == "A":
            out[0,i] = 1
        elif seq[i] == 'C':
            out[1,i] = 1
        elif seq[i] == 'G':
            out[2,i] = 1
        elif seq[i] == 'T':
            out[3,i] = 1

    return out

def reduction_clustering(df,eps=0.3):
    # fixed_random_seed = 42
    # Y = PCA(n_components=3).fit_transform(df)
    Y = umap.UMAP(n_components=2).fit_transform(df)
    # clustering = SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",random_state=0).fit(Y)

    clustering = DBSCAN(eps=eps, min_samples=3).fit(Y)

    cluster_labels = clustering.labels_
    return cluster_labels

def pfm(seqs):
    """
    Inputs:
           seqs: list of seqs
    Outputs:
           A (4, x) matrix where x is the length of input seqs

           A
           C
           G
           T/U

    """
    length = len(seqs[0])
    out = np.zeros((4, length))
    for seq in seqs:
        #print(seq)
        for i in range(length):
            if seq[i] == 'A':
                out[0, i] += 1
            elif seq[i] == 'C':
                out[1, i] += 1
            elif seq[i] == 'G':
                out[2, i] += 1
            elif seq[i] == 'T':
                out[3, i] += 1

    return out

def pwm(pfm):

    totals = []
    for i in range(pfm.shape[1]):
        totals.append(np.sum(pfm[:, i]))

    total = max(totals)
    p = (pfm + (np.sqrt(total) * 1/4)) / (total + (4 * (np.sqrt(total) * 1/4)))
    return p

def cal_consensus_motif(seqs, scores, s_i, e_i, eps=0.19):
    """
    Computing motifs of aligned sequences
    Input:
         Seqs: aligned sequenced
         scores: aggregated ig score over such a short aligned sequence
         eps: parameters for DBSCAN
    """
    data = []
    for i in range(len(seqs)):

        seqs[i] = seqs[i][s_i-1:e_i-1]
        tmp = to_onehot(seqs[i]).T.flatten()
        data.append(tmp)

    df = pd.DataFrame(data=data, index=list(range(len(seqs))))

    class_labels = reduction_clustering(df, eps=eps)
    print(class_labels)
    seqs_dict = {}
    scores_dict = {}
    for i in np.unique(class_labels):
        if i != -1:
            class_seqs = seqs[class_labels == i]
            class_scores = scores[class_labels == i]
            avg_scores = np.sum(class_scores) / len(class_seqs)
            seqs_dict[i] = list(class_seqs)
            scores_dict[i] = avg_scores
            print('class:%d score:%.5f'%(i, avg_scores))
            # print(class_seqs)

    # sort the class by ig score
    index = sorted(scores_dict, key=scores_dict.__getitem__, reverse=True)

    pwm_weights = []
    ig_score = []
    for idx in index:
        pwm_weights.append(np.expand_dims(pwm(pfm(seqs_dict[idx])), axis=0))
        ig_score.append(scores_dict[idx])

    consensus_motif = np.concatenate(pwm_weights, axis=0)

    return consensus_motif, np.array(ig_score)

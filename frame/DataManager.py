
import torch
from torch.utils.data import Dataset, DataLoader

from utils import utils_file, utils_seq_finder_by_motif

class DNADataset(Dataset):
    
    def __init__(self,data,label):
        
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class DataManager():
    
    def __init__(self, learner):
        
        self.iomanager = learner.ioManager
        self.visualizer = learner.visualizer
        self.config = learner.config
        self.mode = self.config.mode
        
        self.train_label = None
        self.test_label = None
        
        self.train_dataset = None
        self.test_dataset = None
        
        self.train_dataloader = None
        self.test_dataloader= None
        
    def load_traindata(self):
        self.train_dataset, self.train_label = utils_file.load_from_tsv_data(self.config.path_train_data)
        self.train_dataloader = self.construct_dataloader(self.train_dataset,
                                                          self.train_label, self.config.cuda, self.config.batch_size)
    
    def load_testdata(self):
        if self.mode == 'test':
            self.test_dataset, self.test_label = utils_seq_finder_by_motif.find_seq_by_motif(self.config.path_test_data,
                                                self.config.label, self.config.motif, self.config.s, self.config.e,
                                                self.config.mask_inside, self.config.mask_num)
        else:
            self.test_dataset, self.test_label = utils_file.load_from_tsv_data(self.config.path_test_data)
        self.test_dataloader = self.construct_dataloader(self.test_dataset,
                                                         self.test_label, self.config.cuda, self.config.batch_size)
        
    def construct_dataloader(self, sequence, labels, cuda, batch_size):
        if cuda:
            device = torch.device('cuda')
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            sequence = torch.tensor(sequence, dtype=torch.long, device=device)
        else:
            labels = torch.tensor(labels, dtype=torch.long, device='cpu')
            sequence = torch.tensor(sequence, dtype=torch.long, device='cpu')
        dataset = DNADataset(sequence, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        print('length of data_loader: ', len(data_loader))
        return data_loader
    
    def get_dataloader(self, name):
        if name == 'train_set':
            return self.train_dataloader
        elif name == 'test_set':
            return self.test_dataloader
        else:
            raise ValueError('please input either train_set or test_set')
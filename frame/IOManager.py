import pickle
import os
import time
import torch

class IOManager():
    
    def __init__(self,learner):
        
        self.config = learner.config
        self.result_path = None
        self.log = None
    def initialize(self):
        self.result_path = self.config.path_save + '/' + self.config.learn_name + str(self.config.kmer) + 'mer'
        
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
#         save a pkl file of config 
        with open(self.result_path+'/config.pkl', 'wb') as f:
            pickle.dump(self.config, f)
        
        self.log = LOG(self.result_path)
            
#         save a txt file of config
        with open(self.result_path + '/config.txt', 'w') as f:
            for key, value in self.config.__dict__.items():
                k_v_pair = f'{key} : {value}'
                f.write(k_v_pair + '\r\n')
    
    def save_model_dict(self, model_dict, save_prefix, metric_name, metric_value):
        file_name = f'{save_prefix}, {metric_name}[{metric_value:.3f}].pt'
        save_path_pt = os.path.join(self.result_path+file_name)
        torch.save(model_dict, save_path_pt)


class LOG():
    
    def __init__(self, root_path):
        log_path = root_path + '/log'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log = open(log_path + '/%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
        log_files = os.listdir(log_path)
        log_files.sort()
        if len(log_files) > 200:
            print('The amount of files exceeds 200, an old file will be deleted', log_files.pop(), file=self.log)
    
    def Info(self, *data):
        msg = time.strftime("%Y_%m_%d_%I_%M_%S") + 'INFO: '
        for i in data:
            if type(i) == int:
                msg = msg + str(i)
                continue
            msg = msg + i
        print(msg)
        print(msg, file=self.log)
    
    def Warn(self, *data):
        msg = time.strftime("%Y_%m_%d_%I_%M_%S") + 'INFO: '
        for i in data:
            if type(i) == int:
                msg = msg + str(i)
                continue
            msg = msg + i
        print(msg)
        print(msg, file=self.log)
        
    def Error(self, *data):
        msg = time.strftime("%Y_%m_%d_%I_%M_%S") + 'INFO: '
        for i in data:
            if type(i) == int:
                msg = msg + str(i)
                continue
            msg = msg + i
        print(msg)
        print(msg, file=self.log)
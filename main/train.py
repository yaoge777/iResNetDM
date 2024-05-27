import sys
import pickle
import os
import numpy as np
import torch
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config_init
from frame import Learner
import argparse

def SL_train(config):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    # roc_datas, prc_datas = [], []

    # 将前置步骤走完
    # 进行初始化
    learner = Learner.Learner(config)
    learner.init_io()
    learner.init_vis()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    learner.load_params()
    learner.init_optimizer()
    learner.init_loss_fn()
    learner.train_model()


def SL_fintune():
    # config = config_SL.get_config()
    config = pickle.load(open('../result/resnet_5_2_256_FL[1, 3]mer/config.pkl', 'rb'))
    config.path_params = '../result/resnet_5_2_256_FL[1, 3]mer/resnet_5_2_256_FL[1, 3]merBERT, ACC[0.862].pt'
    config.learn_name = 'finetune_5_2_256_FL'
    config.model_save_name = 'finetune_5_2_256_FL'
    config.path_train_data, config.path_test_data = select_dataset()
    config.save_best = True
    config.threshold = 0.85
    config.mode = 'train_test'
    config.epoch = 20
    config.lr = 0.00005
    learner = Learner.Learner(config)
    learner.init_io()
    learner.init_vis()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.init_loss_fn()
    learner.train_model()

def draw_resnet(test_data_path):
    config = pickle.load(open('../result/Resnet_atten_5_2_256_FL_newset[1, 3]mer/config.pkl', 'rb'))
    config.path_params = '../result/Resnet_atten_5_2_256_FL_newset[1, 3]mer/Resnet_atten_5_2_256 _FL_newset[1, 3]merBERT, ACC[0.756].pt'
    config.mode = 'test'
    config.path_test_data = test_data_path
    config.label = 2
    config.motif = 'CGAGAAAA'
    config.s = 18
    config.e = 25
    config.mask_inside = False
    config.mask_num = 1
    config.learn_name = '6mA_DM_'+config.motif
    learner = Learner.Learner(config)
    learner.init_io()
    learner.init_vis()
    learner.load_test_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.init_loss_fn()
    acc = learner.test_model()
    return acc

def select_dataset():
    # DNA-MS
    path_train_data = '../data/train_set.tsv'
    path_test_data = '../data/test_set.tsv'

    print("train" + path_train_data, "test" + path_test_data)
    return path_train_data, path_test_data


if __name__ == '__main__':
    config = config_init.get_config()
    config.path_train_data, config.path_test_data = select_dataset()
    print('--------------训练集为{}------------'.format(config.path_train_data))
    print('--------------测试集为{}------------'.format(config.path_test_data))

    # total_acc = []
    # for i in range(10):
    #     acc = draw_resnet('D:\project\DNApred_ResNet\data\DNA_MS\cdhit_cleaned\split_fasta\\6mA\D.melanogaster\\6mA_D.melanogaster_combined_pos.fasta')
    #     total_acc.append(acc)
    # mean_accuracy = np.mean(total_acc)
    # std_accuracy = np.std(total_acc)
    # print(f'mean: {mean_accuracy}, std: {std_accuracy}')
    for i in range(1):
        SL_train(config)
    # SL_fintune()
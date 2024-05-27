import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE

class Visualizer():
    
    def __init__(self, learner):
        self.iomanager = learner.ioManager
        self.config = learner.config
        
    def initialize(self):
        self.step_log_interval = []
        self.train_metric_record = []
        self.train_loss_record = []
        self.step_valid_interval = []
        self.valid_metric_record = []
        self.valid_loss_record = []
        self.step_test_interval = []
        self.test_metric_record = []
        self.test_loss_record = []
        self.roc_data = None
        self.prc_data = None
        self.class_metric_record = []
        self.repres_list = None
        self.label_list = None
        self.atten = None
        self.class_rep = None
        self.activations = None
        self.resnet = None
    def draw_train_test_curve(self):
        # print('total training step: ',self.step_log_interval)
        self.class_metric_record = np.array(self.class_metric_record)
        sns.set(style='darkgrid')
        plt.figure(44, figsize=(44,32))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        
        plt.subplot(2,2,1)
        plt.plot(self.step_log_interval, self.train_metric_record)
        plt.title('train acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        plt.subplot(2,2,2)
        plt.plot(self.step_log_interval, self.train_loss_record)
        plt.title('train loss curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        
        plt.subplot(2,2,3)
        plt.plot(self.step_test_interval, self.test_metric_record)
        plt.title('test acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        plt.subplot(2,2,4)
        plt.plot(self.step_test_interval, self.test_loss_record)
        plt.title('test loss curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        
        path = os.path.join(self.iomanager.result_path, f'{self.config.learn_name}_total_'
                                                        f'{str(self.config.kmer)}mer.{self.config.save_figure_type}')
        plt.savefig(path)



    def draw_seperate_curve(self):

        plt.figure(22, figsize=(66,44))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        
        plt.subplot(3,3,1)
        plt.plot(self.step_test_interval, self.class_metric_record[:,0])
        plt.title('4mC acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        plt.subplot(3,3,2)
        plt.plot(self.step_test_interval, self.class_metric_record[:,1])
        plt.title('5hmC acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        plt.subplot(3,3,3)
        plt.plot(self.step_test_interval, self.class_metric_record[:,2])
        plt.title('6mA acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        plt.subplot(3,3,4)
        plt.plot(self.step_test_interval, self.class_metric_record[:,3])
        plt.title('5mC acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        plt.subplot(3,3,5)
        plt.plot(self.step_test_interval, self.class_metric_record[:,4])
        plt.title('neg acc curve', fontsize=23)
        plt.xlabel('step', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        
        path = os.path.join(self.iomanager.result_path, f'{self.config.learn_name}_seperate_'
                                                        f'{str(self.config.kmer)}mer.{self.config.save_figure_type}')
        plt.savefig(path)



    
    def draw_ROC_PRC_curve(self):
        # fpr, tpr:{0:[], 1:[], 2:[], 3[], macro:[]}, 
        # roc_auc:{0:float, macro:int}, 
        # pre,rcl:{0:[], 1:[], 2:[], 3[], macro:[]}
        # roc_data = [fpr, tpr, roc]
        # prc_data = [pre, rcl]
        sns.set(style='darkgrid')
        plt.figure(figsize=(16,8))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        lw=2
        
        plt.subplot(1,2,1)
        plt.plot(self.step_log_interval, self.train_metric_record)
        # plt.plot(self.roc_data[0]['macro'],self.roc_data[1]['macro'], color='darkorange',
        #         lw=lw, label='macro ROC curve (area=%0.2f)' % self.roc_data[2]['macro'])
        plt.plot(self.roc_data[0][0],self.roc_data[1][0], color=[252/255, 170/255, 103/255],
                lw=lw, label='4mC ROC curve (area=%0.2f)' % self.roc_data[2][0])
        plt.plot(self.roc_data[0][1],self.roc_data[1][1], color=[255/255, 255/255, 199/255],
                lw=lw, label='5hmC ROC curve (area=%0.2f)' % self.roc_data[2][1])
        plt.plot(self.roc_data[0][2],self.roc_data[1][2], color=[84/255, 134/255, 135/255],
                lw=lw, label='6mA ROC curve (area=%0.2f)' % self.roc_data[2][2])
        plt.plot(self.roc_data[0][3],self.roc_data[1][3], color=[71/255, 51/255, 53/255],
                lw=lw, label='5mC ROC curve (area=%0.2f)' % self.roc_data[2][3])
        plt.plot(self.roc_data[0][4],self.roc_data[1][4], color=[189/255, 30/255, 30/255],
                lw=lw, label='neg ROC curve (area=%0.2f)' % self.roc_data[2][4])
        plt.plot([0,1], [1,0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('false positive rate', fontdict={'weight':'normal', 'size':20})
        plt.ylabel('true positive rate', fontdict={'weight':'normal', 'size':20})
        plt.title('receiver operating characteristic curve', fontdict={'weight':'normal', 'size': 23})
        plt.legend(loc='lower right', prop={'weight': 'normal', 'size': 18})
    
        plt.subplot(1,2,2)
        # plt.plot(self.prc_data[1]['macro'],self.prc_data[0]['macro'], color='darkorange',
        #         lw=lw, label='macro PR curve (area=%0.2f)' % self.prc_data[2]['macro'])
        plt.plot(self.prc_data[1][0],self.prc_data[0][0], color=[252/255, 170/255, 103/255],
                lw=lw, label='4mC PR curve (area=%0.2f)' % self.prc_data[2][0])
        plt.plot(self.prc_data[1][1],self.prc_data[0][1], color=[255/255, 255/255, 199/255],
                lw=lw, label='5mhC PR curve (area=%0.2f)' % self.prc_data[2][1])
        plt.plot(self.prc_data[1][2],self.prc_data[0][2], color=[84/255, 134/255, 135/255],
                lw=lw, label='6mA PR curve (area=%0.2f)' % self.prc_data[2][2])
        plt.plot(self.prc_data[1][3],self.prc_data[0][3], color=[71/255, 51/255, 53/255],
                lw=lw, label='5mC PR curve (area=%0.2f)' % self.prc_data[2][3])
        plt.plot(self.prc_data[1][4],self.prc_data[0][4], color=[189/255, 30/255, 30/255],
                lw=lw, label='neg ROC curve (area=%0.2f)' % self.prc_data[2][4])
        # plt.fill_between(self.prc_data[0], self.prc_data[1], step='post', alpha=0.2, color='b')
        plt.plot([0,1], [1,0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('recall', fontdict={'weight':'normal', 'size':20})
        plt.ylabel('precision', fontdict={'weight':'normal', 'size':20})
        plt.title('precision recall curve', fontdict={'weight':'normal', 'size': 23})
        plt.legend(loc='lower left', prop={'weight': 'normal', 'size': 18})
        
        path = os.path.join(self.iomanager.result_path, f'{self.config.learn_name}_ROC_PRC_CURVE_'
                                                        f'{str(self.config.kmer)}mer.{self.config.save_figure_type}')
        plt.savefig(path)

    def draw_atten(self):

        if self.atten is None:
            return
        plt.figure(23, figsize=(88, 44))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        normalized_data = np.zeros_like(self.atten)

        for i in range(normalized_data.shape[0]):
            max_val = np.max(self.atten[i])
            min_val = np.min(self.atten[i])
            normalized_data[i] = (self.atten[i] - min_val) / (max_val - min_val)

        for i in range(len(normalized_data)):
            plt.subplot(2,4,i+1)
            sns.heatmap(normalized_data[i], cmap='YlGnBu', annot=False)

        path = os.path.join(self.iomanager.result_path,
                            f'{self.config.learn_name}_atten_map_{str(self.config.kmer)}mer.'
                            f'{self.config.save_figure_type}')
        plt.savefig(path)
        plt.show()

    def draw_partial_atten(self):
        if self.atten is None:
            return
        plt.figure(230, figsize=(44, 2))
        average_data = np.mean(self.atten, axis=0)
        max_val = np.max(average_data)
        min_val = np.min(average_data)
        average_data = (average_data - min_val) / (max_val - min_val)
        sns.heatmap(average_data[0].reshape(1,-1), cmap='YlGnBu', annot=False)
        path = os.path.join(self.iomanager.result_path,
                            f'{self.config.learn_name}_first_line_atten_map_{str(self.config.kmer)}mer.'
                            f'{self.config.save_figure_type}')
        plt.savefig(path)
        plt.show()

        plt.figure(232, figsize=(44, 44))
        average_data = np.mean(self.atten, axis=0)
        max_val = np.max(average_data)
        min_val = np.min(average_data)
        average_data = (average_data - min_val) / (max_val - min_val)
        sns.heatmap(average_data, cmap='YlGnBu', annot=False)
        path = os.path.join(self.iomanager.result_path,
                            f'{self.config.learn_name}_avg_atten_map_{str(self.config.kmer)}mer.'
                            f'{self.config.save_figure_type}')
        plt.savefig(path)
        plt.show()

    def draw_tsne(self):

        data_index = self.label_list
        for i, v in enumerate(self.repres_list):
            data = self.repres_list[i]

            print('processing data')
            X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
            print('processing data over')
            font = {"color": "darkred", "size": 16, "family": "serif"}
            # plt.style.use("dark_background")
            plt.style.use("default")

            plt.figure(300 + i, figsize=(30, 15))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6,
                                  cmap=plt.cm.get_cmap('rainbow', 5),
                                  norm=plt.Normalize(vmin=0, vmax=4))

            class_labels = ['4mC', '5hmC', '6mA', '5mC', '6mA-neg']
            color_legend = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
                                       markerfacecolor=scatter.cmap(scatter.norm(idx)), markersize=10)
                            for idx, cls in enumerate(class_labels)]
            plt.legend(handles=color_legend, loc='lower right', fontsize='large')
            # if data_label:
            #     for i in range(len(X_tsne)):
            #         plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
            #                      xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
            # if data_label is None:
            #     # cbar = plt.colorbar(ticks=range(self.config.num_class))
            #     # cbar.set_label(label='digit value', fontdict=font)
            #     # cbar.set_ticklabels(['4mc', '5hmc', '6ma', '5mc', '6ma-neg'])
            #     # cbar.mappable.set_clim(0 - 0.5, 5 - 0.5)

            path = os.path.join(self.iomanager.result_path, f'{self.config.learn_name}TSNE_{i}'
                                                            f'{str(self.config.kmer)}mer.{self.config.save_figure_type}')
            plt.savefig(path)
            plt.show()

        # data = self.repres_list
        # data_index = self.label_list
        # data_label = None
        # resnet_data = self.resnet
        # print('processing data')
        # X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
        # print('processing data over')
        #
        # font = {"color": "darkred", "size": 16, "family": "serif"}
        # # plt.style.use("dark_background")
        # plt.style.use("default")
        #
        # plt.figure(300, figsize=(30, 15))
        # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 5),
        #                       norm=plt.Normalize(vmin=0, vmax=4))
        #
        # class_labels = ['4mC', '5hmC', '6mA', '5mC', '6mA-neg']
        # color_legend = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
        #                            markerfacecolor=scatter.cmap(scatter.norm(idx)), markersize=10)
        #                 for idx, cls in enumerate(class_labels)]
        # plt.legend(handles=color_legend, loc='lower right', fontsize='large')
        # # if data_label:
        # #     for i in range(len(X_tsne)):
        # #         plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
        # #                      xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
        # # if data_label is None:
        # #     # cbar = plt.colorbar(ticks=range(self.config.num_class))
        # #     # cbar.set_label(label='digit value', fontdict=font)
        # #     # cbar.set_ticklabels(['4mc', '5hmc', '6ma', '5mc', '6ma-neg'])
        # #     # cbar.mappable.set_clim(0 - 0.5, 5 - 0.5)
        #
        # path = os.path.join(self.iomanager.result_path, f'{self.config.learn_name}TSNE'
        #                                                 f'{str(self.config.kmer)}mer.{self.config.save_figure_type}')
        # plt.savefig(path)
        # plt.show()

    def draw_corr(self):

        correlation_matrix = np.corrcoef(self.class_rep)
        labels = ['4mC', '5hmC', '6mA', '5mC']
        sns.heatmap(correlation_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
        path = os.path.join(self.iomanager.result_path,
                            f'{self.config.learn_name}Corr{str(self.config.kmer)}mer.{self.config.save_figure_type}')
        plt.savefig(path)
        plt.show()

    # def draw_resnt_atten(self):
    #
    #     plt.figure(300, figsize=(66, 88))
    #     plt.subplots_adjust(wspace=0.2, hspace=0.3)
    #     normalized_data = np.zeros_like(self.activations)
    #     for i in range(normalized_data.shape[0]):
    #         max_val = np.max(self.activations[i])
    #         min_val = np.min(self.activations[i])
    #         normalized_data[i] = (self.activations[i] - min_val) / (max_val - min_val)
    #
    #     for i in range(len(normalized_data)):
    #         plt.subplot(3, 2, i+1)
    #         plt.imshow(normalized_data[i], cmap='hot', interpolation='nearest')
    #         plt.colorbar()  # Optional: it adds a color bar to the side to show the scale
    #
    #     path = os.path.join(self.iomanager.result_path,
    #                          f'{self.config.label}_{self.config.motif}_{self.config.s}_{self.config.e}_'
    #                         f'ResNet.'
    #                         f'{self.config.save_figure_type}')
    #     plt.savefig(path)
    #     plt.show()
    #
    #     plt.figure(301, figsize=(88, 44))
    #     average_data = np.mean(self.atten, axis=0)
    #     max_val = np.max(average_data)
    #     min_val = np.min(average_data)
    #     average_data = (average_data - min_val) / (max_val - min_val)
    #     sns.heatmap(average_data[0].reshape(1, -1), cmap='YlGnBu', annot=False)
    #
    #     path = os.path.join(self.iomanager.result_path,
    #                         f'{self.config.label}_{self.config.motif}_{self.config.s}_{self.config.e}'
    #                         f'_atten_map_{str(self.config.kmer)}mer.'
    #                         f'{self.config.save_figure_type}')
    #     plt.savefig(path)
    #     plt.show()

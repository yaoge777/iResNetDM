import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve,average_precision_score
import numpy as np
from model import BERT, Focal_loss, Contrastive_loss
from sklearn.preprocessing import label_binarize


# from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

class MCModelManager():
    

    def __init__(self, learner):
        self.iomanager = learner.ioManager
        self.datamanager = learner.dataManager
        self.config = learner.config
        self.visualizer = learner.visualizer
        self.mode = self.config.mode
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.contrastive = None
        self.best_performance = None
        
        self.test_performance = []
        self.valid_performance = []
        self.avg_test_loss = 0
    
    
    #     to be finished
    def init_model(self):
        if self.mode == 'train_test' or self.mode == 'test':
            self.model = BERT.Encoder(self.config)
            if self.config.cuda:
                self.model.cuda()
        
        else:
            self.iomanager.log.Error('no such mode')
    
    def load_params(self):
        if self.config.path_params:
            if self.mode == 'train_test' or self.mode == 'test':
                self.model = self.__load_params(self.model, self.config.path_params)
            
            elif self.mode == 'fine_tune':
                pretrained_dict = torch.load(self.config.path_params)
                model_dict = self.model.state_dict()
                for param in pretrained_dict:
                    if param == 'prj.seq.6.weight':
                        pretrained_dict[param] = torch.tensor((2,512))
                        print(f'{param} paramters upadated')
                    elif param == 'prj.seq.6.bias':
                        pretrained_dict[param] = torch.tensor((2))
                        print(f'{param} paramters upadated')
                        
                # 更新模型字典
                model_dict.update(pretrained_dict)

                # 加载模型参数
                self.model.load_state_dict(model_dict)

                # 冻结前面的一些层
                for param in self.model.parameters():
                    param.requires_grad = False

                # 解冻最后一层（或者新的分类层）
                for param in self.model.prj.seq[6].parameters(): 
                    param.requires_grad = True

            else:
                self.iomanager.log.Error('No such Mode')
        else:
            self.iomanager.log.Warn('Path of parameters not exist')
    
    def __load_params(self, model, param_path):
        pretrained_dict = torch.load(param_path)
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
        return model
    
    def adjust_model(self):
        self.model = self.__adjust_model(self.model)
        
    def __adjust_model(self, model):
        print('-'*50, 'model.named_parameters', '-'*50)
        for name, value in model.named_parameters():
            print(f'[{name}] -> [{value.shape}], [requires_grad:{value.requires_grad}]')
        
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k+ l
        print('='*50, 'number of total parameters' + str(k), '='*50)
        result_path = self.config.path_save + '/' + self.config.learn_name + str(self.config.kmer) + 'mer'
        with open(result_path + '/config_para_nums.txt', 'w') as f:
            k_v_pair = f'number of parameters : {k}'
            f.write(k_v_pair + '\r\n')
        return model
    
    def init_optimizer(self):
        if self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr, weight_decay=self.config.reg)
        elif self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr, weight_decay=self.config.reg) 
        else:
            self.iomanager.log.Error('no such optimizer')
    
    def init_loss_fn(self):
        if self.mode == 'train_test' or self.mode == 'test':
            if self.config.loss_fn == 'CE':
                self.loss_fn = nn.CrossEntropyLoss()
            
            elif self.config.loss_fn == 'FL':
                self.loss_fn = Focal_loss.FocalLoss(num_classes=self.config.num_class, gamma=self.config.gamma)
           
            else:
                self.iomanager.log.Error('No such loss function')
            if self.config.cuda and self.config.contrastive_loss:
                self.contrastive = Contrastive_loss.Constrastive_loss(self.config.temp,
                                                                      self.config.d_model, len(self.config.kmer)).cuda()
        else:
            self.iomanager.log.Error('No such mode')
            
    
    def __get_loss(self, logits, label, rep = None):
        loss = 0
        c_loss = 0
        if self.config.loss_fn == 'CE':
            loss = self.loss_fn(logits.view(-1, self.config.num_class), label.view(-1))
            # loss = (loss.float()).mean()
            # loss = (loss-self.config.b).abs() + self.config.b
        elif self.config.loss_fn == 'FL':
            loss = self.loss_fn(logits.view(-1, self.config.num_class), label.view(-1))

        if rep is not None and self.config.contrastive_loss:
            c_loss = self.contrastive(rep)
            loss += c_loss
        return loss, c_loss
    
    def __SL_train(self, train_dataloader, test_dataloader):
        step = 0
        best_mcc = 0
        best_performance = None
        best_ROC = None
        best_PRC = None
        best_rep = None
        best_label = None
        best_atten = None

        for epoch in range(1, self.config.epoch + 1):
            self.model.train()

            for batch in train_dataloader:

                data, label = batch
                logits, _, _, _, cts = self.model(data)
                train_loss, c_loss = self.__get_loss(logits, label, cts)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                step += 1

                if step % self.config.interval_log == 0:
                    corrects = (torch.argmax(logits, -1) == label).sum()
                    the_batch_size = label.shape[0]
                    train_acc = 100.0 * corrects / the_batch_size
                    print(f'Epoch[{epoch}] Batch[{batch}] - loss: {train_loss:.6f} | c_loss: {c_loss:.6f} |acc: {train_acc:.4f}%({corrects}/{the_batch_size})')

                    self.visualizer.step_log_interval.append(step)
                    self.visualizer.train_metric_record.append(train_acc.cpu().detach().numpy())
                    self.visualizer.train_loss_record.append(train_loss.cpu().detach().numpy())

            if epoch % self.config.interval_test == 0:
                self.model.eval()

#                    test_performance: [acc, sensitivity, specificity, auc, mcc]
                test_performance, avg_test_loss, ROC_data, PRC_data, rep, label, atten = self.__SL_test(test_dataloader)
                mc4_mcc = test_performance['mc4']['accuracy']
                hmc5_mcc = test_performance['hmc5']['accuracy']
                ma6_mcc = test_performance['ma6']['accuracy']
                mc5_mcc = test_performance['mc5']['accuracy']
                neg_mcc = test_performance['neg']['accuracy']

                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(test_performance['accuracy'])
                self.visualizer.class_metric_record.append([mc4_mcc, hmc5_mcc, ma6_mcc, mc5_mcc, neg_mcc])
                self.visualizer.test_loss_record.append(avg_test_loss.cpu().detach().numpy())

                test_mcc = test_performance['accuracy']
                
                if best_mcc < test_mcc < 0.99 and epoch > 0:
                    best_mcc = test_mcc
                    best_performance = test_performance
                    best_ROC = ROC_data
                    best_PRC = PRC_data
                    best_rep = rep
                    best_label = label
                    best_atten = atten
                    if self.config.save_best and best_mcc > self.config.threshold:
                        self.iomanager.save_model_dict(self.model.state_dict(), self.config.model_save_name, 'ACC', best_mcc)
        return best_performance, best_ROC, best_PRC, best_rep, best_label, best_atten

    def __safe_mcc(self, tp, tn, fp, fn):
        # Convert integers to float for higher precision
        tp, tn, fp, fn = map(float, [tp, tn, fp, fn])

        # Calculate the terms in the logarithmic domain to prevent overflow
        terms_to_sum = [
            np.log(tp + fp) if tp + fp > 0 else 0,
            np.log(tp + fn) if tp + fn > 0 else 0,
            np.log(tn + fp) if tn + fp > 0 else 0,
            np.log(tn + fn) if tn + fn > 0 else 0
        ]

        sum_log_terms = sum(terms_to_sum)
        log_numerator = np.log(tp * tn - fp * fn) if tp * tn > fp * fn else np.nan
        log_denominator = sum_log_terms / 2.0

        # Exponentiate the log difference to get the MCC
        mcc_value = np.exp(log_numerator - log_denominator) if not np.isnan(log_numerator) else 0

        return mcc_value
    def __class_accuracy(self, predicted_labels, labels):


        class_accuracies = []
        class_mcc = []
        for positive_class in range(self.config.num_class):
            # 真正类（True Positives, TP）
            TP = np.sum((predicted_labels == positive_class) & (labels == positive_class))

            # 真负类（True Negatives, TN）
            TN = np.sum((predicted_labels != positive_class) & (labels != positive_class))

            # 假正类（False Positives, FP）
            FP = np.sum((predicted_labels == positive_class) & (labels != positive_class))

            # 假负类（False Negatives, FN）
            FN = np.sum((predicted_labels != positive_class) & (labels == positive_class))

            # 计算准确率（Accuracy）
            accuracy = (TP + TN) / float(TP + FP + TN + FN)
            # 计算当前类别的准确率
            mcc = self.__safe_mcc(tp=TP, tn=TN, fp=FP, fn=FN)

            class_accuracies.append(accuracy.item())
            class_mcc.append(mcc)

        return class_accuracies, class_mcc

    def __cal_metrics(self,logits, labels, pred_labels):

        target_names = ['mc4', 'hmc5', 'ma6', 'mc5', 'neg']
        total_metric = classification_report(pred_labels, labels, target_names= target_names,output_dict=True)
        class_acc, class_mcc = self.__class_accuracy(pred_labels, labels)

        for i,v in enumerate(target_names):
            total_metric[v]['accuracy'] = class_acc[i]
            total_metric[v]['mcc'] = class_mcc[i]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.config.num_class):
            fpr[i], tpr[i], _ = roc_curve((labels == i).astype(float), logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        pre = dict()
        rcl = dict()
        ap = dict()
        for i in range(self.config.num_class):
            pre[i], rcl[i], _ = precision_recall_curve((labels == i).astype(float), logits[:, i])
            ap[i] = average_precision_score((labels == i).astype(float), logits[:, i])

        all_rcl = np.unique(np.concatenate([rcl[i] for i in range(self.config.num_class)]))
        mean_pre = np.zeros_like(all_rcl)
        for i in range(self.config.num_class):
            mean_pre += np.interp(all_rcl, rcl[i], pre[i])
        mean_pre /= self.config.num_class
        pre["macro"] = mean_pre
        rcl["macro"] = all_rcl
        labels = label_binarize(labels, classes=[0, 1, 2, 3, 4])
        ap['macro'] = average_precision_score(labels, logits, average="macro")
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.config.num_class)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.config.num_class):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= self.config.num_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # total_metric: {0:{precision, recall, f1-score, support}, 1{}, 2:{}, 3{}, macro avg:{}, weighted avg:{}, accuracy:float}, 
        # fpr, tpr:{0:[], 1:[], 2:[], 3[], macro:[]}, 
        # roc_auc:{0:float, macro:int}, 
        # pre,rcl:{0:[], 1:[], 2:[], 3[], macro:[]}
        # ap:{0:float, macro:int}, 

        return total_metric, [fpr,tpr,roc_auc], [pre, rcl, ap]
    
    def __SL_test(self, dataloader):
        corrects = 0
        test_batch_num = 0
        test_sample_num = 0
        avg_test_loss = 0
        label_list = []
        logits_list = []
        pred_labels = []
        step = 0
        atten_list = []
        rep_list = [[], [], [], []]
        with torch.no_grad():
            for batch in dataloader:
                step += 1
                data, label = batch
                logits, activations, atten, rep, _ = self.model(data)
                avg_test_loss += self.__get_loss(logits, label)[0]
                for i, v in enumerate(rep_list):
                    v.extend(np.array(rep[i]))
                label_list.extend(label.cpu().detach().numpy())
                logits_list.extend(logits.cpu().detach().numpy())
                pred_labels.extend(logits.argmax(dim=-1).cpu().detach().numpy())
                if atten is not None:
                    atten_list.extend(atten.cpu().detach().numpy())
                corrects += (torch.argmax(logits,-1) == label).sum()
                test_batch_num += 1
    #                 pred_prob_all = F.softmax(logits, dim=1) #[batch_size, class_num]
    #                 pred_prob_positive = pred_prob_all[:, 1]
    #                 pred_prob_sort = torch.max(pred_prob_all, 1) #[batch_size]
    #                 pred_class = pred_prob_sort[1] #seems wrong

                # total_metric: {0:{precision, recall, f1-score, support}, 1{}, 2:{}, 3{}, macro avg:{}, weighted avg:{}, accuracy:float}, 
                # fpr, tpr:{0:[], 1:[], 2:[], 3[], macro:[]}, 
                # roc_auc:{0:float, macro:int}, 
                # pre,rcl:{0:[], 1:[], 2:[], 3[], macro:[]}
        logits_list = np.array(logits_list)
        label_list = np.array(label_list)
        pred_labels = np.array(pred_labels)
        atten_list = np.array(atten_list)
        rep_list = np.array(rep_list)
        performance, ROC_data, PRC_data = self.__cal_metrics(logits_list, label_list, pred_labels)
        test_sample_num += len(label_list)
                


        avg_test_loss /= test_batch_num
        avg_acc = 100.0 * performance['accuracy']
        print(f'Evaluation - loss: {avg_test_loss:.6f} ACC: {avg_acc:.4f} % ({corrects} / {test_sample_num})')

        self.avg_test_loss = avg_test_loss
        return performance, avg_test_loss, ROC_data, PRC_data, rep_list, label_list, atten_list

    def generate_class_rep(self, rep, label):

        averaged_data = np.zeros((self.config.num_class-1, self.config.d_model))

        for i in range(self.config.num_class-1):
            # 选择属于当前类别的数据
            class_data = rep[label == i, :]
            # 计算平均值
            averaged_data[i, :] = class_data.mean(axis=0)

        return averaged_data

    def test_motif(self, dataloader):
        corrects = 0
        test_batch_num = 0
        test_sample_num = 0
        avg_test_loss = 0
        step = 0
        label_list = []
        pred_labels = []
        with torch.no_grad():
            for batch in dataloader:
                step += 1
                data, label = batch
                logits, _, _, _, _, _ = self.model(data)
                label_list.extend(label.cpu().detach().numpy())
                pred_labels.extend(logits.argmax(dim=-1).cpu().detach().numpy())

        #                 pred_prob_all = F.softmax(logits, dim=1) #[batch_size, class_num]
        #                 pred_prob_positive = pred_prob_all[:, 1]
        #                 pred_prob_sort = torch.max(pred_prob_all, 1) #[batch_size]
        #                 pred_class = pred_prob_sort[1] #seems wrong

        # total_metric: {0:{precision, recall, f1-score, support}, 1{}, 2:{}, 3{}, macro avg:{}, weighted avg:{}, accuracy:float},
        # fpr, tpr:{0:[], 1:[], 2:[], 3[], macro:[]},
        # roc_auc:{0:float, macro:int},
        # pre,rcl:{0:[], 1:[], 2:[], 3[], macro:[]}
        label_list = np.array(label_list)
        pred_labels = np.array(pred_labels)
        accuracy = np.mean(label_list == pred_labels)
        return accuracy

    def train(self):
        
        if self.config.mode == 'train_test':
            train_dataloader = self.datamanager.get_dataloader(name='train_set')
            test_dataloader = self.datamanager.get_dataloader(name='test_set')
            best_performance, best_ROC, best_PRC, rep, label, atten = self.__SL_train(train_dataloader, test_dataloader)
            average_atten = np.mean(atten, axis=0)
            self.best_performance = best_performance
            self.visualizer.repres_list = rep
            self.visualizer.class_rep = self.generate_class_rep(rep[-1], label)
            self.visualizer.label_list = label
            self.visualizer.roc_data = best_ROC
            self.visualizer.prc_data = best_PRC
            self.visualizer.atten = average_atten

            self.iomanager.log.Info(f'Performance: {self.best_performance}')      

    def test(self):

        test_dataloader = self.datamanager.get_dataloader(name='test_set')
        acc = self.test_motif(test_dataloader)

        # average_atten = np.mean(atten, axis=0)
        # self.best_performance = best_performance
        # self.visualizer.repres_list = rep
        # self.visualizer.class_rep = self.generate_class_rep(rep, label)
        # self.visualizer.label_list = label
        # self.visualizer.roc_data = best_ROC
        # self.visualizer.prc_data = best_PRC
        # self.visualizer.atten = average_atten


        self.iomanager.log.Info(f'test data over, acc: {acc, self.config.mask_inside}')
        return acc
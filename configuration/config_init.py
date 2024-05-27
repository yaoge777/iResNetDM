
import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    # project parameters
    parse.add_argument('-learn_name', type=str, default='Resnet_atten_5_2_256_FL_newset', help='本次训练名称')

    parse.add_argument('-path_save', type=str, default='../result/', help='保存字典的位置')
    parse.add_argument('-save_best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-cuda_num', type=int, default=[3, 2], nargs='+', help='设置cuda的编号')
    parse.add_argument('-threshold', type=float, default=0.75, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    # parse.add_argument('-cuda', type=bool, default=False)


    # TODO  change kmer
    parse.add_argument('-kmer', type=int, nargs='+', default=[1, 3])
    parse.add_argument('-kernel_size', type=int, default=7)


    # save path
    parse.add_argument('-train-name', type=str, help='-train-name')
    parse.add_argument('-test-name', type=str, help='-test-name')

    parse.add_argument('-path_train_data1', type=str, default='../data/test.txt', help='训练数据1的位置')
    parse.add_argument('-path_train_data', type=str, default='../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv',
                       help='训练数据2的位置')

    parse.add_argument('-path_test_data', type=str, default='../data/test.txt', help='训练数据的位置')

    parse.add_argument('-path_params', type=str, default='D:\project\DNApred_ResNet\\result\Resnet_atten_5_2_256_FL_newset[1, 3]mer\Resnet_atten_5_2_256 _FL_newset[1, 3]merBERT, ACC[0.756].pt', help='模型参数路径')  # 这里是没有加载模型参数的
    parse.add_argument('-model_save_name', type=str, default='BERT', help='保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default='png', help='保存图片的文件类型')

    # training
    parse.add_argument('-mode', type=str, default='train_test', help='共分为几类')
    parse.add_argument('-num_class', type=int, default=5, help='共分为几类')


    # TODO modify model type
    # parse.add_argument('-model', type=str, default='ClassificationDNAbert', help='训练模型名称')
    parse.add_argument('-model', type=str, default='BERT', help='训练模型名称')
    #
    parse.add_argument('-interval_log', type=int, default=10, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval_test', type=int, default=1, help='经过多少epoch对测试集进行测试')

    parse.add_argument('-epoch', type=int, default=1, help='迭代次数')
    # parse.add_argument('-optimizer', type=str, default='Adam', help='优化器名称')
    parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-loss_fn', type=str, default='FL', help='损失函数名称, CE/FL')
    parse.add_argument('-contrastive_loss', type=bool, default=False, help='whether to use contrastive loss')


    # TODO change batchSize
    parse.add_argument('-word_num', type=int, default=5)
    parse.add_argument('-d_model', type=int, default=256)
    parse.add_argument('-layers_num', type=int, default=2)
    parse.add_argument('-resnet_layer', type=int, default=5)
    parse.add_argument('-lstm_layers', type=int, default=1)
    parse.add_argument('-heads_num', type=int, default=8)
    parse.add_argument('-seq_num', type=int, default=42)
    parse.add_argument('-hidden', type=int, default=256)
    parse.add_argument('-dropout', type=int, default=0.2)
    parse.add_argument('-ff', type=str, default='linear')
    parse.add_argument('-rep', type=str, default='default')
    parse.add_argument('-batch_size', type=int, default=128)
    parse.add_argument('-lr', type=float, default=0.00005)
    parse.add_argument('-growth_rate', type=int, default=2)

    parse.add_argument('-reg', type=float, default=0.0, help='正则化lambda')
    parse.add_argument('-b', type=float, default=0.06, help='flooding model')
    parse.add_argument('-gamma', type=float, default=2, help='gamma in Focal Loss')
    parse.add_argument('-temp', type=float, default=1, help='temperature in Contrastive Loss')

    # parse.add_argument('-alpha', type=float, default=0.01, help='alpha in Hinge Loss')

    parse.add_argument('-label', type=int, default=0)
    parse.add_argument('-motif', type=str, default='A')
    parse.add_argument('-mask_inside', type=bool, default=True)
    parse.add_argument('-mask_num', type=int, default=1)
    parse.add_argument('-s', type=int, default=20)
    parse.add_argument('-e', type=int, default=21)

    config = parse.parse_args()
    return config
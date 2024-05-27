from frame import DataManager, IOManager, ModelManager, Visualizer

class Learner():
    
    def __init__(self, config):
        self.config = config
        self.ioManager = IOManager.IOManager(self)
        self.visualizer = Visualizer.Visualizer(self)
        self.dataManager = DataManager.DataManager(self)
        self.modelManager = ModelManager.MCModelManager(self)
    
    def init_io(self):
        self.ioManager.initialize()
        self.ioManager.log.Info('io initialized')
        
    def init_vis(self):
        self.visualizer.initialize()
        self.ioManager.log.Info('vis initialized')
    
    def load_data(self):
        self.dataManager.load_traindata()
        self.dataManager.load_testdata()
        self.ioManager.log.Info('dataset loaded')

    def load_test_data(self):
        self.dataManager.load_testdata()
        self.ioManager.log.Info('dataset loaded')
    def init_model(self):
        self.modelManager.init_model()
        self.ioManager.log.Info('model loaded')
    
    def load_params(self):
        self.modelManager.load_params()
        self.ioManager.log.Info('param loaded')
    
    def adjust_model(self):
        self.modelManager.adjust_model()
        self.ioManager.log.Info('model adjusted')
    
    def init_optimizer(self):
        self.modelManager.init_optimizer()
        self.ioManager.log.Info('optimizer initialized')
    
    def init_loss_fn(self):
        self.modelManager.init_loss_fn()
        self.ioManager.log.Info('loss_fn initialized')
    
    def train_model(self):
        self.ioManager.log.Info('train model start')
        self.ioManager.log.Info(f'learn name: {self.config.learn_name}')
        self.ioManager.log.Info(f'config: {self.config}')
        self.modelManager.train()
        self.visualizer.draw_train_test_curve()
        self.visualizer.draw_ROC_PRC_curve()
        self.visualizer.draw_seperate_curve()
        self.visualizer.draw_tsne()
        self.visualizer.draw_atten()
        self.visualizer.draw_partial_atten()
        self.visualizer.draw_corr()
        self.ioManager.log.Info('train model over')
        
    def test_model(self):
        self.ioManager.log.Info('test model start')

        acc = self.modelManager.test()
        # self.visualizer.draw_resnt_atten()
        self.ioManager.log.Info('test model over')
        return acc
    
    def reset_iomanager(self):
        self.ioManager = IOManager.IOManager(self)
        self.ioManager.initialize()
        self.ioManager.log.Info('Reset Visualizer Over.')
        
    def reset_visualizer(self):
        self.visualizer = Visualizer.Visualizer(self)
        self.visualizer.initialize()
        self.ioManager.log.Info('Reset Visualizer Over.')

    def reset_dataManager(self):
        self.dataManager = DataManager.DataManager(self)

        self.ioManager.log.Info('Reset DataManager Over.')

    def resset_modelManager(self):
        self.modelManager = ModelManager.ModelManager(self)
        self.modelManager.init_model()
        self.ioManager.log.Info('Reset ModelManager Over.')

import torch


class BertConfig(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Bert'

        self.save_path = './model/' + self.model_name + '.pkl'  # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备

        self.seed = 721
        self.dropout = 0.5  # 随机失活
        self.early_stop = 10  # 早停机制
        self.num_classes = 5  # 类别数
        self.vocab_size = 652  # 词表大小，在运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 32  # batch大小
        self.max_len = 40  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.out_dim = 64  # 输出维度

class OtherConfig(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Lstm'

        self.save_path = './model/' + self.model_name + '.pkl'  # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备

        self.seed = 114514
        self.dropout = 0.5  # 随机失活
        self.early_stop = 10  # 早停机制
        self.num_classes = 5  # 类别数
        self.vocab_size = 652  # 词表大小，在运行时赋值
        self.num_epochs = 100  # epoch数
        self.batch_size = 32  # batch大小
        self.max_len = 20  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.out_dim = 64  # 输出维度
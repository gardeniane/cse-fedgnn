from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:  # 以读（‘r’）的方式打开目录为config（第一个）的文件内容以字符串的形式保存在config（第二个）中
            self.config = json.load(config)  # 以json的格式解析字符串config
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Clients --
        fields = ['total', 'per_round', 'data_partition',
                  'do_test', 'noniid_degree']
        defaults = (0, 0, 'label-based', False, 0)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total  # 一个round的clients小于等于总数时执行下面的语句，否则抛出异常

        # -- 读取dataset name以及对应的training parameters --
        self.dataset_name = config['dataset_name']
        fields = ['num_layers', 'hidden_channels', 'dropout', 'drop_input',
                  'batch_norm', 'residual', 'num_parts', 'batch_size',
                  'max_steps', 'pool_size', 'num_workers', 'lr',
                  'reg_weight_decay', 'nonreg_weight_decay', 'grad_norm',
                  'epochs', 'target_accuracy']
        defaults = (0, 0, 0, False, False, False, 0, 0, 0, 0, 0, 0, 0.0,
                    0.0, None, 0, 0.0)
        params = [config[self.dataset_name].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple(self.dataset_name, fields)(*params)

        # -- Federated learning --
        fields = ['iterations', 'task', 'stages']
        defaults = (0, 'train', 1)
        params = [config['FL'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('FL', fields)(*params)

        # -- Paths --
        fields = ['data', 'model', 'reports']
        defaults = ('./data', './results', None)
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        # Set specific model path
        params[fields.index('model')] += '/' + self.dataset_name
        self.paths = namedtuple('paths', fields)(*params)

        # -- Clusters --
        self.clusters = config['clusters']

        # -- training mode --
        self.mode = config['training_mode']

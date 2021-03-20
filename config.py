from dataclasses import dataclass, asdict

L1 = 'l1'
L2 = 'l2'
SMOOTH_L1 = 'smooth_l1'

MOBILENET_V2 = 'mobilenet_v2_skip_add'
MOBILENET = 'mobilenet_skip_add'

SGD = 'sgd'
ADAM = 'adam'

@dataclass
class Config(object):
    arch: str
    epochs: int
    seed: int
    loss_function: str
    batch_size: int
    lr: float
    momentum: float
    optimizer: str
    b1: float
    b2: float
    weight_decay: float
    datapath: str
    subset_datapath:str
    train_subset = False

    def __init__(self):
        self.arch = 'mobilenet_v2_skip_add'
        self.epochs = 100
        self.seed = 13
        self.loss_function = SMOOTH_L1
        self.batch_size = 64 # 128 works better if you have the GPU capacity
        self.lr = 0.0804
        self.momentum = 0.9
        self.optimizer = 'sgd'
        self.b1 = 0.9 # For adam optimizer
        self.b2 = 0.999
        self.weight_decay = 1e-4
        self.datapath = '/home/deman/dev/data/nyudepthv2/'
        self.subset_datapath = '/home/deman/dev/data/nyu_depth_v2_labeled.mat'
        self.train_subset = False
        self.checkpoint_dir = 'model_checkpoints'
        self.modelname_prefix = 'model'

    def write(self, path=None):
        import json
        path = path if path is not None else 'configuration.json'
        s = json.dumps(asdict(self))
        with open(path, 'w') as f:
            f.write(s)
